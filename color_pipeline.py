import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
import logging
from decord import VideoReader, cpu, gpu
from PIL import Image
from tqdm import tqdm

from model_loader import model_manager
from optimization import optimizer
import utils

logger = logging.getLogger(__name__)

class TrilinearLUT(torch.nn.Module):
    def __init__(self, lut_size=33):
        super().__init__()
        self.lut_size = lut_size

    def forward(self, img, lut):
        """
        img: (B, 3, H, W) float32 [0, 1]
        lut: (B, 3, D, H, W) or (1, 3, D, H, W) - usually D=H=W=lut_size
             Note: Torch grid_sample expects (B, C, D, H, W)
             But we usually have LUT as (B, 3, S, S, S).
             We need to treat 'img' as coordinates to sample from 'lut'.
        """
        # img serves as the grid for sampling
        # grid_sample expects coordinates in [-1, 1]
        # img is [0, 1], so we transform to [-1, 1]
        grid = img.permute(0, 2, 3, 1).unsqueeze(1) # (B, 1, H, W, 3)
        grid = (grid * 2) - 1.0
        
        # LUT needs to be reshaped/permuted to (B, 3, S, S, S)
        # If LUT is a single tensor for the batch, expand it
        if lut.shape[0] == 1 and img.shape[0] > 1:
            lut = lut.expand(img.shape[0], -1, -1, -1, -1)
            
        # Sample
        # grid has shape (B, 1, H, W, 3) -> 3 corresponds to x,y,z coordinates
        # output will be (B, 3, 1, H, W)
        out = F.grid_sample(lut, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return out.squeeze(2) # (B, 3, H, W)

class ColorPipeline:
    def __init__(self):
        self.models_loaded = False
        self.lut_applier = TrilinearLUT().to(optimizer.device)
        self.lut_applier = optimizer.optimize_model(self.lut_applier)

    def load_resources(self):
        if not self.models_loaded:
            self.gs_extractor, self.l_diffuser = model_manager.get_models()
            self.models_loaded = True

    def process_video(self, 
                      video_path, 
                      ref_image_path=None, 
                      quality_mode="balanced", 
                      stabilization=True,
                      output_resolution="auto",
                      save_path="output.mp4"):
        
        self.load_resources()
        
        # 1. Decode Video
        logger.info(f"Processing video: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Determine Batch Size based on quality/VRAM
        batch_size = 16 if quality_mode == "fast" else 8
        if quality_mode == "high":
            batch_size = 4
            
        # 2. Prepare Reference
        ref_features = self._prepare_reference(ref_image_path, vr)
        
        # 3. Generate LUT (Temporal Aware if needed)
        # For simplicity and speed, we generate one LUT for the scene or per batch
        # If stabilization is True, we generate a base LUT and apply it globally or smooth it
        # The paper mentions "Look-Up Table Generation". 
        # We will generate a LUT based on the reference and the *content* of the video.
        
        # To maintain temporal stability, it's best to generate a GLOBAL LUT for the whole clip 
        # (or scene) rather than per-frame, unless the scene changes drastically.
        # We'll use a representative frame (middle of video) + reference to generate the LUT.
        
        middle_frame_idx = total_frames // 2
        content_frame = vr[middle_frame_idx].asnumpy()
        content_tensor = utils.numpy_to_tensor(content_frame).unsqueeze(0).to(optimizer.device)
        
        logger.info("Generating LUT...")
        with torch.no_grad(), optimizer.get_autocast_context():
            # This is a placeholder for the actual inference call
            # We assume l_diffuser takes (content, style_features) -> LUT
            lut = self._generate_lut(content_tensor, ref_features)
            
        # 4. Process Frames
        logger.info(f"Applying grading to {total_frames} frames...")
        processed_frames = []
        
        for i in tqdm(range(0, total_frames, batch_size)):
            # Load batch
            batch_indices = range(i, min(i + batch_size, total_frames))
            batch_frames = vr.get_batch(batch_indices).asnumpy()
            
            # Preprocess
            batch_tensor = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float() / 255.0 # B, C, H, W
            batch_tensor = batch_tensor.to(optimizer.device)
            
            # Downscale if needed for speed (processing resolution)
            orig_H, orig_W = batch_tensor.shape[2], batch_tensor.shape[3]
            proc_tensor = batch_tensor
            if quality_mode == "fast":
                proc_tensor = F.interpolate(batch_tensor, scale_factor=0.5, mode='bilinear')
            
            # Apply LUT
            with torch.no_grad(), optimizer.get_autocast_context():
                graded_tensor = self.lut_applier(proc_tensor, lut)
            
            # Upscale back if downscaled
            if quality_mode == "fast":
                graded_tensor = F.interpolate(graded_tensor, size=(orig_H, orig_W), mode='bilinear')
                
            # Post-processing (Tone mapping, exposure - simplified)
            # In a real pipeline, we might refine this. 
            # Here we assume the LUT handles the look.
            
            # Convert back to uint8
            graded_frames = (graded_tensor.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).byte().cpu().numpy()
            processed_frames.extend(graded_frames)

        # 5. Save Video
        logger.info(f"Saving video to {save_path}...")
        utils.save_video_ffmpeg(processed_frames, save_path, fps=fps)
        
        return save_path

    def _prepare_reference(self, ref_path, video_reader):
        if ref_path and os.path.exists(ref_path):
            logger.info(f"Using reference image: {ref_path}")
            ref_img = utils.load_image(ref_path)
            ref_tensor = utils.numpy_to_tensor(ref_img).unsqueeze(0).to(optimizer.device)
        else:
            logger.info("No reference provided. Using auto-grading (self-reference).")
            # Use the middle frame as "style" reference (auto-enhance)
            # Or use a default style vector if the model supports it
            mid_idx = len(video_reader) // 2
            ref_img = video_reader[mid_idx].asnumpy()
            ref_tensor = utils.numpy_to_tensor(ref_img).unsqueeze(0).to(optimizer.device)
            
        # Extract features using GS-Extractor
        with torch.no_grad(), optimizer.get_autocast_context():
            # Assume gs_extractor(image) -> features
            # Verify method name in actual repo
            if hasattr(self.gs_extractor, 'extract_features'):
                 return self.gs_extractor.extract_features(ref_tensor)
            else:
                # Fallback: assume it's a callable
                return self.gs_extractor(ref_tensor)

    def _generate_lut(self, content, style_features):
        # Generate LUT using L-Diffuser
        # Assume l_diffuser(content, style) -> lut_weights or lut_volume
        # The output should be a 3D LUT (3, 33, 33, 33) or similar
        
        if hasattr(self.l_diffuser, 'generate_lut'):
            lut = self.l_diffuser.generate_lut(content, style_features)
        else:
            lut = self.l_diffuser(content, style_features)
            
        # Reshape to (1, 3, D, H, W) if needed
        # Assuming standard 33^3 LUT flattened or direct
        if lut.dim() == 2: # flattened
            dim = int(round((lut.shape[1] // 3) ** (1/3)))
            lut = lut.view(1, 3, dim, dim, dim)
        
        return lut

# Global Pipeline
pipeline = ColorPipeline()
