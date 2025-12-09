import cv2
import numpy as np
import torch
from PIL import Image
import ffmpeg

def load_image(path, target_size=None):
    img = Image.open(path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img)

def save_video_ffmpeg(frames, output_path, fps=30):
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=fps)
        .output(output_path, pix_fmt='yuv420p', vcodec='libx264', crf=18)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    
    process.stdin.close()
    process.wait()

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0) # C, H, W -> H, W, C

def numpy_to_tensor(array):
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0 # H, W, C -> C, H, W
