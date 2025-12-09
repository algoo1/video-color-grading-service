import torch
import os
import sys
import yaml
import logging
from optimization import optimizer

logger = logging.getLogger(__name__)

# Add the repository to path
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "VideoColorGrading"))
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

class ModelManager:
    def __init__(self, config_path=None):
        self.device = optimizer.device
        self.dtype = optimizer.dtype
        self.gs_extractor = None
        self.l_diffuser = None
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_models(self, checkpoint_dir="pretrained"):
        """
        Loads the GS-Extractor and L-Diffuser models.
        Assumes the VideoColorGrading repo structure is available.
        """
        logger.info(f"Loading models from {checkpoint_dir}...")
        
        try:
            # Attempt to import from the repository
            # These imports are based on the paper and typical structure
            # Adjust these if the actual repo has different file names
            from models.gs_extractor import GSExtractor
            from models.l_diffuser import LDiffuser
            from utils.util import load_checkpoint # Assuming utility exists
        except ImportError as e:
            logger.error(f"Failed to import model classes: {e}")
            logger.error("Ensure the 'VideoColorGrading' repo is cloned and in the python path.")
            raise

        # Load GS-Extractor
        try:
            self.gs_extractor = GSExtractor() # Initialize with config if needed
            gs_ckpt = os.path.join(checkpoint_dir, "gs_extractor.pth")
            if os.path.exists(gs_ckpt):
                state_dict = torch.load(gs_ckpt, map_location="cpu")
                self.gs_extractor.load_state_dict(state_dict)
            self.gs_extractor = optimizer.optimize_model(self.gs_extractor)
            logger.info("GS-Extractor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading GS-Extractor: {e}")
            raise

        # Load L-Diffuser
        try:
            self.l_diffuser = LDiffuser() # Initialize with config if needed
            ld_ckpt = os.path.join(checkpoint_dir, "l_diffuser.pth")
            if os.path.exists(ld_ckpt):
                state_dict = torch.load(ld_ckpt, map_location="cpu")
                self.l_diffuser.load_state_dict(state_dict)
            self.l_diffuser = optimizer.optimize_model(self.l_diffuser)
            logger.info("L-Diffuser loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading L-Diffuser: {e}")
            raise

        return self.gs_extractor, self.l_diffuser

    def get_models(self):
        if self.gs_extractor is None or self.l_diffuser is None:
            return self.load_models()
        return self.gs_extractor, self.l_diffuser

# Global instance
model_manager = ModelManager()
