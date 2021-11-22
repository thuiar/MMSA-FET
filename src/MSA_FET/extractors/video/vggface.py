import os
import os.path as osp
from glob import glob
from pathlib import Path

import cv2
import torch
import torch.nn as nn

from ...models import Senet50_ft_dag
from ...utils import download_file
from ..baseExtractor import baseAudioExtractor, baseExtractor


class vggfaceExtractor(baseAudioExtractor):
    """
    Video feature extractor using pretrained model from VGGFace2. 
    Ref: https://www.robots.ox.ac.uk/~albanie/pytorch-models.html
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing VGGFace2 video extractor...")
            super().__init__(config, logger)
            self.model = Senet50_ft_dag()
            save_path = Path.home() / '.MSA-FET' / 'pretrained'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            url = "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/senet50_ft_dag.pth"
            if not osp.isfile(osp.join(save_path, 'senet50_ft_dag.pth')):
                download_file(url, save_path / 'senet50_ft_dag.pth')
            self.model.load_state_dict(torch.load(save_path / 'senet50_ft_dag.pth'))
            self.model.eval()
        except Exception as e:
            self.logger.error("Failed to initialize vggfaceExtractor.")
            os.remove(osp.join(save_path, 'senet50_ft_dag.pth'))
            raise e

    def extract(self, img_dir, video_name=None):
        """
        Function:
            Extract features from video file using VGGFace2 pretrained model.

        Parameters:
            img_dir: path to directory of images.

        Returns:
            video_result: extracted video features in numpy array.
        """
        try:
            for image_path in sorted(glob(osp.join(img_dir, '*.bmp'))):
                name = Path(image_path).stem
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image)
                self.model(image)
        except Exception as e:
            self.logger.error("Failed to extract video features with VGGFace2 pretrained model.")
            raise e
