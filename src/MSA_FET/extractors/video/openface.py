import os
import platform
import subprocess
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..baseExtractor import baseVideoExtractor


class openfaceExtractor(baseVideoExtractor):
    """
    Video feature extractor using OpenFace. 
    Ref: https://mediapipe.dev/
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing OpenFace video feature extractor...")
            super().__init__(config, logger)
            if self.config['average_over'] < 1:
                self.pool_size = 1
                logger.warning("'average_over' is less than 1, set to 1.")
            else:
                self.pool_size = self.config['average_over']
                
            self.args = self._parse_args(self.config['args'])
            self.tool_dir = Path(__file__).parent.parent.parent / "exts" / "OpenFace"
            if platform.system() == 'Windows':
                self.tool = self.tool_dir / "FeatureExtraction.exe"
            elif platform.system() == 'Linux':
                self.tool = self.tool_dir / "FeatureExtraction"
            else:
                raise RuntimeError("Cannot Determine OS type.")
            if not self.tool.is_file():
                raise FileNotFoundError("OpenFace tool not found.")
        except Exception as e:
            self.logger.error("Failed to initialize OpenFaceExtractor.")
            raise e

    def _parse_args(self, args):
        res = []
        if 'hogalign' in args and args['hogalign']:
            res.append('-hogalign')
        if 'simalign' in args and args['simalign']:
            res.append('-simalign')
        if 'nobadaligned' in args and args['nobadaligned']:
            res.append('-nobadaligned')
        if 'tracked' in args and args['tracked']:
            res.append('-tracked')
        if 'pdmparams' in args and args['pdmparams']:
            res.append('-pdmparams')
        if 'landmark_2D' in args and args['landmark_2D']:
            res.append('-2Dfp')
        if 'landmark_3D' in args and args['landmark_3D']:
            res.append('-3Dfp')
        if 'head_pose' in args and args['head_pose']:
            res.append('-pose')
        if 'action_units' in args and args['action_units']:
            res.append('-aus')
        if 'gaze' in args and args['gaze']:
            res.append('-gaze')
        return res
    
    def extract(self, img_dir, video_name=None, tool_output=False):
        """
        Function:
            Extract features from video file using OpenFace.

        Parameters:
            img_dir: path to directory of images.
            video_name: video name used to save annotation images.
            tool_output: if False, disable stdout of OpenFace tool.

        Returns:
            video_result: extracted video features in numpy array.
        """
        try:
            self.img_dir = Path(img_dir)
            self.num_imgs = len(glob(str(self.img_dir / "*.bmp")))
            args = self.args.copy()
            args.extend(['-fdir', str(img_dir), '-out_dir', str(img_dir)])
            if not tool_output:
                args.append('-quiet')
            cmd = str(self.tool) + " " + " ".join(args)
            os.system(cmd)

            name = str(Path(img_dir).stem)
            df = pd.read_csv(Path(img_dir) / (name + '.csv'))
            features, local_features = [], []
            for i in range(len(df)):
                local_features.append(np.array(df.loc[i][df.columns[5:]]))
                if (i + 1) % self.pool_size == 0:
                    features.append(np.array(local_features).mean(axis=0))
                    local_features = []
            if len(local_features) != 0:
                features.append(np.array(local_features).mean(axis=0))
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Failed to extract video features with OpenFace from {video_name}.")
            raise e

    def get_feature_names(self) -> list:
        pass

    def get_timestamps(self):
        fps = self.config['fps']
        return np.arange(0, self.num_imgs / fps, 1 / fps)

    @staticmethod
    def get_annotated_video(input : str, output : str) -> bytes:
        try:
            tool = Path(__file__).parent.parent.parent / "exts" / "OpenFace" / "FeatureExtraction"
            cmd = str(tool) + f" -f {input} -of {output} -oc avc1 -tracked" # default settings
            args = cmd.split()
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            if p.returncode != 0: # BUG: OpenFace won't set return code >0 on error
                raise RuntimeError("openface", out, err)
            return out
        except Exception as e:
            raise e
