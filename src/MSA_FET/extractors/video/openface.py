import os
# import time
# from ctypes import *
from pathlib import Path

import numpy as np
import pandas as pd

from ..baseExtractor import baseExtractor


class openfaceExtractor(baseExtractor):
    """
    Video feature extractor using OpenFace. 
    Ref: https://mediapipe.dev/
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing OpenFace video feature extractor...")
            super().__init__(config, logger)
            self.args = self._parse_args(self.config['args'])
            # self.cur_dir = os.getcwd()
            self.tool_dir = Path(__file__).parent.parent.parent / "exts" / "OpenFace"
            # self.libpath = self.tool_dir / "libOpenFaceExtractor.so"
            # self.c_module = cdll.LoadLibrary(self.libpath)
        except Exception as e:
            self.logger.error("Failed to initialize mediapipeExtractor.")
            raise e

    def _parse_args(self, args):
        self.pool_size = self.config['args']['average_over']
        assert self.pool_size > 0, "Pool size must be greater than 0."
        # res = ['0'] # the first arg is reserved according to openface FeatureExtraction.exe
        res = []
        if 'hogalign' in args and args['hogalign']:
            res.append('-hogalign')
        if 'simalign' in args and args['simalign']:
            res.append('-simalign')
        if 'nobadaligned' in args and args['nobadaligned']:
            res.append('-nobadaligned')
        if 'track' in args and args['track']:
            res.append('-track')
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
            args = self.args.copy()
            args.extend(['-fdir', img_dir, '-out_dir', img_dir])
            if not tool_output:
                args.append('-quiet')
            cmd = str(self.tool_dir) + "/FeatureExtraction " + " ".join(args)
            # start = time.time()
            os.system(cmd)
            # end = time.time()
            # print("time used: ", end - start)

            # n_args = len(args)
            # arg_type = (c_char_p * n_args)
            # arg_array = arg_type()
            # for i, v in enumerate(args):
            #     arg_array[i] = v.encode('utf-8')
            # self.c_module.extract.argtypes = [c_int, arg_type]

            # os.chdir(self.tool_dir)
            # start = time.time()
            # return_status = self.c_module.extract(n_args, arg_array)
            # end = time.time()
            # print("time used: ", end - start)
            # os.chdir(self.cur_dir)

            # if return_status:
            #     raise RuntimeError("C library call failed.")

            name = Path(img_dir).stem
            df = pd.read_csv(Path(img_dir) / (str(name) + '.csv'))
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
