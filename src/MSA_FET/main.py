import json
import logging
import os
import os.path as osp
import pickle
from glob import glob
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from tqdm import tqdm

from .extractors import *
from .utils import *

AUDIO_EXTRACTOR_MAP = {
    "librosa": librosaExtractor,
    "opensmile": opensmileExtractor,
    "wav2vec": wav2vec2Extractor,
}

VIDEO_EXTRACTOR_MAP = {
    "mediapipe": mediapipeExtractor,
    "vggface": vggfaceExtractor,
    # "3dcnn": 3dcnn_extractor,
}

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--dataset-mode', action='store_true',
#                         help="Switch to dataset mode if specified.")
#     parser.add_argument('-i', '--input-file', type=str, required=True,
#                         help="Path to input file, or dataset name in dataset mode.")
#     parser.add_argument('-t', '--text-file', type=str, required=False,
#                         help="Path to text file, will be ignored in dataset mode. \
#                              If omitted, speech recognition api will be used to generate text.")
#     parser.add_argument('-c', '--config-file', type=str, required=True,
#                         help="Path to config file.")
#     parser.add_argument('-o', '--output-file', type=str, required=True,
#                         help="Path to output pkl file.")
#     parser.add_argument('-v', '--verbose', action='store_true',
#                         help="Print more information to stdout.")
#     parser.add_argument('-q', '--quiet', action='store_true',
#                         help="Print only errors to stdout.")
#     return parser.parse_args()


class FeatureExtractionTool(object):
    """
    Feature Extraction Tool for Multimodal Sentiment Analysis tasks.

    Parameters:
        config_file: 
            Path to config file.
        dataset_root_dir: 
            Path to dataset root directory. Required when extracting dataset features.
        tmp_dir: 
            Temporary directory path. Default: '~/.MSA-FET/tmp'.
        log_dir: 
            Log directory path Default: '~/.MSA-FET/log'.
        verbose: 
            Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1.
    """

    def __init__(
        self,
        config_file,
        dataset_root_dir=None,
        tmp_dir=osp.join(Path.home(), '.MSA-FET/tmp'),
        log_dir=osp.join(Path.home(), '.MSA-FET/log'),
        verbose=1
    ):
        self.config_file = config_file
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        self.tmp_dir = tmp_dir
        self.log_dir = log_dir
        self.dataset_root_dir = dataset_root_dir
        self.verbose = verbose

        if not osp.isdir(self.tmp_dir):
            Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        if not osp.isdir(self.log_dir):
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            
        self.logger = logging.getLogger("MSA-FET")
        if self.verbose == 1:
            self.__set_logger(logging.INFO)
        elif self.verbose == 0:
            self.__set_logger(logging.ERROR)
        elif self.verbose == 2:
            self.__set_logger(logging.DEBUG)
        else:
            raise ValueError(f"Invalid verbose level '{self.verbose}'.")
        
        self.logger.info("")
        self.logger.info("========================== MSA-FET Started ==========================")
        self.logger.info(f"Temporary directory: {self.tmp_dir}")
        self.logger.info(f"Log file is saved at: {osp.join(self.log_dir, 'MSA-FET.log')}")

        self.__init_extractors()
        

    def __set_logger(self, stream_level):
        self.logger.setLevel(logging.DEBUG)

        fh = RotatingFileHandler(osp.join(self.log_dir, 'MSA-FET.log'), maxBytes=2e7, backupCount=5)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
        fh.setFormatter(fh_formatter)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(stream_level)
        ch_formatter = logging.Formatter('%(name)s - %(message)s')
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)

    
    def __init_extractors(self):
        self.logger.info(f"Initializing feature extractors with config file '{self.config_file}'.")
        if 'audio' in self.config:
            audio_cfg = self.config['audio']
            extractor_name = audio_cfg['tool']
            self.audio_extractor = AUDIO_EXTRACTOR_MAP[extractor_name](audio_cfg, self.logger)
        if 'video' in self.config:
            video_cfg = self.config['video']
            extractor_name = video_cfg['tool']
            self.video_extractor = VIDEO_EXTRACTOR_MAP[extractor_name](video_cfg, self.logger)
    
    def __audio_extract_single(self, in_file):
        # extract audio from video file
        extension = get_codec_name(in_file, 'audio')
        tmp_audio_file = osp.join(self.tmp_dir, 'tmp_audio.' + extension)
        ffmpeg_extract(in_file, tmp_audio_file, mode='audio')
        
        # extract audio features
        audio_result = self.audio_extractor.extract(tmp_audio_file)
        # delete tmp audio file
        os.remove(tmp_audio_file)
        return audio_result

    def __video_extract_single(self, in_file):
        # extract images from video
        fps = self.config['video']['fps']
        ffmpeg_extract(in_file, self.tmp_dir, mode='image', fps=fps)

        # extract video features
        name = 'video_' + Path(in_file).stem
        video_result = self.video_extractor.extract(self.tmp_dir, name)
        # delete tmp images
        for image_path in glob(osp.join(self.tmp_dir, '*.bmp')):
            os.remove(image_path)
        return video_result

    def __audio_extract_dataset(self, dataset_name):
        # iterate over dataset

        pass

    def __video_extract_dataset(self, dataset_name):
        pass

    def __save_result(self, result, out_file):
        if osp.exists(out_file):
            raise RuntimeError(f"Output file '{out_file}' already exists.")
        with open(out_file, 'wb') as f:
            pickle.dump(result, f)

    def run_single(self, in_file, out_file=None, text_file=None, return_type='pt'):
        """
        Extract features from single file.

        Parameters:
            in_file: path to input video file.
            return_type: 'pt' for pytorch tensor, 'np' for numpy array. Default: 'pt'.
            out_file (optional): path to output file. Default: None.
            text_file (optional): path to text file. used for feature alignment. Default: None.
        
        Returns:
            final_result: dictionary of extracted features.
        """
        try:
            self.logger.info(f"Extracting features from '{in_file}'.")
            final_result = {}
            if 'audio' in self.config:
                audio_result = self.__audio_extract_single(in_file)
            if 'video' in self.config:
                video_result = self.__video_extract_single(in_file)
            # combine audio and video features
            if return_type == 'pt':
                final_result['audio'] = torch.from_numpy(audio_result)
                final_result['video'] = torch.from_numpy(video_result)
            elif return_type == 'np':
                final_result['audio'] = audio_result
                final_result['video'] = video_result
            else:
                raise ValueError(f"Invalid return type '{return_type}'.")
            # save result
            if out_file:
                self.__save_result(final_result, out_file)
            return final_result
        except Exception:
            self.logger.exception("An Error Occured:")

    def run_dataset(self, dataset_name=None, dataset_dir=None, out_file=None):
        """
        Extract features from dataset and save in MMSA compatible format.

        Parameters:
            dataset_name: name of dataset. Either 'dataset_name' or 'dataset_dir' must be specified.
            dataset_dir: Path to dataset directory. If specified, will override 'dataset_name'. Either 'dataset_name' or 'dataset_dir' must be specified.
            out_file: output feature file. If not specified, will be saved in the same directory as the dataset using the default name 'feature.pkl'.
        """
        try:
            assert dataset_name is not None or dataset_dir is not None, "Either 'dataset_name' or 'dataset_dir' must be specified."
            if dataset_dir: # Use dataset_dir
                self.dataset_dir = osp.normpath(dataset_dir)
                if not osp.exists(self.dataset_dir):
                    raise RuntimeError(f"Dataset directory '{self.dataset_dir}' does not exist.")
                if not osp.exists(osp.join(self.dataset_dir, 'Raw')):
                    raise RuntimeError(f"Could not find 'Raw' folder in Dataset Directory '{self.dataset_dir}'.")
                self.logger.info(f"Using dataset directory '{self.dataset_dir}'.")
                self.dataset_name = osp.basename(self.dataset_dir)
                files = sorted(glob(osp.join(self.dataset_dir, 'Raw', '*/*')), key=natural_keys)
                files = [f for f in files if osp.isfile(f)]
                self.logger.info(f"Found {len(files)} files.")
                for file in tqdm(files):
                    pass
            else: # Use dataset_name
                if self.dataset_root_dir is None:
                    raise ValueError("Dataset root directory is not specified.")
                elif not osp.isdir(self.dataset_root_dir):
                    raise RuntimeError(f"Dataset root directory '{self.dataset_root_dir}' does not exist.")
                    
                self.dataset_name = dataset_name
                dataset_config_file = osp.join(self.dataset_root_dir, 'config.json')
                if osp.isfile(dataset_config_file):
                    with open(dataset_config_file) as f:
                        dataset_config = json.load(f)
                    if self.dataset_name in dataset_config:
                        # TODO: load dataset config
                        pass
                    else:
                        # TODO: locate dataset using dataset_name
                        pass
                else:
                    self.logger.info(f"Dataset config not found. Trying to locate dataset directory...")
                    # TODO: locate dataset using dataset_name

            self.logger.info(f"Extracting dataset features from '{self.dataset_dir}'.")
            if 'audio' in self.config:
                self.__audio_extract_dataset(dataset_name)
            if 'video' in self.config:
                self.__video_extract_dataset(dataset_name)
        except Exception:
            self.logger.exception("An Error Occured:")
        
