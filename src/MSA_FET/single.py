import json
import logging
import os
import pickle
import shutil
import time
from glob import glob
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import *


class FeatureExtractionTool(object):
    """
    Feature Extraction Tool for Multimodal Sentiment Analysis tasks.

    Parameters:
        config: 
            Python dictionary or path to a JSON file.
        dataset_root_dir: 
            (Deprecated) Dataset root directory where datasets are stored in seperate folders.
        tmp_dir: 
            Directory for temporary files. Default: '~/.MSA-FET/tmp'.
        log_dir: 
            Log directory. Default: '~/.MSA-FET/log'.
        verbose: 
            Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1.

    TODOs:
        [ ] Support VGGFace2 or DenseFace
        [ ] Add option to pad zeros instead of discard the frame when no human faces are detected.
        [ ] Add csv/dataframe output format.
        [ ] Support specifying existing feature files, modify only some of the modalities.
        [ ] Implement resume function for run_dataset().
        [ ] Clean up tmp folder before run_single.
        [ ] Better error logs, optimize stack traces to avoid duplicate messages.
        [ ] Set gpu_id during init, not in config.
    """

    def __init__(
        self,
        config : dict | str,
        dataset_root_dir : Path | str = None,
        tmp_dir : Path | str = Path.home() / '.MMSA-FET/tmp',
        log_dir : Path | str = Path.home() / '.MMSA-FET/log',
        verbose : int = 1
    ) -> None:
        if type(config) == dict:
            self.config = config
        elif type(config) == str:
            if Path(config).is_file():
                with open(config, 'r') as f:
                    self.config = json.load(f)
            elif Path(name := Path(__file__).parent / 'example_configs' / f"{config}.json").is_file():
                with open(name, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Config file {config} does not exist.")
        else:
            raise ValueError("Invalid argument type for `config`.")
        self.tmp_dir = Path(tmp_dir)
        self.log_dir = Path(log_dir)
        self.verbose = verbose

        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("FET-Single")
        if self.verbose == 1:
            self._set_logger(logging.INFO)
        elif self.verbose == 0:
            self._set_logger(logging.ERROR)
        elif self.verbose == 2:
            self._set_logger(logging.DEBUG)
        else:
            raise ValueError(f"Invalid verbose level '{self.verbose}'.")
        
        self.logger.info("")
        self.logger.info("========================== MMSA-FET Started ==========================")
        self.logger.info(f"Temporary directory: {self.tmp_dir}")
        self.logger.info(f"Log file: '{self.log_dir / 'MMSA-FET.log'}'")
        # self.logger.info(f"Config file: '{self.config_file}'")
        self.logger.info(f"Config: {self.config}")

        if dataset_root_dir is not None:
            self.logger.warning("`dataset_root_dir` is deprecated. To extract dataset features, use `run_dataset()` from .")

        self.video_extractor, self.audio_extractor, self.text_extractor, self.aligner = None, None, None, None

    def _set_logger(self, stream_level : int) -> None:
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)

        fh = RotatingFileHandler(self.log_dir / 'MSA-FET.log', maxBytes=2e7, backupCount=2)
        fh_formatter = logging.Formatter(
            fmt= '%(asctime)s - %(name)s [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fh_formatter)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(stream_level)
        ch_formatter = logging.Formatter('%(name)s - %(message)s')
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)

    def _init_extractors(self) -> None:
        from .aligner import Aligner
        from .extractors import (AUDIO_EXTRACTOR_MAP, TEXT_EXTRACTOR_MAP,
                                 VIDEO_EXTRACTOR_MAP)
        if 'audio' in self.config and self.audio_extractor is None:
            # self.logger.info(f"Initializing audio feature extractor...")
            audio_cfg = self.config['audio']
            extractor_name = audio_cfg['tool']
            self.audio_extractor = AUDIO_EXTRACTOR_MAP[extractor_name](audio_cfg, self.logger)
        if 'video' in self.config and self.video_extractor is None:
            # self.logger.info(f"Initializing video feature extractor...")
            video_cfg = self.config['video']
            extractor_name = video_cfg['tool']
            self.video_extractor = VIDEO_EXTRACTOR_MAP[extractor_name](video_cfg, self.logger)
        if 'text' in self.config and self.text_extractor is None:
            # self.logger.info(f"Initializing text feature extractor...")
            text_cfg = self.config['text']
            extractor_name = text_cfg['model']
            self.text_extractor = TEXT_EXTRACTOR_MAP[extractor_name](text_cfg, self.logger)
        if 'align' in self.config and self.aligner is None:
            align_cfg = self.config['align']
            self.aligner = Aligner(align_cfg, self.logger)
    
    def _audio_extract_single(self, in_file : Path, keep_tmp_file : bool = False) -> np.ndarray:
        # extract audio from video file
        # extension = get_codec_name(in_file, 'audio')
        tmp_audio_file = self.tmp_dir / 'tmp_audio.wav'
        ffmpeg_extract(in_file, tmp_audio_file, mode='audio')
        
        # extract audio features
        audio_result = self.audio_extractor.extract(tmp_audio_file)
        # delete tmp audio file
        if not keep_tmp_file:
            os.remove(tmp_audio_file)
        return audio_result

    def _video_extract_single(self, in_file : Path, keep_tmp_file : bool = False) -> np.ndarray:
        # extract images from video
        fps = self.config['video']['fps']
        if self.config['video'].get('multiFace', {}).get('enable', False):
            # enable Active Speaker Detection
            from .ASD import run_ASD
            run_ASD(in_file, self.tmp_dir, fps, self.config['video']['multiFace'])
        else:
            ffmpeg_extract(in_file, self.tmp_dir, mode='image', fps=fps)

        # extract video features
        name = 'video_' + Path(in_file).stem
        video_result = self.video_extractor.extract(self.tmp_dir, name, tool_output=self.verbose>0)
        # delete tmp images
        if not keep_tmp_file:
            for image_path in glob(str(self.tmp_dir / '*.bmp')):
                os.remove(image_path)
            for image_path in glob(str(self.tmp_dir / '*.jpg')):
                os.remove(image_path)
        return video_result

    def _text_extract_single(self, in_file : Path, in_text : str = None) -> np.ndarray:
        if in_text:
            text = in_text
        else:
            text = self.text_extractor.load_text_from_file(in_file)
        text_result = self.text_extractor.extract(text)
        text_tokens = self.text_extractor.tokenize(text)
        text_tokens = text_tokens.transpose(1, 0)
        return text_result, text_tokens

    def _aligned_extract_single(
        self, 
        align_result : list[dict], 
        word_ids : list[int], 
        audio_result : np.ndarray = None, 
        video_result : np.ndarray = None
    ) -> tuple[np.ndarray]:
        word_count = len(align_result)
        df = pd.DataFrame(align_result)
        start = df['start'].values
        end = df['end'].values
        if audio_result is not None:
            audio_timestamp = self.audio_extractor.get_timestamps()
            start_idx = np.searchsorted(audio_timestamp, start)
            end_idx = np.searchsorted(audio_timestamp, end)
            tmp_result = np.array([np.mean(audio_result[x:y], axis=0) for x, y in zip(start_idx, end_idx)])
            assert len(tmp_result) == word_count
            # align with text tokens, add zero padding or duplicate features
            aligned_audio_result = []
            for i in word_ids:
                if i is None:
                    aligned_audio_result.append(np.zeros(tmp_result.shape[1]))
                else:
                    aligned_audio_result.append(tmp_result[i])
            aligned_audio_result = np.asarray(aligned_audio_result)
        else:
            aligned_audio_result = None
        if video_result is not None:
            video_timestamp = self.video_extractor.get_timestamps()
            start_idx = np.searchsorted(video_timestamp, start)
            end_idx = np.searchsorted(video_timestamp, end)
            tmp_result = np.array([np.mean(video_result[x:y], axis=0) for x, y in zip(start_idx, end_idx)])
            assert len(tmp_result) == word_count
            # align with text tokens, add zero padding or duplicate features
            aligned_video_result = []
            for i in word_ids:
                if i is None:
                    aligned_video_result.append(np.zeros(tmp_result.shape[1]))
                else:
                    aligned_video_result.append(tmp_result[i])
            aligned_video_result = np.asarray(aligned_video_result)
        else:
            aligned_video_result = None
        return aligned_audio_result, aligned_video_result

    def _save_result(self, result: dict, out_file : Path):
        if out_file.exists():
            out_file_alt = out_file.parent / (out_file.stem + '_' + str(int(time.time())) + '.pkl')
            self.logger.warning(f"Output file '{out_file}' already exists. Saving to '{out_file_alt}' instead.")
            out_file = out_file_alt
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Feature file saved: '{out_file}'.")
    
    def _remove_tmp_folder(self, tmp_dir : Path) -> None:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def run_single(
        self, 
        in_file : Path | str, 
        out_file : Path | str = None, 
        text : str = None, 
        text_file : Path | str = None, 
        return_type : str = 'np'
    ) -> dict:
        """
        Extract features from single file.

        Parameters:
            in_file: path to input video file.
            return_type: `'pt'` for pytorch tensor, `'np'` for numpy array. 
                `'pd'` for pandas dataframe. Default: `'np'`.
            out_file (optional): path to output file.
            text (optional): text to be extracted.
            text_file (optional): path to text file. Ignored if `text` is not None.
        
        Returns:
            final_result: dictionary of extracted features.
        """
        try:
            self._init_extractors()
            in_file = Path(in_file)
            if out_file:
                out_file = Path(out_file)
            self.logger.info(f"Extracting features from '{in_file}'.")
            final_result = {}
            if 'audio' in self.config:
                audio_result = self._audio_extract_single(in_file, keep_tmp_file=True)
            if 'video' in self.config:
                video_result = self._video_extract_single(in_file)
            if 'align' in self.config:
                assert 'audio' in self.config or 'video' in self.config
                assert 'text' in self.config, "Text feature is required for alignment. Please add 'text' section in config."
                if text is None and text_file is None:
                    if self.aligner.has_transcript:
                        raise ValueError("Text file is not specified.")
                    self.logger.warning("Text file is not specified. Using ASR result.")
                    align_result = self.aligner.asr_and_align(in_file)
                    text = self.aligner.get_asr_result()
                else:
                    text = text if text is not None else open(text_file).read().strip()
                    align_result = self.aligner.align(in_file, text)
                word_ids = self.text_extractor.get_word_ids(text)
                audio_result, video_result = self._aligned_extract_single(
                    align_result, word_ids, audio_result, video_result
                )
            if 'text' in self.config:
                text_result, text_tokens = self._text_extract_single(None, text)
            if 'align' in self.config:
                # verify aligned sequence length
                if 'audio' in self.config:
                    assert audio_result.shape[0] == text_result.shape[0]
                if 'video' in self.config:
                    assert video_result.shape[0] == text_result.shape[0]
            # combine audio and video features
            if return_type == 'pt':
                if 'audio' in self.config:
                    final_result['audio'] = torch.from_numpy(audio_result)
                if 'video' in self.config:
                    final_result['vision'] = torch.from_numpy(video_result)
                if 'text' in self.config:
                    final_result['text'] = torch.from_numpy(text_result)
                    final_result['text_bert'] = torch.from_numpy(text_tokens)
            elif return_type == 'np':
                if 'audio' in self.config:
                    final_result['audio'] = audio_result
                if 'video' in self.config:
                    final_result['vision'] = video_result
                if 'text' in self.config:
                    final_result['text'] = text_result
                    final_result['text_bert'] = text_tokens
            elif return_type == 'df':
                pass
            else:
                raise ValueError(f"Invalid return type '{return_type}'.")
            # save configs
            final_result['config'] = self.config
            # save align results
            if 'align' in self.config:
                final_result['align'] = align_result
            # save result
            if out_file:
                self._save_result(final_result, out_file)
            return final_result
        except Exception as e:
            self.logger.exception("An Error Occured:")
            self.logger.debug("Removing temporary files.")
            self._remove_tmp_folder(self.tmp_dir)
            raise e

    def run_dataset(self, **kwargs):
        raise DeprecationWarning('This method is deprecated. Please use "run_dataset" from "MSA_FET.dataset" instead.')
