from glob import glob
import logging
import os
import os.path as osp
import shutil

import numpy as np
from torch.utils.data import Dataset

from .ASD import run_ASD
from .extractors import *
from .utils import ffmpeg_extract


class FET_Dataset(Dataset):
    """
    Dataset for MMSA-FET
    """

    def __init__(
        self, 
        df, 
        dataset_dir, 
        dataset_name, 
        config,
        dataset_config, 
        tmp_dir,
        ignore_error=True
    ):
        self.df = df
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.config = config
        self.dataset_config = dataset_config
        self.tmp_dir = tmp_dir
        self.ignore_error = ignore_error
        self.annotation_dict = {
            'Negative': -1,
            'Neutral': 0,
            'Positive': 1
        }
        if self.dataset_config:
            if 'annotations' in self.dataset_config and type(self.dataset_config['annotations']) == dict:
                self.annotation_dict = self.dataset_config['annotations']
        self.logger = logging.getLogger("MMSA-FET")
        self._init_extractors()

    def __len__(self):
        return len(self.df)

    def _init_extractors(self):
        if 'audio' in self.config:
            audio_cfg = self.config['audio']
            extractor_name = audio_cfg['tool']
            self.audio_extractor = AUDIO_EXTRACTOR_MAP[extractor_name](audio_cfg, self.logger)
        if 'video' in self.config:
            video_cfg = self.config['video']
            extractor_name = video_cfg['tool']
            self.video_extractor = VIDEO_EXTRACTOR_MAP[extractor_name](video_cfg, self.logger)
        if 'text' in self.config:
            text_cfg = self.config['text']
            extractor_name = text_cfg['model']
            self.text_extractor = TEXT_EXTRACTOR_MAP[extractor_name](text_cfg, self.logger)

    def _extract_video(self, video_path, video_id):
        # extract images from video
        fps = self.config['video']['fps']
        out_path = osp.join(self.tmp_dir, video_id)
        os.makedirs(out_path, exist_ok=False)

        if 'multiFace' in self.config['video'] and self.config['video']['multiFace']['enable'] == True:
            # enable Active Speaker Detection
            run_ASD(video_path, out_path, fps, self.config['video']['multiFace'])
            if len(glob(osp.join(out_path, '*.jpg'))) == 0:
                self.logger.warning(f'ASD returned empty results for video {video_id}')
                shutil.rmtree(out_path)
                return np.zeros((1,1)) # TODO: return zero tensor with the same dimension as normal features, require calculating the dimension from the config
        else:
            ffmpeg_extract(video_path, out_path, mode='image', fps=fps)

        # extract video features
        video_result = self.video_extractor.extract(out_path, video_id)
        # delete tmp images
        # for image_path in glob(osp.join(out_path, '*.bmp')):
        #     os.remove(image_path)
        shutil.rmtree(out_path)
        return video_result

    def _extract_audio(self, video_path, video_id):
        # extract audio from video file
        tmp_audio_file = osp.join(self.tmp_dir, video_id + '.wav')
        ffmpeg_extract(video_path, tmp_audio_file, mode='audio')

        # extract audio features
        audio_result = self.audio_extractor.extract(tmp_audio_file)
        # delete tmp audio file
        os.remove(tmp_audio_file)
        return audio_result

    def _extract_text(self, text):
        # extract text features
        text_result = self.text_extractor.extract(text)
        return text_result

    def _preprocess_text(self, text):
        # tokenize text, for models that use bert
        token_result = self.text_extractor.tokenize(text)
        return token_result

    def __getitem__(self, index):
        # some csv files may have incorrect order of columns
        tmp_dict = self.df.iloc[index]
        video_id, clip_id, text, label, label_T, label_A, label_V, annotation, mode = \
            tmp_dict['video_id'], tmp_dict['clip_id'], tmp_dict['text'], \
            tmp_dict['label'], tmp_dict['label_T'], tmp_dict['label_A'], \
            tmp_dict['label_V'], tmp_dict['annotation'], tmp_dict['mode']
        cur_id = video_id + '$_$' + clip_id
        tmp_id = video_id + '_' + clip_id # "$" is not allowed in file names
        res = {
            'id': cur_id,
            # 'audio': feature_A,
            # 'vision': feature_V,
            'raw_text': text,
            # 'text': feature_T,
            # 'text_bert': text_bert,
            # 'audio_lengths': seq_A,
            # 'vision_lengths': seq_V,
            'annotations': annotation,
            # 'classification_labels': np.nan if np.isnan(annotation) else self.annotation_dict[annotation],
            'regression_labels': label,
            'regression_labels_A': label_A,
            'regression_labels_V': label_V,
            'regression_labels_T': label_T,
            'mode': mode
        }
        video_path = osp.join(self.dataset_dir, 'Raw', video_id, clip_id + '.mp4') # TODO: file extension should be configurable
        try:
            # video
            if 'video' in self.config:
                feature_V = self._extract_video(video_path, tmp_id)
                seq_V = feature_V.shape[0]
                res['vision'] = feature_V
                res['vision_lengths'] = seq_V
            # audio
            if 'audio' in self.config:
                feature_A = self._extract_audio(video_path, tmp_id)
                seq_A = feature_A.shape[0]
                res['audio'] = feature_A
                res['audio_lengths'] = seq_A
            # text
            if 'text' in self.config:
                feature_T = self._extract_text(text)
                seq_T = feature_T.shape[0]
                text_bert = self._preprocess_text(text)
                res['text'] = feature_T
                res['text_bert'] = text_bert
                if type(res['text_bert']) != np.ndarray:
                    res.pop('text_bert')
            return res
        except Exception as e:
            self.logger.error(f'Error occurred when extracting features for video {video_id} clip {clip_id}')
            if self.ignore_error:
                self.logger.warning(f'Ignore error and continue, see the log for details.')
                self.logger.debug(str(e))
                return None
            else:
                raise e
        