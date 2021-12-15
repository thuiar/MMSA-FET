import os
import os.path as osp
from glob import glob
import logging
import shutil

from torch.utils.data import Dataset

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
    ):
        self.df = df
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.config = config
        self.dataset_config = dataset_config
        self.tmp_dir = tmp_dir
        self.annotation_dict = {
            'Negative': -1,
            'Neutral': 0,
            'Positive': 1
        }
        if self.dataset_config:
            if 'annotations' in self.dataset_config and type(self.dataset_config['annotations']) == dict:
                self.annotation_dict = self.dataset_config['annotations']
        self.logger = logging.getLogger("MMSA-FET")
        self.__init_extractors()

    def __len__(self):
        return len(self.df)

    def __init_extractors(self):
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

    def __extract_video(self, video_path, video_id):
        # extract images from video
        fps = self.config['video']['fps']
        out_path = osp.join(self.tmp_dir, video_id)
        os.makedirs(out_path, exist_ok=False)
        ffmpeg_extract(video_path, out_path, mode='image', fps=fps)

        # extract video features
        video_result = self.video_extractor.extract(out_path, video_id)
        # delete tmp images
        # for image_path in glob(osp.join(out_path, '*.bmp')):
        #     os.remove(image_path)
        shutil.rmtree(out_path)
        return video_result

    def __extract_audio(self, video_path, video_id):
        # extract audio from video file
        tmp_audio_file = osp.join(self.tmp_dir, video_id + '.wav')
        ffmpeg_extract(video_path, tmp_audio_file, mode='audio')

        # extract audio features
        audio_result = self.audio_extractor.extract(tmp_audio_file)
        # delete tmp audio file
        os.remove(tmp_audio_file)
        return audio_result

    def __extract_text(self, text):
        # extract text features
        text_result = self.text_extractor.extract(text)
        return text_result

    def __preprocess_text(self, text):
        # tokenize text, for compatibility with MMSA
        token_result = self.text_extractor.tokenize(text)
        return token_result

    def __getitem__(self, index):
        video_id, clip_id, text, label, label_T, label_A, label_V, annotation, mode = self.df.iloc[index]
        cur_id = video_id + '_' + clip_id
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
            'classification_labels': self.annotation_dict[annotation],
            'regression_labels': label,
            'regression_labels_A': label_A,
            'regression_labels_V': label_V,
            'regression_labels_T': label_T,
            'mode': mode
        }
        # video
        video_path = osp.join(self.dataset_dir, 'Raw', video_id, clip_id + '.mp4')
        if 'video' in self.config:
            feature_V = self.__extract_video(video_path, cur_id)
            seq_V = feature_V.shape[0]
            res['vision'] = feature_V
            res['vision_lengths'] = seq_V
        # audio
        if 'audio' in self.config:
            feature_A = self.__extract_audio(video_path, cur_id)
            seq_A = feature_A.shape[0]
            res['audio'] = feature_A
            res['audio_lengths'] = seq_A
        # text
        if 'text' in self.config:
            feature_T = self.__extract_text(text)
            seq_T = feature_T.shape[0]
            # text_bert = self.__preprocess_text(text)
            res['text'] = feature_T
            # res['text_bert'] = text_bert

        return res
