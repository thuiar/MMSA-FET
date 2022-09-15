from .audio.librosa import librosaExtractor
from .audio.opensmile import opensmileExtractor
from .audio.wave2vec import wav2vec2Extractor
from .video.mediapipe import mediapipeExtractor
from .video.openface import openfaceExtractor
from .text.bert import bertExtractor
from .text.roberta import robertaExtractor
from .text.glove import gloveExtractor

AUDIO_EXTRACTOR_MAP = {
    "librosa": librosaExtractor,
    "opensmile": opensmileExtractor,
    "wav2vec": wav2vec2Extractor,
}

VIDEO_EXTRACTOR_MAP = {
    "mediapipe": mediapipeExtractor,
    "openface": openfaceExtractor,
    # "3dcnn": 3dcnn_extractor,
}

TEXT_EXTRACTOR_MAP = {
    "bert": bertExtractor,
    "roberta": robertaExtractor,
    "glove": gloveExtractor,
}