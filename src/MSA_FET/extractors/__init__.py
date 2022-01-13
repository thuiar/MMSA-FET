from .audio.librosa import librosaExtractor
from .audio.opensmile import opensmileExtractor
from .audio.wave2vec import wav2vec2Extractor
from .video.mediapipe import mediapipeExtractor
from .video.openface import openfaceExtractor
from .video.vggface import vggfaceExtractor
from .text.bert import bertExtractor

__all__ = ['AUDIO_EXTRACTOR_MAP', 'VIDEO_EXTRACTOR_MAP', 'TEXT_EXTRACTOR_MAP']

AUDIO_EXTRACTOR_MAP = {
    "librosa": librosaExtractor,
    "opensmile": opensmileExtractor,
    "wav2vec": wav2vec2Extractor,
}

VIDEO_EXTRACTOR_MAP = {
    "mediapipe": mediapipeExtractor,
    "openface": openfaceExtractor,
    # "vggface": vggfaceExtractor,
    # "3dcnn": 3dcnn_extractor,
}

TEXT_EXTRACTOR_MAP = {
    "bert": bertExtractor,
    # "xlnet": xlnetExtractor,
}