from .audio.librosa import librosaExtractor
from .audio.opensmile import opensmileExtractor
from .audio.wave2vec import wav2vec2Extractor
from .video.mediapipe import mediapipeExtractor
from .video.vggface import vggfaceExtractor

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