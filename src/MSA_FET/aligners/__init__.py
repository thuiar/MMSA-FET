from .wav2vec2 import Wav2Vec2Aligner

__all__ = ['ALIGNER_MAP']

ALIGNER_MAP = {
    "wav2vec": Wav2Vec2Aligner,
}
