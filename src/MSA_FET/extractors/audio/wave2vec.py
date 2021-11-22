from transformers import Wav2Vec2FeatureExtractor
from ..baseExtractor import baseAudioExtractor


class wav2vec2Extractor(baseAudioExtractor):
    """
    Audio feature extractor using Wav2Vec2. 
    Ref: https://huggingface.co/transformers/model_doc/wav2vec2.html
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing Wav2Vec2 audio feature extractor.")
            super().__init__(config, logger)
            self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(config['pretrained'])
        except Exception as e:
            self.logger.error("Failed to initialize Wav2VecExtractor.")
            raise e

    def extract(self, file):
        """
        Function:
            Extract features from audio file using wav2vec2 pretrained model.

        Parameters:
            file: path to audio file

        Returns:
            audio_result: extracted audio features in numpy array
        """
        try:
            y, sr = self.load_audio(file)
            audio_result = self.extractor(y, sampling_rate=sr, return_tensors="np").input_values.T
            return audio_result
        except Exception as e:
            self.logger.error(f"Failed to extract audio features with Wav2Vec2 from {file}.")
            raise e
