import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from ..baseExtractor import baseAudioExtractor


class wav2vec2Extractor(baseAudioExtractor):
    """
    Audio feature extractor using Wav2Vec2. 
    Ref: https://huggingface.co/transformers/model_doc/wav2vec2.html
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing Wav2Vec2 audio feature extractor...")
            super().__init__(config, logger)
            self.preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(config['pretrained'])
            self.extractor = Wav2Vec2Model.from_pretrained(config['pretrained'])
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
            audio_result = self.preprocessor(y, sampling_rate=sr, return_tensors="pt").input_values
            with torch.no_grad():
                audio_result = self.extractor(audio_result).last_hidden_state.squeeze(0)
            return audio_result
        except Exception as e:
            self.logger.error(f"Failed to extract audio features with Wav2Vec2 from {file}.")
            raise e
