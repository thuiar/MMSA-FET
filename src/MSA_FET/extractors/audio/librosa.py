import numpy as np
import librosa
from ..baseExtractor import baseAudioExtractor


class librosaExtractor(baseAudioExtractor):
    """
    Audio feature extractor using librosa. 
    Ref: https://librosa.org/doc/latest/index.html
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing librosa audio feature extractor.")
            super().__init__(config, logger)
        except Exception as e:
            self.logger.error("Failed to initialize librosaExtractor.")
            raise e

    def extract(self, file):
        """
        Function:
            Extract features from audio file using librosa.

        Parameters:
            file: path to audio file

        Returns:
            audio_result: extracted audio features in numpy array
        """
        try:
            y, sr = self.load_audio(file)
            res = {}
            for audio_feature in self.config['args'].keys():
                kwargs = self.config['args'][audio_feature]
                method = getattr(librosa.feature, audio_feature)
                try:
                    res[audio_feature] = method(y=y, sr=sr, **kwargs).T
                except TypeError:
                    res[audio_feature] = method(y=y, **kwargs).T
            # concatenate all features
            audio_result = np.concatenate(list(res.values()), axis=1)
            return audio_result
        except Exception as e:
            self.logger.error(f"Failed to extract audio features with librosa from {file}.")
            raise e
