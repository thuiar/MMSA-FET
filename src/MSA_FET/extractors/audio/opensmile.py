from ..baseExtractor import baseAudioExtractor
import opensmile

class opensmileExtractor(baseAudioExtractor):
    """
    Audio feature extractor using openSMILE. 
    Ref: https://github.com/audeering/opensmile-python
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing openSMILE audio feature extractor")
            super().__init__(config, logger)
            self.args = self.config['args']
            self.extractor = opensmile.Smile(
                feature_set=eval(f"opensmile.FeatureSet.{self.args['feature_set']}"),
                feature_level=eval(f"opensmile.FeatureLevel.{self.args['feature_level']}"),
                num_workers=None,
            )
        except Exception as e:
            self.logger.error("Failed to initialize opensmileExtractor.")
            raise e

    def extract(self, file):
        """
        Function:
            Extract features from audio file using openSMILE.

        Parameters:
            file: path to audio file

        Returns:
            audio_result: extracted audio features in numpy array
        """
        try:
            y, sr = self.load_audio(file)
            audio_result = self.extractor.process_signal(
                signal = y, 
                sampling_rate = sr, 
                start = self.args['start'],
                end = self.args['end']
            )
            audio_result = self.extractor.to_numpy(audio_result).reshape((1, -1))
            return audio_result
        except Exception as e:
            self.logger.error(f"Failed to extract audio features with openSMILE from {file}")
            raise e