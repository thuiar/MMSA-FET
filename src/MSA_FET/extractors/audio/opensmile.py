from ..baseExtractor import baseAudioExtractor
import opensmile

class opensmileExtractor(baseAudioExtractor):
    """
    Audio feature extractor using openSMILE. 
    Ref: https://github.com/audeering/opensmile-python
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing openSMILE audio feature extractor...")
            super().__init__(config, logger)
            self.args = self.config['args']
            self.extractor = opensmile.Smile(
                feature_set=eval(f"opensmile.FeatureSet.{self.args['feature_set']}"),
                feature_level=eval(f"opensmile.FeatureLevel.{self.args['feature_level']}"),
                num_workers=None,
            )
            self.time_stamps = None
        except Exception as e:
            self.logger.error("Failed to initialize opensmileExtractor.")
            raise e

    def extract(self, file, start=None, end=None):
        """
        Function:
            Extract features from audio file using openSMILE.

        Parameters:
            file: path to audio file
            start: start time in seconds, will overwrite the value in config.
            end: end time in seconds, will overwrite the value in config.

        Returns:
            audio_result: extracted audio features in numpy array
        """
        try:
            start = self.args['start'] if start is None else start
            end = self.args['end'] if end is None else end
            y, sr = self.load_audio(file)
            audio_result = self.extractor.process_signal(
                signal = y, 
                sampling_rate = sr, 
                start = start,
                end = end
            )
            self.timestamps = list(audio_result.index)
            audio_result = self.extractor.to_numpy(audio_result).squeeze(0).transpose()
            return audio_result
        except Exception as e:
            self.logger.error(f"Failed to extract audio features with openSMILE from {file}")
            raise e

    def get_feature_names(self):
        return self.extractor.feature_names

    def get_timestamps(self) -> list:
        # Only call this function after extract
        result = []
        for timestamp in self.timestamps:
            result.append(
                (timestamp[0].total_seconds() + timestamp[1].total_seconds()) / 2
            )
        return result