from email.policy import default
import inspect

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
            logger.info("Initializing librosa audio feature extractor...")
            super().__init__(config, logger)
            self.n_fft = self.config.get('n_fft', 512)
            self.hop_length = self.config.get('hop_length', 128)
            self.seq_len, self.feature_dim = 0, 0
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
            for audio_feature, kwargs in self.config['features'].items():
                method = getattr(librosa.feature, audio_feature)
                # Get function args
                valid_args = set([
                    val.name for val in inspect.signature(method).parameters.values()
                ])
                # Check invalid args if there's no kwargs
                if not 'kwargs' in valid_args:
                    for key in kwargs.keys():
                        if key not in valid_args:
                            self.logger.warning(
                                f"Removing invalid parameter {key} for feature {audio_feature}"
                            )
                            kwargs.pop(key)
                # Add sr to kwargs if it's a valid arg
                if 'sr' in valid_args:
                    kwargs['sr'] = sr
                # Add hop_length to kwargs if it's a valid arg
                kwargs['hop_length'] = self.hop_length
                # Add n_fft or frame_length to kwargs accordingly
                if 'n_fft' in valid_args:
                    kwargs['n_fft'] = self.n_fft
                elif 'frame_length' in valid_args:
                    kwargs['frame_length'] = self.n_fft
                # Add hidden params to mfcc feature
                if audio_feature == 'mfcc':
                    kwargs['n_fft'] = self.n_fft
                    kwargs['hop_length'] = self.hop_length
                res[audio_feature] = method(y=y, **kwargs).T
            # concatenate all features
            audio_result = np.concatenate(list(res.values()), axis=1)
            self.seq_len, self.feature_dim = audio_result.shape[0], audio_result.shape[1]
            return audio_result
        except Exception as e:
            self.logger.error(f"Failed to extract audio features with librosa from {file}.")
            raise e

    def get_feature_names(self):
        names = []
        for audio_feature, args in self.config['features'].items():
            # requires python>=3.10
            match audio_feature:
                case 'mfcc':
                    n_mfcc = args.get('n_mfcc', 20)
                    for i in range(n_mfcc):
                        names.append(f'mfcc_{i}')
                case 'chroma_stft' | 'chroma_cqt' | 'chroma_cens':
                    n_chroma = args.get('n_chroma', 12)
                    for i in range(n_chroma):
                        names.append(f'{audio_feature}_{i}')
                case 'spectral_contrast':
                    n_bands = args.get('n_bands', 6)
                    for i in range(n_bands + 1):
                        names.append(f'spectral_contrast_{i}')
                case 'poly_features':
                    order = args.get('order', 1)
                    for i in range(order + 1):
                        names.append(f'poly_features_{i}')
                case 'tonnetz':
                    names.extend([
                        'Fifth x-axis',
                        'Fifth y-axis',
                        'Minor x-axis',
                        'Minor y-axis',
                        'Major x-axis',
                        'Major y-axis'
                    ])
                case _:
                    names.append(audio_feature)
        assert len(names) == self.feature_dim
        return names

    def get_timestamps(self) -> list:
        indicies = np.arange(0, self.seq_len)
        result = librosa.frames_to_time(
            indicies, 
            sr=self.config['sample_rate'], 
            hop_length=self.hop_length
        )
        return list(result)
    
