import librosa


class baseExtractor(object):
    """
    Base class for all extractors.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def extract(self, file):
        """
        Extract features from input file.
        """
        raise NotImplementedError("extract() not implemented")


class baseAudioExtractor(baseExtractor):
    """
    Base class for all audio extractors.
    """
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def load_audio(self, file):
        """
        Load audio file using librosa.
        """
        y, sr = librosa.load(file, sr=self.config['sample_rate'])
        return y, sr


class baseTextExtractor(baseExtractor):
    """
    Base class for all text extractors.
    """
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def load_text(self, file):
        """
        Load text from file.
        """
        with open(file, 'r') as f:
            text = f.read()
        return text

    def tokenize(self, text):
        """
        Tokenize the input text.
        """
        raise NotImplementedError("tokenize() not implemented")

# class baseVideoExtractor(baseExtractor):
#     """
#     Base class for all video extractors.
#     """
#     def __init__(self, config, logger):
#         super().__init__(config, logger)

#     def load_images(self, img_dir):
#         """
#         Load image files using cv2.
#         """
#         images = []
#         for image_path in sorted(glob(osp.join(img_dir, '*.bmp'))):
#             name = Path(image_path).stem
#             image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#             images.append(image)
#         return images