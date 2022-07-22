from ..baseExtractor import baseTextExtractor
import numpy as np


class gloveExtractor(baseTextExtractor):
    """
    Text feature extractor using GLOVE
    Ref: https://huggingface.co/docs/transformers/model_doc/bert
    Pretrained models: https://huggingface.co/models
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing BERT text feature extractor...")
            super().__init__(config, logger)
            self.embedding, self.dim = self.get_glove_embeddings(config['embedding_file'])
        except Exception as e:
            logger.error("Failed to initialize gloveExtractor.")
            raise e

    def get_glove_embeddings(self, emb_file):
        res = {}
        with open(emb_file, "r") as f:
            while data := f.readline():
                data = data.split(' ')
                res[data[0]] = np.asarray(data[1:])
        return res, len(res['the'])
    
    def extract(self, text):
        try:
            text_list = text.split(' ')
            result = []
            for word in text_list:
                if word.lower() in self.embedding:
                    result.append(self.embedding[word.lower()])
                else:
                    result.append(np.zeros(self.dim))
            return np.asarray(result)
        except Exception as e:
            self.logger.error(f"Failed to extract text features with Glove for '{text}'.")
            raise e
    
    def tokenize(self, text):
        return None