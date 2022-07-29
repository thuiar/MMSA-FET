class BaseAligner(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.has_asr = False

    def align_with_transcript(self, audio_file, transcript):
        raise NotImplementedError("align_with_transcript() not implemented")
    
    def do_asr_and_align(self, audio_file):
        if self.has_asr:
            raise NotImplementedError("do_asr_and_align() not implemented")
        else:
            raise RuntimeError("This aligner does not support ASR")
    
    def get_asr_result(self):
        if self.has_asr:
            raise NotImplementedError("get_asr_result() not implemented")
        else:
            raise RuntimeError("This aligner does not support ASR")