from typing import List

import ctc_segmentation
import numpy as np
import librosa
import torch
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC,
                          Wav2Vec2Processor)

__all__ = ['Wav2Vec2Aligner']

class Wav2Vec2Aligner(object):
    def __init__(self, config, logger) -> None:
        try:
            logger.info("Initializing Wav2vec2 Aligner...")
            self.logger = logger
            self.config = config
            self.model_name = config['args']['model_name']
            assert self.model_name, "config `args.model_name` is not set"

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.sample_rate = 16000
            self.has_asr = True
            self.asr_text = None
        except Exception as e:
            logger.error("Failed to initialize Wav2Vec2Aligner.")
            raise e

    def align_with_transcript(self, audio_file, transcript):
        # Run prediction, get logits and probs
        audio, _ = librosa.load(audio_file, sr=self.sample_rate)
        features = self.processor(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt", 
            padding="longest"
        )
        with torch.no_grad():
            logits = self.model(features.input_values).logits.cpu()[0]
            probs = torch.nn.functional.log_softmax(logits,dim=-1)

        # Tokenize transcripts
        transcripts = transcript.split(" ")
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v:k for k,v in vocab.items()}
        unk_id = vocab["<unk>"]
        tokens = []
        for transcript in transcripts:
            assert len(transcript) > 0
            tok_ids = self.tokenizer(transcript.replace("\n"," ").lower())['input_ids']
            tok_ids = np.array(tok_ids,dtype=np.int32)
            tokens.append(tok_ids[tok_ids != unk_id])
        
        # Do align
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
        config.index_duration = audio.shape[0] / probs.size()[0] / self.sample_rate
        
        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
        return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : np.exp(p[2])} for t,p in zip(transcripts, segments)]

    def do_asr_and_align(self, audio_file):
        # Run ASR, get transcripts
        audio, _ = librosa.load(audio_file, sr=self.sample_rate)
        features = self.processor(
            audio, 
            sampling_rate=self.sample_rate,
            return_tensors="pt", 
            padding="longest"
        )
        with torch.no_grad():
            logits = self.model(features.input_values).logits.cpu()[0]
            probs = torch.nn.functional.log_softmax(logits,dim=-1)

        predicted_ids = torch.argmax(logits, dim=-1)
        self.asr_text = self.processor.decode(predicted_ids)
        transcripts = self.asr_text.split(" ")

        # Do align
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v:k for k,v in vocab.items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
        config.index_duration = audio.shape[0] / probs.size()[0] / self.sample_rate
        
        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, transcripts)
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
        return [{"text" : w, "start" : p[0], "end" : p[1], "conf" : p[2]} for w,p in zip(transcripts, segments)]
    
    def get_asr_result(self):
        return self.asr_text