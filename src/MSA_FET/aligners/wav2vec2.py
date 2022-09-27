import logging
from pathlib import Path

import ctc_segmentation
import librosa
import numpy as np
import torch
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC,
                          Wav2Vec2Processor)

from .base_aligner import BaseAligner

__all__ = ['Wav2Vec2Aligner']

class Wav2Vec2Aligner(BaseAligner):
    def __init__(self, config : dict, logger : logging.Logger) -> None:
        try:
            logger.info("Initializing Wav2vec2 Aligner...")
            super().__init__(config, logger)
            self.model_name = config['args']['model_name']
            self.device = config['device']
            assert self.model_name, "config `args.model_name` is not set"

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
            self.sample_rate = 16000
            self.has_asr = True
            self.asr_text = None
        except Exception as e:
            logger.error("Failed to initialize Wav2Vec2Aligner.")
            raise e

    def align_with_transcript(self, audio_file : Path | str, transcript : str) -> list[dict]:
        # Run prediction, get logits and probs
        audio, _ = librosa.load(audio_file, sr=self.sample_rate)
        features = self.processor(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt", 
            padding="longest"
        )
        features = features.to(self.device)
        with torch.no_grad():
            logits = self.model(features.input_values).logits.cpu()[0]
            probs = torch.nn.functional.log_softmax(logits,dim=-1)

        # Tokenize transcripts
        transcripts = transcript.split()
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v:k for k,v in vocab.items()}
        unk_id = vocab["<unk>"]
        tokens = []
        for transcript in transcripts:
            assert len(transcript) > 0
            tok_ids = self.tokenizer(transcript.lower())['input_ids']
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

    def do_asr_and_align(self, audio_file : Path | str) -> list[dict]:
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
        transcripts = self.asr_text.split()

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
    
    def get_asr_result(self) -> str:
        return self.asr_text

    # @staticmethod
    # def do_asr(
    #     audio_file : str | Path, 
    #     model_name : str = "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    #     device : torch.device = torch.device("cpu")
    # ) -> str:
    #     try:
    #         processor = Wav2Vec2Processor.from_pretrained(model_name)
    #         model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    #         sample_rate = 16000
    #         audio, _ = librosa.load(audio_file, sr=sample_rate)
    #         features = processor(
    #             audio, 
    #             sampling_rate=sample_rate,
    #             return_tensors="pt", 
    #             padding="longest"
    #         )
    #         with torch.no_grad():
    #             logits = model(features.input_values.to(device)).logits.cpu()[0]

    #         predicted_ids = torch.argmax(logits, dim=-1)
    #         asr_text = processor.decode(predicted_ids)
    #         return asr_text
    #     except Exception as e:
    #         raise e
    
    # @staticmethod
    # def do_alignment(
    #     audio_file : str | Path, 
    #     transcript : str, 
    #     model_name : str = "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    #     device : torch.device = torch.device("cpu")
    # ) -> list[dict]:
    #     try:
    #         processor = Wav2Vec2Processor.from_pretrained(model_name)
    #         tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    #         model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    #         sample_rate = 16000
    #         audio, _ = librosa.load(audio_file, sr=sample_rate)
    #         features = processor(
    #             audio, 
    #             sampling_rate=sample_rate, 
    #             return_tensors="pt", 
    #             padding="longest"
    #         )
    #         with torch.no_grad():
    #             logits = model(features.input_values.to(device)).logits.cpu()[0]
    #             probs = torch.nn.functional.log_softmax(logits,dim=-1)

    #         # Tokenize transcripts
    #         transcripts = transcript.split(" ")
    #         vocab = tokenizer.get_vocab()
    #         inv_vocab = {v:k for k,v in vocab.items()}
    #         unk_id = vocab["<unk>"]
    #         tokens = []
    #         for transcript in transcripts:
    #             assert len(transcript) > 0
    #             tok_ids = tokenizer(transcript.lower())['input_ids']
    #             tok_ids = np.array(tok_ids,dtype=np.int32)
    #             tokens.append(tok_ids[tok_ids != unk_id])
            
    #         # Do align
    #         char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    #         config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    #         config.index_duration = audio.shape[0] / probs.size()[0] / sample_rate
            
    #         ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    #         timings, char_probs, _ = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    #         segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
    #         return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : np.exp(p[2])} for t,p in zip(transcripts, segments)]
    #     except Exception as e:
    #         raise e