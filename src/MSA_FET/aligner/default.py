import logging
from pathlib import Path

import librosa
import numpy as np
import sentencepiece as spm
import torch
import yaml
from ctc_segmentation import (CtcSegmentationParameters, ctc_segmentation,
                              determine_utterance_segments, prepare_text,
                              prepare_token_list)
from espnet2.bin.asr_inference import Speech2Text
from espnet2.tasks.asr import ASRTask
from espnet_model_zoo.downloader import ModelDownloader
from typeguard import typechecked


class Aligner(object):
    @typechecked
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._parse_config()
        # init model
        self.logger.info(f"Initializing Aligner for {self.language}...")
        d = ModelDownloader(cachedir=self.model_download_dir)
        model = d.download_and_unpack(self.model_name)
        # model to calculate ctc logits
        self.asr_model, self.asr_train_args = ASRTask.build_model_from_file(
            config_file = model["asr_train_config"],
            model_file = model["asr_model_file"],
            device = self.device,
        )
        # sentencepiece model for en-us
        if self.language == "en-us":
            self.sp = spm.SentencePieceProcessor()
            with open(model["asr_train_config"], "r") as f:
                tmp = yaml.safe_load(f)
            self.sp.load(tmp["bpemodel"])
        # Speech2Text Task uses `beam search` which is better than `logits argmax`
        # in terms of asr performance
        if not self.has_transcript:
            self.asr_infer = Speech2Text(
                asr_train_config = model["asr_train_config"],
                asr_model_file = model["asr_model_file"],
                device = self.device,
            )
        self.char_list = self.asr_model.token_list[:-1]
    
    def _parse_config(self):
        config = self.config
        self.sr = config.get("sr", 16000)
        self.language = config.get("language", "en-us")
        if self.language == "zh-cn":
            self.model_name = "espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char"
        elif self.language == "en-us": # use huggingface model for english
            self.model_name = "kamo-naoyuki/librispeech_asr_train_asr_conformer5_raw_bpe5000_scheduler_confwarmup_steps25000_batch_bins140000000_optim_conflr0.0015_initnone_accum_grad2_sp_valid.acc.ave"
        self.device = config.get("device", "cpu")
        self.has_transcript = config.get("has_transcript", True)
        self.model_download_dir = config.get("model_download_dir", "default")
        if self.model_download_dir == "default":
            self.model_download_dir = Path().home() / ".cache" / "espnet"
        
    @typechecked
    @torch.no_grad()
    def align(self, speech_path: Path | str, text: str) -> list[dict]:
        speech, _ = librosa.load(speech_path, sr=self.sr)
        speech = torch.tensor(speech).unsqueeze(0).to(self.device)
        lengths = torch.tensor(speech.shape[1]).unsqueeze(0).to(self.device)
        enc, _ = self.asr_model.encode(speech=speech, speech_lengths=lengths)
        probs = self.asr_model.ctc.log_softmax(enc).detach().squeeze(0).cpu()
        index_duration = speech.shape[1] / probs.shape[0] / self.sr
        config = CtcSegmentationParameters(
            char_list=self.char_list,
            index_duration=index_duration,
        )
        if self.language == "zh-cn":
            text = [ch for ch in text]
        elif self.language == "en-us":
            text = text.split(" ")
        token_ints = []
        unk = self.char_list.index("<unk>")
        if self.language == "zh-cn":
            for ch in text:
                try:
                    token_id = self.char_list.index(ch)
                    token_id = np.array([token_id])
                    token_ints.append(token_id)
                except:
                    token_id = unk # skip unk
        elif self.language == "en-us":
            for word in text:
                tokens = self.sp.EncodeAsPieces(word)
                token_ids = []
                for t in tokens:
                    try:
                        token_id = self.char_list.index(t)
                        token_ids.append(token_id)
                    except:
                        token_id = unk # skip unk
                token_ints.append(np.array(token_ids))
        ground_truth_mat, utt_begin_indices = prepare_token_list(config, token_ints)
        timings, char_probs, state_list = ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
        res = []
        for i,(t,p) in enumerate(zip(text, segments)):
            if self.language == "zh-cn": # add 0.11s offset to end timings of chinese characters
                res.append({
                    "text" : t, 
                    "start" : p[0] if i == 0 else p[0] + 0.11, 
                    "end" : min(p[1] + 0.11, speech.shape[1] / self.sr), 
                    "conf" : np.exp(p[2])
                })
            elif self.language == "en-us":
                res.append({
                    "text" : t, 
                    "start" : p[0], 
                    "end" : p[1], 
                    "conf" : np.exp(p[2])
                })
        return res

    @typechecked
    @torch.no_grad()
    def asr_and_align(self, speech_path: Path | str) -> list[dict]:
        speech, _ = librosa.load(speech_path, sr=self.sr)
        speech = torch.tensor(speech).to(self.device)
        self.asr_result, *_ = self.asr_infer(speech)[0]
        if self.language == "zh-cn":
            asr_result = [ch for ch in self.asr_result]
        elif self.language == "en-us":
            asr_result = self.asr_result.split(" ")
        lengths = torch.tensor(speech.shape[0]).unsqueeze(0).to(self.device)
        enc, _ = self.asr_model.encode(speech=speech.unsqueeze(0), speech_lengths=lengths)   
        probs = self.asr_model.ctc.log_softmax(enc).detach().squeeze(0).cpu()
        index_duration = speech.shape[0] / probs.shape[0] / self.sr
        config = CtcSegmentationParameters(
            char_list=self.char_list,
            index_duration=index_duration,
        )
        token_ints = []
        unk = self.char_list.index("<unk>")
        if self.language == "zh-cn":
            for ch in asr_result:
                try:
                    token_id = self.char_list.index(ch)
                    token_id = np.array([token_id])
                    token_ints.append(token_id)
                except:
                    token_id = unk # skip unk
        elif self.language == "en-us":
            for word in asr_result:
                tokens = self.sp.EncodeAsPieces(word)
                token_ids = []
                for t in tokens:
                    try:
                        token_id = self.char_list.index(t)
                        token_ids.append(token_id)
                    except:
                        token_id = unk # skip unk
                token_ints.append(np.array(token_ids))
        ground_truth_mat, utt_begin_indices = prepare_token_list(config, token_ints)
        timings, char_probs, state_list = ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, asr_result)
        res = []
        for i,(t,p) in enumerate(zip(asr_result, segments)):
            if self.language == "zh-cn": # add 0.11s offset to end timings of chinese characters
                res.append({
                    "text" : t, 
                    "start" : p[0] if i == 0 else p[0] + 0.11, 
                    "end" : min(p[1] + 0.11, speech.shape[0] / self.sr), 
                    "conf" : np.exp(p[2])
                })
            elif self.language == "en-us":
                res.append({
                    "text" : t, 
                    "start" : p[0], 
                    "end" : p[1], 
                    "conf" : np.exp(p[2])
                })
        return res
    
    @typechecked
    @staticmethod
    @torch.no_grad()
    def asr(speech: np.ndarray | Path, language: str = "en-us", device: str = "cpu") -> str:
        if type(speech) == Path:
            speech, _ = librosa.load(speech, sr=16000)
        speech = torch.tensor(speech)
        if language == "zh-cn":
            model_name = "espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char"
        elif language == "en-us":
            model_name = "kamo-naoyuki/librispeech_asr_train_asr_conformer5_raw_bpe5000_scheduler_confwarmup_steps25000_batch_bins140000000_optim_conflr0.0015_initnone_accum_grad2_sp_valid.acc.ave"
        model = Speech2Text.from_pretrained(
            model_tag = model_name,
            device = device,
        )
        res, *_ = model(speech)[0]
        return res
    
    @typechecked
    def get_asr_result(self) -> str:
        return self.asr_result