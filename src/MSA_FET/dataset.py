import json
import multiprocessing
import os
import pickle
import shutil
import signal
import time
from functools import wraps
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .aligner import *
from .ASD import run_ASD
from .extractors import *
from .mp_logger import *
from .utils import *


def handle_ctrl_c(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler) # the default
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                ctrl_c_entered = True
                return KeyboardInterrupt()
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt()
    return wrapper

def pool_ctrl_c_handler(*args, **kwargs):
    global ctrl_c_entered
    ctrl_c_entered = True

def init_pool(extractors, config, log_queue, temp_dir, work_dir):
    # Set global variables in child processes
    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

    global logger
    logger = init_log_worker(log_queue)
    global cfg
    cfg = config
    global tmp_dir
    tmp_dir = temp_dir
    global dataset_dir
    dataset_dir = work_dir
    global audio_extractor
    audio_extractor = extractors['audio'](config['audio'], logger) if extractors['audio'] else None
    global video_extractor
    video_extractor = extractors['video'](config['video'], logger) if extractors['video'] else None
    global text_extractor
    text_extractor = extractors['text'](config['text'], logger) if extractors['text'] else None
    global aligner
    aligner = extractors['align'](config['align'], logger) if extractors['align'] else None
    
    logger.info("Subprocess initialized.")

@handle_ctrl_c
def extract_one(row : pd.Series) -> dict:
    logger = logging.getLogger("FET-Subprocess")
    # use global variables in child process.
    global dataset_dir, tmp_dir, cfg
    global video_extractor, audio_extractor, text_extractor, aligner
    try:
        row = row[1]
        video_id, clip_id, text, label, label_T, label_A, label_V, annotation, mode = \
            row['video_id'], row['clip_id'], row['text'], \
            row['label'], row['label_T'], row['label_A'], \
            row['label_V'], row['annotation'], row['mode']
        cur_id = video_id + '$_$' + clip_id
        tmp_id = video_id + '_' + clip_id # "$" is not allowed in file names
        res = {
            'id': cur_id,
            'raw_text': text,
            'annotations': annotation,
            'regression_labels': label,
            'regression_labels_A': label_A,
            'regression_labels_V': label_V,
            'regression_labels_T': label_T,
            'mode': mode,
        }
        video_path = Path(dataset_dir) / 'Raw' / video_id / (clip_id + '.mp4') # TODO: file extension should be configurable
        assert video_path.exists(), f"Video file {video_path} does not exist"
        # video
        if video_extractor:
            feature_V = extract_video(video_path, tmp_id)
            seq_V = feature_V.shape[0]
            res['vision'] = feature_V
            res['vision_lengths'] = seq_V
        # audio
        if audio_extractor:
            feature_A = extract_audio(video_path, tmp_id)
            seq_A = feature_A.shape[0]
            res['audio'] = feature_A
            res['audio_lengths'] = seq_A
        # text
        if text_extractor:
            feature_T = extract_text(text)
            text_bert = preprocess_text(text)
            res['text'] = feature_T
            res['text_bert'] = text_bert
        if type(res['text_bert']) != np.ndarray:
            res.pop('text_bert')
        # align
        if aligner:
            align_result = aligner.align_with_transcript(video_path, text)
            word_ids = text_extractor.get_word_ids(text)
            feature_A, feature_V = extract_align(
                align_result, word_ids, feature_A, feature_V
            )
            assert feature_A.shape[0] == feature_T.shape[0]
            assert feature_V.shape[0] == feature_T.shape[0]
            res['vision'] = feature_V
            res['audio'] = feature_A
            res['align'] = align_result
        return res
    except Exception as e:
        logger.error(f'An error occurred while extracting features for video {video_id} clip {clip_id}')
        logger.error(f'Ignore error and continue, see log for details.')
        logger.exception(e)
        return row

def extract_video(video_path, video_id):
    logger = logging.getLogger("FET-Subprocess")
    # extract images from video
    out_path = Path(tmp_dir) / video_id
    out_path.mkdir(exist_ok=False)
    fps = cfg['video']['fps']

    if multiface_cfg := cfg['video'].get('multiface') is not None and multiface_cfg['enable'] == True:
        # enable Active Speaker Detection
        run_ASD(video_path, out_path, fps, multiface_cfg)
        if len(out_path.glob('*.jpg')) == 0:
            logger.warning(f'ASD returned empty results for video {video_id}')
            shutil.rmtree(out_path)
            return np.zeros((1,1)) # TODO: return zero tensor with the same dimension as normal features, require calculating the dimension from the config
    else:
        ffmpeg_extract(video_path, out_path, mode='image', fps=fps)

    # extract video features
    video_result = video_extractor.extract(out_path, video_id)
    # delete tmp images
    shutil.rmtree(out_path)
    return video_result

def extract_audio(video_path, video_id):
    # extract audio from video file
    tmp_audio_file = Path(tmp_dir) / (video_id + '.wav')
    ffmpeg_extract(video_path, tmp_audio_file, mode='audio')

    # extract audio features
    audio_result = audio_extractor.extract(tmp_audio_file)
    # delete tmp audio file
    os.remove(tmp_audio_file)
    return audio_result

def extract_text(text):
    text_result = text_extractor.extract(text)
    return text_result

def preprocess_text(text):
    # tokenize text for models that use bert
    token_result = text_extractor.tokenize(text)
    return token_result

def extract_align(align_result, word_ids, feature_A, feature_V):
    word_count = len(align_result)
    audio_timestamp = audio_extractor.get_timestamps()
    df = pd.DataFrame(align_result)
    start = df['start'].values
    end = df['end'].values
    start_idx = np.searchsorted(audio_timestamp, start)
    end_idx = np.searchsorted(audio_timestamp, end)
    tmp_result = np.array([np.mean(feature_A[x:y], axis=0) for x, y in zip(start_idx, end_idx)])
    assert len(tmp_result) == word_count
    # align with text tokens, add zero padding or duplicate features
    aligned_feature_A = []
    for i in word_ids:
        if i is None:
            aligned_feature_A.append(np.zeros(tmp_result.shape[1]))
        else:
            aligned_feature_A.append(tmp_result[i])
    aligned_feature_A = np.asarray(aligned_feature_A)
    video_timestamp = video_extractor.get_timestamps()
    start_idx = np.searchsorted(video_timestamp, start)
    end_idx = np.searchsorted(video_timestamp, end)
    tmp_result = np.array([np.mean(feature_V[x:y], axis=0) for x, y in zip(start_idx, end_idx)])
    assert len(tmp_result) == word_count
    # align with text tokens, add zero padding or duplicate features
    aligned_feature_V = []
    for i in word_ids:
        if i is None:
            aligned_feature_V.append(np.zeros(tmp_result.shape[1]))
        else:
            aligned_feature_V.append(tmp_result[i])
    aligned_feature_V = np.asarray(aligned_feature_V)
    return aligned_feature_A, aligned_feature_V

def read_label_file(dataset_name, dataset_root_dir, dataset_dir):
    # Locate and read label.csv file
    assert dataset_name is not None or dataset_dir is not None, "Either 'dataset_name' or 'dataset_dir' must be specified."
    if dataset_dir: # Use dataset_dir
        dataset_dir = Path(dataset_dir)
        dataset_name = dataset_dir.name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
        if not (dataset_dir / 'label.csv').exists():
            raise FileNotFoundError(f"Label file '{dataset_dir}/label.csv' does not exist.")
        label_df = pd.read_csv(
            dataset_dir / 'label.csv',
            dtype={'clip_id': str, 'video_id': str, 'text': str}
        )
        return label_df, dataset_dir, dataset_name, None
    else: # Use dataset_name
        if dataset_root_dir is None:
            raise ValueError("Dataset root directory is not specified.")
        dataset_root_dir = Path(dataset_root_dir)
        if not dataset_root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory '{dataset_root_dir}' does not exist.")
        try: # Try to locate label.csv according to global dataset config file
            with open(dataset_root_dir / 'config.json', 'r') as f:
                dataset_config_all = json.load(f)
            dataset_config = dataset_config_all[dataset_name]
            label_file = dataset_root_dir / dataset_config['label_path']
        except: # If failed, try to locate label.csv using joined path
            label_file = dataset_root_dir / dataset_name / 'label.csv'
        if not label_file.exists():
            raise FileNotFoundError(f"Label file '{label_file}' does not exist.")
        label_df = pd.read_csv(
            label_file,
            dtype={'clip_id': str, 'video_id': str, 'text': str}
        )
        return label_df, label_file.parent, dataset_name, dataset_config

def padding(feature, MAX_LEN, value='zero', location='end'):
    """
    Parameters:
        mode: 
            zero: padding with 0
            norm: padding with normal distribution
        location: start / end
    """
    assert value in ['zero', 'norm'], "Padding value must be 'zero' or 'norm'"
    assert location in ['start', 'end'], "Padding location must be 'start' or 'end'"

    length = feature.shape[0]
    if length >= MAX_LEN:
        return feature[:MAX_LEN, :]
    
    if value == "zero":
        pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
    elif value == "normal":
        mean, std = feature.mean(), feature.std()
        pad = np.random.normal(mean, std, (MAX_LEN-length, feature.shape[1]))

    feature = np.concatenate((pad, feature), axis=0) if(location == "start") else \
                np.concatenate((feature, pad), axis=0)
    return feature

def paddingSequence(sequences, value, location):
    """
    Pad features to the same length according to the mean length of the features.
    """
    feature_dim = sequences[0].shape[-1]
    lengths = [s.shape[0] for s in sequences]
    # use (mean + 3 * std) as the max length
    final_length = int(np.mean(lengths) + 3 * np.std(lengths))
    final_sequence = np.zeros([len(sequences), final_length, feature_dim])
    for i, s in enumerate(sequences):
        if len(s) != 0:
            final_sequence[i] = padding(s, final_length, value, location)
    return final_sequence, final_length

def save_result(result: dict, out_file : Path):
    logger = logging.getLogger("FET-Dataset")
    if out_file.exists():
        out_file_alt = out_file.parent / (out_file.stem + '_' + str(int(time.time())) + '.pkl')
        logger.warning(f"Output file '{out_file}' already exists. Saving to '{out_file_alt}' instead.")
        out_file = out_file_alt
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Feature file saved: '{out_file}'.")

def remove_tmp_folder(tmp_dir : Path) -> None:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

def run_dataset(
    config : dict | str,
    dataset_name : str = None, 
    dataset_root_dir : Path | str = None, 
    dataset_dir : Path | str = None, 
    out_file : Path | str = None, 
    return_type : str = 'np', 
    num_workers : int = 4,
    padding_value : str = 'zero', 
    padding_location : str = 'end', 
    face_detection_failure : str = 'skip', 
    tmp_dir : Path | str = Path.home() / '.MMSA-FET/tmp',
    log_dir : Path | str = Path.home() / '.MMSA-FET/log',
    log_level : int = logging.INFO,
    progress_q : Queue = None, 
    task_id : int = None
) -> dict:
    """
    Extract features from dataset and save in MMSA compatible format.

    Parameters:
        config: Python dictionary of config, or path to a JSON file, or name of an example config.
        dataset_name: [DEPRECATED] Name of dataset. Either 'dataset_name' or 'dataset_dir' must be specified.
        dataset_root_dir: [DEPRECATED] Root directory of dataset. If specified, will override 'dataset_root_dir' set when initializing MSA-FET.
        dataset_dir: Path to dataset directory. If specified, will override 'dataset_name'. Either 'dataset_name' or 'dataset_dir' must be specified.
        out_file: Output feature file. If not specified, features will be saved under the dataset directory with the name 'feature.pkl'.
        return_type: 'pt' for pytorch tensor, 'np' for numpy array. Default: 'np'.
        num_workers: Number of workers for parallel processing. Default: 4.
        padding_value: Padding value for sequence padding. 'zero' or 'norm'. Default: 'zero'.
        padding_location: Padding location for sequence padding. 'end' or 'start'. Default: 'end'.
        face_detection_failure: Actions to take in case of face detection failure. 'skip' the frame or 'pad' with zeros. Default: 'skip'.
        tmp_dir: Directory for temporary files. Default: '~/.MSA-FET/tmp'.
        log_dir: Log directory. Default: '~/.MSA-FET/log'.
        log_level: Verbose level of stdout. Default: logging.INFO
        progress_q: multiprocessing queue for progress reporting with M-SENA.
        task_id: task id for M-SENA.
    """
    # TODO: Select multiple gpus
    # TODO: Batch processing to accelerate GPU models
    # TODO: add database operation for M-SENA
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    mp_ctx = multiprocessing.get_context('spawn')
    log_q = mp_ctx.Queue(-1)
    log_listener = init_log_listener(log_q, log_dir, log_level)
    log_listener.start()
    logger = init_log_worker(log_q, "FET-Dataset")

    if type(config) == str:
        if Path(config).is_file():
            with open(config, 'r') as f:
                config = json.load(f)
        elif (name := Path(__file__).parent / 'example_configs' / f"{config}.json").is_file():
            with open(name, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Config file {config} does not exist.")
    elif type(config) == dict:
        pass
    else:
        raise ValueError("Invalid argument type for `config`.")

    try:
        label_df, dataset_dir, dataset_name, dataset_config = \
            read_label_file(dataset_name, dataset_root_dir, dataset_dir)
        error_df = pd.DataFrame()
        
        logger.info(f"Extracting features from '{dataset_name}' dataset.")
        logger.info(f"Dataset directory: '{dataset_dir}'")

        report = None
        if type(progress_q) == Queue and task_id is not None:
            report = {'task_id': task_id, 'msg': 'Preparing', 'processed': 0, 'total': 0}
            progress_q.put(report)

        data = {
            "id": [], 
            "audio": [],
            "vision": [],
            "raw_text": [],
            "text": [],
            "text_bert": [],
            "audio_lengths": [], # not included in aligned features
            "vision_lengths": [], # not included in aligned features
            "annotations": [],
            # "classification_labels": [],  # no longer supported by MMSA
            "regression_labels": [],
            'regression_labels_A': [],
            'regression_labels_V': [],
            'regression_labels_T': [],
            'align': [],
            "mode": []
        }

        extractors = {
            "audio": AUDIO_EXTRACTOR_MAP[config['audio']['tool']] if config.get('audio') else None,
            "video": VIDEO_EXTRACTOR_MAP[config['video']['tool']] if config.get('video') else None,
            "text": TEXT_EXTRACTOR_MAP[config['text']['model']] if config.get('text') else None,
            "align": Aligner[config['align']['tool']] if config.get('align') else None
        }

        pool = mp_ctx.Pool(
            processes = num_workers,
            initializer=init_pool,
            initargs=(extractors, config, log_q, tmp_dir, dataset_dir)
        )

        if report is not None:
            report['msg'] = 'Processing'
            report['total'] = len(label_df)
            progress_q.put(report)
        
        for result in (pbar := tqdm(pool.imap_unordered(extract_one, label_df.iterrows(), chunksize=5), total=len(label_df))):
            if type(result) is pd.Series:
                error_df = pd.concat([error_df, result.to_frame().T])
                continue
            for k, v in result.items():
                data[k].append(v)
            if report is not None:
                report['processed'] = pbar.n
                progress_q.put(report)
        
        pool.close()
        pool.join()

        if report is not None:
            report['msg'] = 'Finalizing'
            progress_q.put(report)
        
        # remove unimodal labels if not exist
        for key in ['regression_labels_A', 'regression_labels_V', 'regression_labels_T']:
            if np.isnan(np.sum(data[key])):
                data.pop(key)
        # remove empty features
        for key in ['audio', 'vision', 'text', 'text_bert', 'audio_lengths', 'vision_lengths']:
            if len(data[key]) == 0:
                data.pop(key)
        # remove lengths for aligned feature
        if config.get('align'):
            data.pop("vision_lengths")
            data.pop("audio_lengths")
        else:
            data.pop("align")
        # padding features
        for item in ['audio', 'vision', 'text', 'text_bert']:
            if item in data:
                data[item], final_length = paddingSequence(data[item], padding_value, padding_location)
                if f"{item}_lengths" in data:
                    for i, length in enumerate(data[f"{item}_lengths"]):
                        if length > final_length:
                            data[f"{item}_lengths"][i] = final_length
        # transpose text_bert
        if 'text_bert' in data:
            data['text_bert'] = data['text_bert'].transpose(0, 2, 1)
        # repack features
        idx_dict = {
            mode + '_index': [i for i, v in enumerate(data['mode']) if v == mode]
            for mode in ['train', 'valid', 'test']
        }
        data.pop('mode')
        final_data = {k: {} for k in ['train', 'valid', 'test']}
        for mode in ['train', 'valid', 'test']:
            indexes = idx_dict[mode + '_index']
            for item in data.keys():
                if isinstance(data[item], list):
                    final_data[mode][item] = np.array([data[item][v] for v in indexes])
                else:
                    final_data[mode][item] = data[item][indexes]
        data = final_data
        data['config'] = config
        # convert labels to numpy array

        # convert to pytorch tensors
        if return_type == 'pt':
            for mode in data.keys():
                if mode == 'config':
                    continue
                for key in ['audio', 'vision', 'text', 'text_bert']:
                    if key in data[mode]:
                        data[mode][key] = torch.from_numpy(data[mode][key])
        # save result
        if out_file is None:
            out_file = dataset_dir / 'feature.pkl'
        else:
            out_file = Path(out_file)
        save_result(data, out_file)
        error_df.to_csv(out_file.parent / "error.csv")
        logger.info(f"Feature extraction complete!")
        if report is not None:
            report['msg'] = 'Finished'
            progress_q.put(report)
        log_listener.stop()
        return data
    except KeyboardInterrupt:
        logger.info("User aborted feature extraction!")
        pool.terminate()
        pool.join()
        logger.info("Removing temporary files.")
        remove_tmp_folder(tmp_dir)
        if report is not None:
            report['msg'] = 'Terminated'
            progress_q.put(report)
        log_listener.stop()
    except Exception:
        logger.exception("An Error Occured:")
        pool.terminate()
        pool.join()
        logger.info("Removing temporary files.")
        remove_tmp_folder(tmp_dir)
        if report is not None:
            report['msg'] = 'Error'
            progress_q.put(report)
        log_listener.stop()

