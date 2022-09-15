import io
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def get_default_config(tool_name: str) -> dict:
    """
    Get default configuration for a tool.

    Args:
        tool_name: name of the tool.

    Returns:
        Python dictionary containing the config.
    """
    path = Path(__file__).parent / "example_configs" / f"{tool_name}.json"
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def get_codec_name(file : Path, mode : str = 'audio') -> str:
    """
    Get video/audio codec of the file.

    Args:
        file: Path to the file.
        mode: Should be 'video' or 'audio'.

    Returns:
        codec: Codec name.
        
    """
    assert mode in ['audio', 'video'], "Parameter 'mode' must be 'audio' or 'video'."

    args = ['ffprobe', '-show_format', '-show_streams', '-of', 'json', str(file)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffprobe", out, err)
    prob_result = json.loads(out.decode('utf-8'))
    for track in prob_result['streams']:
        if track['codec_type'] == mode:
            return track['codec_name']


def get_video_size(file : Path) -> tuple[int, int]:
    """
    Get height and width of video file.

    Args:
        file: Path to video file.

    Returns:
        height: Video height.
        width: Video width.
    """
    args = ['ffprobe', '-select_streams', 'v', '-show_entries', 'stream=height,width', '-of', 'json', '-i', str(file)]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffprobe", out, err)
    prob_result = json.loads(out.decode('utf-8'))
    return prob_result['streams'][0]['height'], prob_result['streams'][0]['width']


def ffmpeg_extract(in_file : Path, out_path : Path, mode : str = 'audio', fps : int = 10) -> None:
    """
    Extract audio/image from input video file and save to disk.

    Args:
        in_file: Path to the input file.
        out_path: Path to the output file.
        mode: Should be 'audio' or 'image'.
        fps: Frames per second, will be ignored if mode is 'audio'.

    """
    assert mode in ['audio', 'image'], "Parameter 'mode' must be 'audio' or 'image'."
    
    if mode == 'audio':
        args = ['ffmpeg', '-i', str(in_file), '-vn', '-acodec', 'pcm_s16le', '-y', str(out_path)]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ffmpeg", out, err)
    elif mode == 'image':
        args = ['ffmpeg', '-i', str(in_file), '-vf', f'fps={fps}', '-y', str(out_path / '%03d.bmp')]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ffmpeg", out, err)


def ffmpeg_extract_fast(in_file : Path, mode : str = 'audio', fps : int = 10) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, int]:
    """
    Extract audio/image from input video file and return the numpy array.

    This function runs completely in memory without any disk IO, thus is faster than `ffmpeg_extract`.

    Args:
        in_file: Path to the input file.
        mode: Should be 'audio' or 'image'.
        fps: Frames per second, will be ignored if mode is 'audio'.

    Returns:
        data: Numpy array containing the data.
        sr: Audio sample rate.
        height: Video height.
        width: Video width.
    """
    assert mode in ['audio', 'image'], "Parameter 'mode' must be 'audio' or 'image'."
    
    if mode == 'audio':
        # raw mono audio with original sample rate
        args = ['ffmpeg', '-i', str(in_file), '-vn', '-ac', '1', '-acodec', 'pcm_s16le', '-f', 'wav', 'pipe:']
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ffmpeg", out, err)
        data, sr = sf.read(io.BytesIO(out))
        return data, sr
    elif mode == 'image':
        # extract raw RGB images with given fps
        height, width = get_video_size(in_file)
        args = ['ffmpeg', '-i', str(in_file), '-vf', f'fps={fps}', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:']
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ffmpeg", out, err)
        data = np.frombuffer(out, dtype=np.uint8).reshape([-1, height, width, 3])
        return data, height, width


def download_file(url : str, save_path : Path) -> None:
    """
    Download file from url.

    Args:
        url: Url of file to be downloaded.
        save_path: Save path, including filename and extension.

    """
    with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', -1))
        if total_size < 0:
            print("Unknown file size")
            data = response.read()
            out_file.write(data)
        else:
            print("Downloading: %s Bytes: %s" % (save_path, total_size))
            pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
            while True:
                data = response.read(1024)
                if not data:
                    break
                out_file.write(data)
                pbar.update(len(data))
            pbar.close()

