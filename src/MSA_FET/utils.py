import subprocess
import json
import os.path as osp
import urllib.request
from tqdm import tqdm
import re
from pathlib import Path


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


def get_codec_name(file, mode):
    """
    Function:
        Get video/audio codec of the file.

    Parameters:
        file: Path to the file.
        mode: Should be 'video' or 'audio'.

    Returns:
        codec: Codec name.
        
    """
    assert mode in ['audio', 'video'], "Parameter 'mode' must be 'audio' or 'video'."

    args = ['ffprobe', '-show_format', '-show_streams', '-of', 'json']
    args += [file]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffprobe", out, err)
    prob_result = json.loads(out.decode('utf-8'))
    for track in prob_result['streams']:
        if track['codec_type'] == mode:
            return track['codec_name']


def ffmpeg_extract(in_file, out_path, mode='audio', fps=25):
    """
    Function:
        Extract audio/image from the input file.

    Params:
        in_file: Path to the input file.
        out_path: Path to the output file.
        mode: Should be 'audio' or 'image'.
        fps: Frames per second, will be ignored if mode is 'audio'.

    """
    assert mode in ['audio', 'image'], "Parameter 'mode' must be 'audio' or 'image'."
    
    if mode == 'audio':
        # For `out_path`, better use m4a as output format, 
        # aac will lose timestamps and may result in shorter audio files
        args = ['ffmpeg', '-i', in_file, '-vn', '-acodec', 'copy', '-y', out_path]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ffprobe", out, err)
    elif mode == 'image':
        args = ['ffmpeg', '-i', in_file, '-r', f'{fps}/1', osp.join(out_path, '%03d.bmp')]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("ffprobe", out, err)


def download_file(url, save_path):
    """
    Function:
        Download file from url.

    Params:
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


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]