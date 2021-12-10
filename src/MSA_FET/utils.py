import ffmpeg
import os.path as osp
import urllib.request
from tqdm import tqdm
import re


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

    prob_result = ffmpeg.probe(file)
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
        ffmpeg.input(in_file).output(out_path)\
            .run(overwrite_output=True, quiet=True)
    elif mode == 'image':
        ffmpeg.input(in_file)\
            .output(osp.join(out_path, '%03d.bmp'), r=f'{fps}/1')\
            .run(quiet=True)


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