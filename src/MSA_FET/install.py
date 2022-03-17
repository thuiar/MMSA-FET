import platform
import sys
import os
from pathlib import Path
import shutil

import gdown

OPENFACE_BASE = "https://drive.google.com/file/d/17T1WSbkwBokTqNq1IKu51oMX8P122pu-/view?usp=sharing"
OPENFACE_LINUX = "https://drive.google.com/file/d/1Y-WX9xoCYVQqy-VYXyA3697RCiSAldml/view?usp=sharing"
OPENFACE_WIN64 = "https://drive.google.com/file/d/1oPTB5m7K8QPR7o1E6nWMDmSaNkBBF93g/view?usp=sharing"
PRETRAINED = "https://drive.google.com/file/d/1Yzetew3mRjQqhhHLsZP1uT6zT9lR_DOG/view?usp=sharing"

# Check file availability
def integrity_check():
    ext_dir = Path(__file__).parent / "exts"
    if not ext_dir.exists():
        ext_dir.mkdir()
    openface_dir = ext_dir / "OpenFace"
    pretrained_dir = ext_dir / "pretrained"
    res = {}
    res["OpenFace"] = True if openface_dir.exists() else False
    res["pretrained"] = True if pretrained_dir.exists() else False
    return res

# Download from Google Drive and extract
def download_and_extract(url, save_path, remove_zip=True, proxy=None):
    gdown.download(url, save_path, quiet=False, fuzzy=True, proxy=proxy)
    gdown.extractall(save_path)
    if remove_zip:
        Path(save_path).unlink()

# Download OpenFace
def download_openface(proxy=None):
    is_64bits = sys.maxsize > 2**32 # check if system is 64-bit
    if not is_64bits:
        print("We only provide 64-bit version of OpenFace. For 32-bit version, please download the source code and complie it manually.")
        return
    system = platform.system()
    if system == "":
        print("Cannot determine system type. Please download OpenFace manually.")
        return
    ext_dir = Path(__file__).parent / "exts"
    print("Downloading OpenFace...")
    download_and_extract(OPENFACE_BASE, str(ext_dir / "OpenFace_Base.zip"), proxy=proxy)
    if system == "Linux":
        download_and_extract(OPENFACE_LINUX, str(ext_dir / "OpenFace" / "OpenFace_Linux.zip"), proxy=proxy)
        os.chmod(str(ext_dir / "OpenFace"  / "FeatureExtraction"), 0o755)
    if system == "Windows":
        download_and_extract(OPENFACE_WIN64, str(ext_dir / "OpenFace" / "OpenFace_Win64.zip"), proxy=proxy)
    print("Done.")

# Download pretrained models
def download_pretrained(proxy=None):
    ext_dir = Path(__file__).parent / "exts"
    download_and_extract(PRETRAINED, str(ext_dir / "pretrained.zip"), proxy=proxy)


def download_missing(proxy=None):
    # For first run. Will only download if the whole folder is missing. 
    res = integrity_check()
    if not res["OpenFace"]:
        download_openface(proxy=proxy)
    if not res["pretrained"]:
        download_pretrained(proxy=proxy)

def force_redownload(proxy=None):
    ext_dir = Path(__file__).parent / "exts"
    ext_dir.mkdir("")
    shutil.rmtree(ext_dir / "OpenFace")
    download_openface(proxy=proxy)
    shutil.rmtree(ext_dir / "pretrained")
    download_pretrained(proxy=proxy)
