# MMSA-Feature Extraction Toolkit

[![](https://badgen.net/badge/license/GPL-3.0/green)](#License) 
[![](https://badgen.net/pypi/v/MMSA-FET)](https://pypi.org/project/MMSA-FET/) 
![](https://badgen.net/pypi/python/MMSA-FET/)
[![](https://badgen.net/badge/contact/THUIAR/purple)](https://thuiar.github.io/)

MMSA-Feature Extraction Toolkit extracts multimodal features for Multimodal Sentiment Analysis Datasets. It integrates several commonly used tools for visual, acoustic and text modality. The extracted features are compatible with the [MMSA](https://github.com/thuiar/MMSA) Framework and thus can be used directly. The tool can also extract features for single videos.

This work is included in the ACL-2022 DEMO paper: [M-SENA: An Integrated Platform for Multimodal Sentiment Analysis](). If you find our work useful, don't hesitate to cite our paper. Thank you!

```text
@article{mao2022m,
  title={M-SENA: An Integrated Platform for Multimodal Sentiment Analysis},
  author={Mao, Huisheng and Yuan, Ziqi and Xu, Hua and Yu, Wenmeng and Liu, Yihe and Gao, Kai},
  journal={arXiv preprint arXiv:2203.12441},
  year={2022}
}
```

### Features

- Extract fully customized features for single videos or datasets. 
- Integrate some most commonly used tools, including Librosa, OpenFace, Transformers, etc. 
- Support Active Speaker Detection in case multiple faces exists in a video. 
- Easy to use, provides Python APIs and commandline tools. 
- Extracted features are compatible with [MMSA](https://github.com/thuiar/MMSA), a unified training & testing framework for Multimodal Sentiment Analysis.

## 1. Installation

MMSA-Feature Extraction Toolkit is available from PyPI. Due to package size limitation on PyPi, large model files cannot be shipped with the package. Users need to run a post install command to download these files manually. If you can't access Google Drive, please refer to [this page](https://github.com/FlameSky-S/MMSA-FET/wiki/Dependency-Installation#2-post-installation-script) for manual download. 

```bash
# Install package from PyPI
$ pip install MMSA-FET
# Download models & libraries from Google Drive. Use --proxy if needed.
$ python -m MSA_FET install
```

> **Note:** A few system-wide dependancies need to be installed manually. See [Dependency Installation](https://github.com/FlameSky-S/MMSA-FET/wiki/Dependency-Installation) for more information.

## 2. Quick Start

MMSA-FET is fairly easy to use. You can either call API in python or use commandline interface. Below is a basic example using python APIs.

> **Note:** To extract features for datasets, the datasets need to be organized in a specific file structure, and a `label.csv` file is needed. See [Dataset and Structure](https://github.com/FlameSky-S/MMSA-FET/wiki/Dataset-and-Structure) for details. Raw video files and label files for MOSI, MOSEI and CH-SIMS can be downloaded from [BaiduYunDisk](https://pan.baidu.com/s/1XmobKHUqnXciAm7hfnj2gg) `code: mfet` or [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk?usp=sharing).

```python
from MSA_FET import FeatureExtractionTool
from MSA_FET import run_dataset

# initialize with default librosa config which only extracts audio features
fet = FeatureExtractionTool("openface")

# alternatively initialize with a custom config file
fet = FeatureExtractionTool("custom_config.json")

# extract features for single video
feature1 = fet.run_single("input1.mp4")
print(feature1)
feature2 = fet.run_single("input2.mp4")

# extract for dataset & save features to file
run_dataset(
    config = "aligned",
    dataset_dir="~/MOSI", 
    out_file="output/feature.pkl",
    num_workers=4
)
```

The `custom_config.json` is the path to a custom config file, the format of which is introduced below.

For detailed usage, please read [APIs](https://github.com/FlameSky-S/MMSA-FET/wiki/APIs) and [Command Line Arguments](https://github.com/FlameSky-S/MMSA-FET/wiki/Command-Line-Arguments).

## 3. Config File

MMSA-FET comes with a few example configs which can be used like below.

```python
# Each supported tool has an example config
fet = FeatureExtractionTool(config="aligned")
fet = FeatureExtractionTool(config="librosa")
fet = FeatureExtractionTool(config="opensmile")
fet = FeatureExtractionTool(config="wav2vec")
fet = FeatureExtractionTool(config="openface")
fet = FeatureExtractionTool(config="mediapipe")
fet = FeatureExtractionTool(config="bert")
fet = FeatureExtractionTool(config="roberta")
```

For customized features, you can: 

1. Edit the default configs and pass a dictionary to the config parameter like the example below:

```python
from MSA_FET import FeatureExtractionTool, get_default_config

# here we only extract audio and video features
config_a = get_default_config('opensmile')
config_v = get_default_config('openface')

# modify default config
config_a['audio']['args']['feature_level'] = 'LowLevelDescriptors'

# combine audio and video configs
config = {**config_a, **config_v}

# initialize
fet = FeatureExtractionTool(config=config)
```

2. Provide a config json file. The below example extracts features of all three modalities. To extract unimodal features, just remove unnecessary sections from the file.

```json
{
  "audio": {
    "tool": "librosa",
    "sample_rate": null,
    "args": {
      "mfcc": {
        "n_mfcc": 20,
        "htk": true
      },
      "rms": {},
      "zero_crossing_rate": {},
      "spectral_rolloff": {},
      "spectral_centroid": {}
    }
  },
  "video": {
    "tool": "openface",
    "fps": 25,
    "average_over": 3,
    "args": {
      "hogalign": false,
      "simalign": false,
      "nobadaligned": false,
      "landmark_2D": true,
      "landmark_3D": false,
      "pdmparams": false,
      "head_pose": true,
      "action_units": true,
      "gaze": true,
      "tracked": false
    }
  },
  "text": {
    "model": "bert",
    "device": "cpu",
    "pretrained": "models/bert_base_uncased",
    "args": {}
  }
}
```

## 4. Supported Tools & Features

### 4.1 Audio Tools

- **Librosa** ([link](https://librosa.org/doc/latest/index.html))

  Supports all librosa features listed [here](https://librosa.org/doc/latest/feature.html), including: [mfcc](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html#librosa.feature.mfcc), [rms](https://librosa.org/doc/latest/generated/librosa.feature.rms.html#librosa.feature.rms), [zero_crossing_rate](https://librosa.org/doc/latest/generated/librosa.feature.zero_crossing_rate.html#librosa.feature.zero_crossing_rate), [spectral_rolloff](https://librosa.org/doc/latest/generated/librosa.feature.spectral_rolloff.html#librosa.feature.spectral_rolloff), [spectral_centroid](https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html#librosa.feature.spectral_centroid), etc. Detailed configurations can be found [here](https://github.com/FlameSky-S/MMSA-FET/wiki/Configurations#11-librosa).

- **openSMILE** ([link](https://audeering.github.io/opensmile-python/))

  Supports all feature sets listed [here](https://audeering.github.io/opensmile-python/api-smile.html#featureset), including: [ComParE_2016](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf), [GeMAPS](https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf), [eGeMAPS](https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf), emobase, etc. Detailed configurations can be found [here](https://github.com/FlameSky-S/MMSA-FET/wiki/Configurations#12-opensmile).

- **Wav2vec2** ([link](https://huggingface.co/docs/transformers/model_doc/wav2vec2))

  Integrated from huggingface transformers. Detailed configurations can be found [here](https://github.com/FlameSky-S/MMSA-FET/wiki/Configurations#13-wav2vec2).

### 4.2 Video Tools

- **OpenFace** ([link](https://github.com/TadasBaltrusaitis/OpenFace))

  Supports all features in OpenFace's FeatureExtraction binary, including: facial landmarks in 2D and 3D, head pose, gaze related, facial action units, HOG binary files. Details of these features can be found in the OpenFace Wiki [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format) and [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units). Detailed configurations can be found [here](https://github.com/FlameSky-S/MMSA-FET/wiki/Configurations#21-openface).

- **MediaPipe** ([link](https://google.github.io/mediapipe/))

  Supports [face](https://google.github.io/mediapipe/solutions/face_mesh.html) mesh and [holistic](https://google.github.io/mediapipe/solutions/holistic)(face, hand, pose) solutions. Detailed configurations can be found [here](https://github.com/FlameSky-S/MMSA-FET/wiki/Configurations#22-mediapipe).

- **TalkNet**([link](https://github.com/TaoRuijie/TalkNet_ASD))

  TalkNet is used to support Active Speaker Detection in case there are multiple human faces in the video. 

### 4.3 Text Tools

- **BERT** ([link](https://huggingface.co/docs/transformers/model_doc/bert))

  Integrated from huggingface transformers. Detailed configurations can be found [here](https://github.com/FlameSky-S/MMSA-FET/wiki/Configurations#31-bert).

- **XLNet** ([link](https://huggingface.co/docs/transformers/model_doc/xlnet))

  Integrated from huggingface transformers. Detailed configurations can be found [here](https://github.com/FlameSky-S/MMSA-FET/wiki/Configurations#32-xlnet).

### 4.4 Aligners

- **Wav2vec CTC Aligner**

  Using pretrained Wav2vec ASR model to generate timestamps for each word, then align video & audio with text. Currently only support English.
