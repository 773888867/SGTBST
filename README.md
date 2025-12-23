# SGT-BST

### A Graph Neural Network Approach for Multi-Object Tracking of Solid-Colored Cattle in Controlled Pasture Settings

**Authors:** Ji Li, Boyu Liu, Kejian Wang*, Lingling Liu, Yongsheng Si, Zhenxue He

## Introduction

This is the official implementation of **SGT-BST**, an improved tracker based on [SGT](https://github.com/HYUNJS/SGT). It is designed for tracking **solid-colored cattle** (e.g., Black Angus) and achieves **91.17% MOTA** and **94.61% IDF1** on the CVB dataset.

### Key Features
* **PSA (Part-based Sequential Aggregation):** Uses GRU gating to fuse body part features, significantly improving recovery from long-term occlusion.
* **ITA (Inter-frame Target Association):** Balances stable and dynamic cues to handle feature inconsistency during rapid movement.
* **DRWN (Dynamic Relation Weighting Network):** Enhances tracking robustness in dense/overlapping scenarios via hierarchical feature coordination.

## Results on CVB Dataset

| Method | MOTA | IDF1 | FP | FN | IDS |
|--------|------|------|----|----|-----|
| SGT (baseline) | 87.38% | 88.00% | 711 | 886 | 259 |
| **SGT-BST (ours)** | **91.17%** | **94.61%** | **375** | **818** | **108** |

## Installation

Please refer to the original [SGT Installation Guide](https://github.com/HYUNJS/SGT/blob/main/INSTALL.md) for detailed installation instructions.

### Quick Setup

```bash
# Create conda environment
conda create -n sgtbst python=3.8
conda activate sgtbst

# Install PyTorch (adjust cuda version as needed)
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Install other dependencies
pip install opencv-python numpy scipy

```

### Dataset Setup
This work uses the CVB (C-attle Visual Behaviors) dataset. Follow the dataset preparation from SGT DATASET.md.

**Preparation Steps:**
1. **Convert to MOT17:** Convert the raw CVB data into standard **MOT17 format**.
2. **Follow SGT Pipeline:** Organize the directory and run the data preparation scripts (e.g., `gen_trainval.py`, `gen_labels_mot.py`) following the original **[SGT DATASET.md](https://github.com/HYUNJS/SGT/blob/main/DATASET.md)**.

**Directory Structure:**
Ensure your workspace looks like this after processing:
```text
DATA_ROOT/
    └── MOT17/          # Organized in MOT17 format
        ├── train/    # Contains seqinfo.ini, gt/, img1/
        └── test/
```

### Training
```bash
python projects/SGTBST/train_net.py
--config-file projects/SGTBST/configs/MOT17/sgt_bst.yaml
--data-dir /path/to/datasets
--num-gpus 2
```

### Inference
```bash
python projects/SGTBST/train_net.py
--config-file projects/SGTBST/configs/MOT17/sgt_bst.yaml
--data-dir /path/to/datasets
--num-gpus 1
--eval-only
```

##  Acknowledgement
This code is built upon SGT and Detectron2. We thank the authors for their excellent work.

## Citation
If you find this work useful, please cite:

```bibtex
@article{wang2025sgtbst,
title={Multi-Object Tracking Method for Ranch Cattle Based on SGT-BST},
author={Wang, Kejian and Li, Ji and Liu, Boyu and Si, Yongsheng and Liu, Lingling and Gao, Yuan},
journal={[Your Journal Name]},
year={2025}
}
```

And also cite the original SGT paper:

```bibtex
@inproceedings{hyun2023detection,
title={Detection recovery in online multi-object tracking with sparse graph tracker},
author={Hyun, Jeongseok and Kang, Myunggu and Wee, Dongyoon and Yeung, Dit-Yan},
booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
pages={4850--4859},
year={2023}
}
```

License
This project follows the same license as SGT (CC-BY-NC 4.0) for research and academic purposes.