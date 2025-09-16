# VMAF Proxy

This repository provides a deep learning model to estimate frame-level VMAF directly from a reference and a distorted video stream. This repo includes the full data pipeline from **dataset split → compression (x265) → VMAF calculation → frame extraction → CSV metadata → training** as well as cloud-friendly options and seeds for determinism.

**Project status:** Training is in progress. I will publish checkpoints and full benchmarks once convergence is reached. In the meantime, the pipeline is fully reproducible—feel free to generate your own results.

This work builds a **neural proxy** that:

- **Estimates frame-level VMAF** directly from a **reference** and a **distorted** frame stack and outputs a scalar VMAF score
- Uses a **siamese 3D-CNN** backbone for reference and distorted frames plus a third absolute difference (|reference−distorted|) branch. Features are then fused, globally pooled, and fed to a small MLP regressor.
- Operates on **luminance channel** to reduce I/O while preserving perceptual signal

To generate sequences of different quality, I used **x265** as the primary encoder. Other encoders can also be used and tested for performance.

## Note: 
Found a bug or have an improvement? Contributions are welcome! 🙌
1. **Open an issue** with a minimal repro (dataset slice, command line, logs).
2. **Submit a Merge Request / Pull Request** referencing the issue.
3. Please follow this checklist:
   - Clear description of the bug/fix
   - Steps to reproduce (commands, args, env details)
   - Affected files

I will review merge requests and leave feedback or merge when ready. Thanks for helping improve the project! 🚀

# Dataset
This project trains and validates on **YouTube UGC (User-Generated Content) 720p Dataset**. Training on this dataset has a wide variety of advantages such as:
- **Real-world distribution:** UGC covers the long tail of scenes, devices, and capture conditions
- **High content diversity:** Strong variation in motion, texture, lighting, scene dynamics, and camera pipelines improves **generalization** of the learned mapping from input frame to reconstructed frame

 I generated frames of different quality by encoding each raw YUV input sequence with **x265** using CRF rate control mode. I used a wide range of **CRF values: 19, 23, 27, 31, 35, 39, 43, 47, 48, 49, 50, 51** to span low to high quality. To ensure all quality levels are well represented I apply binning and sample an approximately equal number of sequences per bin so training sees a balanced spread from visually pristine to very compressed, improving robustness of the learning.

## Model Architecture
- **Inputs**
  - Two stacks of **luminance channels**: a **Reference** stack and a **Distorted** stack.
  - Each stack has an **odd temporal window** centered at frame *t* so temporal neighbors are symmetric around the center
  - A **shared spatial crop** is applied to both stacks to ensure perfect alignment

- **Backbone (Siamese 3D CNN)**
  - Two **weight-sharing 3D convolutional towers** independently encode the Reference and Distorted stacks.
  - Each tower is composed of **Conv3D → GroupNorm → nonlinearity (+ optional dropout)** blocks with temporal and spatial receptive fields to capture short-range motion and texture cues.
  - Towers output compact spatio-temporal feature volumes

- **Quality-delta feature path**
  - In addition to concatenating reference and distorted feature volumes, the network forms an **absolute difference** feature map to emphasize compression artifacts and detail loss
  - The three tensors above are **channel-wise concatenated** and passed through an additional 3D CNN stage to blend cues

- **Global Aggregation**
  - A **global pooling** layer collapses the spatio-temporal dimensions to a single feature vector per sample (summarizing content and artifact evidence across the crop and frames)

- **Regressor Head**
  - A lightweight **MLP (fully connected stack)** maps the pooled features to a **single scalar** prediction
  - The training pipeline interprets the prediction as **normalized VMAF**; loss/metrics are computed after **clamping to `[0, 1]`** and reported back in **0–100** for interpretability.

- **Design Notes**
  - **Temporal sensitivity** comes from 3D kernels over the frame window (not optical flow), keeping the model simple and deployable
  - **Luma-only** inputs reduce I/O and memory while preserving most perceptual quality signal used by VMAF
  - The architecture is **encoder-agnostic** and it learns from pairs regardless of the codec and CRF used to generate the distorted stream

## Runtime environments

This codebase is **Vertex AI–first**:

- **Primary target**: Google Vertex AI (custom training jobs), GCS storage, and `gs://` paths.
- **Local runs**: It is supported, but you may need to turn off cloud-only features and/or install extra packages.

## Repository layout
```text
vmaf_proxy/
├─ README.md
├─ LICENSE
├─ gitignore
├─ src/
│  └─ vmaf_proxy/
│     ├─ data/
│     │  └─ vmaf_data_loader.py  # VMAFDataset
│     ├─ models/
│     │  └─ model.py          # 3D CNN + SE, fusion head
│     └─ train/
│        └─ train.py             # CLI entry; training loop, metrics, ckpts
├─ scripts/
│  ├─ compress.py              # encode sources at CRFs (x264/x265/SVT-AV1)
│  ├─ compute_vmaf.py          # produce frame-level VMAF JSON logs
│  ├─ extract_frames.py        # dump gray PNG frames for ref/dist
│  ├─ generate_csv.py          # parse VMAF JSON to per-split CSVs
│  ├─ build_groups.py          # combines multiple CSVs into one + balancing
│  └─ dataset_split.py         # split raw videos into train/val/test
```
