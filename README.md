# Neuro-Pose
Paper code: Neuromorphic bullying detection network based on multi-scale spatio-temporal attention and asymmetric gated fusion

![Neuro-Pose Architecture](img/model.png)


# Environment

* Ubuntu 18.04 / 20.04
* Python 3.8+
* PyTorch 1.10.0+ (Tested on 2.0.0 + CUDA 11.8)
* Dependencies can be installed via:

```bash
pip install -r requirements.txt

# Dataset

### An example of the dataset directory structure is organized as follows, where each folder contains:

```text
/dataset_root
    ├── train_keypoints.json
    ├── val_keypoints.json
    ├── Fingerguess/
    │   ├── video_001.npy
    │   └── ...
    ├── Kicking/
    │   ├── video_102.npy
    │   └── ...
    └── ...
```

* .npy files represent the event streams transformed into frames.
* .json files represent the pose keypoints (COCO format).
* You should follow the format and organize your own data in a `dataset` folder.

### Training sets

* [Bullying10K](https://www.brain-cog.network/dataset/Bullying10k/)

### Test sets

* [Bullying10K](https://www.brain-cog.network/dataset/Bullying10k/)
