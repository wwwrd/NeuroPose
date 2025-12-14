import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm


def allfile(path):
    fileall = []
    for curpath, _, filelist in os.walk(path):
        for i in filelist:
            fileall.append(os.path.join(curpath, i))
    return fileall


def readnplong(filename):
    return np.load(filename, allow_pickle=True)


def event2frame(event, index, frame):
    if event is not None and len(
            event) > 0 and 'y' in event.dtype.names and 'x' in event.dtype.names and 'polarity' in event.dtype.names:
        valid_indices = (event['y'] < frame.shape[2]) & (event['x'] < frame.shape[3])
        frame[index, event['polarity'][valid_indices], event['y'][valid_indices], event['x'][valid_indices]] = 1
    return frame


class Bullying10kPoseDataset(Dataset):
    """
    Dataset loader for Bullying10K capable of loading both Event (.npy) and Pose (.json) data.
    """

    def __init__(self, root, train, class_mapping, keypoints_json_path, step=10, gap=2, pose_sequence_length=16,
                 transform=None):
        super().__init__()
        self.root = root
        self.is_train = train
        self.class_mapping = class_mapping
        self.step = step
        self.gap = gap
        self.pose_sequence_length = pose_sequence_length
        self.transform = transform

        self.video_to_pose_map = self._load_keypoints(keypoints_json_path)
        if not self.video_to_pose_map:
            print(f"Warning: Failed to load pose file {keypoints_json_path}. Pose data will be zero vectors.")

        self.samples = []
        for class_name, label_idx in self.class_mapping.items():
            class_path = os.path.join(self.root, class_name)
            if os.path.isdir(class_path):
                for npy_file in allfile(class_path):
                    if npy_file.endswith('.npy'):
                        self.samples.append((npy_file, label_idx))
            else:
                print(f"Warning: Class directory {class_path} not found.")

        if not self.samples:
            raise FileNotFoundError(f"No .npy files found in {self.root}. Please check the dataset path.")

        all_indices = list(range(len(self.samples)))
        test_indices = {i for i in all_indices if i % 5 == 0}

        if self.is_train:
            self.sample_indices = [i for i in all_indices if i not in test_indices]
        else:
            self.sample_indices = list(test_indices)

    def _load_keypoints(self, json_path):
        """Robust JSON parser grouping keypoints by video ID."""
        if not os.path.exists(json_path):
            print(f"Warning: JSON path does not exist: {json_path}")
            return {}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            images_info = data.get('images', [])
            image_id_to_filename = {img['id']: img['file_name'] for img in images_info}

            video_id_to_kps = {}
            annotations = data.get('annotations', [])

            for ann in annotations:
                image_id = ann.get('image_id')
                keypoints = ann.get('keypoints')

                if image_id is None or keypoints is None:
                    continue

                file_name = image_id_to_filename.get(image_id)
                if not file_name:
                    continue

                # Extract video identifier from path (e.g., "action_name/video_name_folder/frame.png")
                try:
                    video_id = file_name.replace('\\', '/').split('/')[-2]
                    if 'dvSave' not in video_id:
                        continue
                except IndexError:
                    continue

                if video_id not in video_id_to_kps:
                    video_id_to_kps[video_id] = []

                # Store (frame_id, keypoints) for sorting
                frame_id = ann.get('id', image_id)
                video_id_to_kps[video_id].append((frame_id, keypoints))

            # Sort keypoints by frame ID for each video
            for video_id, kps_list in video_id_to_kps.items():
                kps_list.sort(key=lambda x: x[0])
                video_id_to_kps[video_id] = [item[1] for item in kps_list]

            return video_id_to_kps

        except Exception as e:
            import traceback
            print(f"Error: Unknown error processing JSON file {json_path}: {e}")
            traceback.print_exc()
            return {}

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        actual_idx = self.sample_indices[idx]
        event_filepath, label = self.samples[actual_idx]

        # --- A. Load Event Data ---
        try:
            event_chunks = readnplong(event_filepath)
        except Exception:
            event_chunks = []

        if len(event_chunks) >= self.step * self.gap:
            start_idx = random.randint(0, len(event_chunks) - self.step * self.gap)
            sampled_events = [event_chunks[i] for i in range(start_idx, start_idx + self.step * self.gap, self.gap)]
        else:
            sampled_events = event_chunks[::self.gap][:self.step]

        num_found_events = len(sampled_events)
        if num_found_events < self.step:
            if num_found_events == 0:
                dummy_dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('polarity', 'i1')])
                sampled_events = [np.array([], dtype=dummy_dtype) for _ in range(self.step)]
            else:
                padding = [sampled_events[-1]] * (self.step - num_found_events)
                sampled_events.extend(padding)

        event_frames = np.zeros((self.step, 2, 260, 346), dtype=np.float32)
        for i, chunk in enumerate(sampled_events):
            event_frames = event2frame(chunk, i, event_frames)

        event_tensor = torch.from_numpy(event_frames)
        if self.transform:
            event_tensor = self.transform(event_tensor)

        # --- B. Load Pose Data ---
        video_id = os.path.splitext(os.path.basename(event_filepath))[0]
        raw_keypoints_frames = self.video_to_pose_map.get(video_id, [])

        pose_sequence = []
        if raw_keypoints_frames:
            for kps_for_one_frame in raw_keypoints_frames:
                # Assume taking the first person detected in each frame
                xy_kps = [v / (346 if i % 3 == 0 else 260) for i, v in enumerate(kps_for_one_frame) if i % 3 != 2]
                pose_sequence.append(xy_kps)

        pose_feature_dim = 52
        if len(pose_sequence) > self.pose_sequence_length:
            start_idx = random.randint(0, len(pose_sequence) - self.pose_sequence_length)
            pose_sequence = pose_sequence[start_idx: start_idx + self.pose_sequence_length]

        num_found_poses = len(pose_sequence)
        if num_found_poses < self.pose_sequence_length:
            if num_found_poses == 0:
                pose_sequence = [[0.0] * pose_feature_dim for _ in range(self.pose_sequence_length)]
            else:
                padding = [pose_sequence[-1]] * (self.pose_sequence_length - num_found_poses)
                pose_sequence.extend(padding)

        pose_tensor = torch.tensor(pose_sequence, dtype=torch.float32)

        if pose_tensor.shape != (self.pose_sequence_length, pose_feature_dim):
            pose_tensor = torch.zeros(self.pose_sequence_length, pose_feature_dim)

        return event_tensor, label, pose_tensor


def get_bullying10k_data(data_root, batch_size, num_workers=0, use_train_augmentations=True, **kwargs):
    size = kwargs.get("size", 112)
    step = kwargs.get("step", 16)
    gap = kwargs.get("gap", 4)
    # Allow overriding json paths via kwargs, else defaults
    train_json_path = kwargs.get("train_json", os.path.join(data_root, "train_keypoints.json"))
    val_json_path = kwargs.get("val_json", os.path.join(data_root, "val_keypoints.json"))
    
    pose_len = step

    class_folders = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    if not class_folders:
        raise ValueError(f"No class subfolders found in {data_root}.")
    class_names = class_folders
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    class Interpolate(object):
        def __init__(self, size): self.size = size

        def __call__(self, x): return F.interpolate(x, size=(self.size, self.size), mode='bilinear',
                                                    align_corners=False)

    class RandomHorizontalFlip(object):
        def __init__(self, p=0.5): self.p = p

        def __call__(self, x): return torch.flip(x, dims=[-1]) if random.random() < self.p else x

    train_transforms_list = []
    if use_train_augmentations:
        train_transforms_list.append(RandomHorizontalFlip(p=0.5))
    train_transforms_list.append(Interpolate(size))

    train_transform = transforms.Compose(train_transforms_list)
    test_transform = transforms.Compose([Interpolate(size)])

    train_dataset = Bullying10kPoseDataset(root=data_root, train=True, class_mapping=class_to_idx,
                                           keypoints_json_path=train_json_path, step=step, gap=gap,
                                           pose_sequence_length=pose_len, transform=train_transform)

    test_val_dataset = Bullying10kPoseDataset(root=data_root, train=False, class_mapping=class_to_idx,
                                              keypoints_json_path=val_json_path, step=step, gap=gap,
                                              pose_sequence_length=pose_len, transform=test_transform)

    test_val_size = len(test_val_dataset)
    indices = list(range(test_val_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_indices = indices[:test_val_size // 2]
    test_indices = indices[test_val_size // 2:]

    val_dataset = Subset(test_val_dataset, val_indices)
    final_test_dataset = Subset(test_val_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(final_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    print(f"Data Loaded: Train {len(train_dataset)} | Val {len(val_dataset)} | Test {len(final_test_dataset)}")

    return train_loader, val_loader, test_loader, num_classes, class_names