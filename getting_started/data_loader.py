from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import EmbodimentTag

import os
import gr00t

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATA_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")

print("Loading dataset... from", DATA_PATH)

# 2. modality configs
modality_configs = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["video.ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "state.left_arm",
            "state.left_hand",
            "state.left_leg",
            "state.neck",
            "state.right_arm",
            "state.right_hand",
            "state.right_leg",
            "state.waist",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "action.left_hand",
            "action.right_hand",
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description", "annotation.human.validity"],
    ),
}


# 3. gr00t embodiment tag
embodiment_tag = EmbodimentTag.GR1

print('tag', embodiment_tag)

# load the dataset
dataset = LeRobotSingleDataset(DATA_PATH, modality_configs,  embodiment_tag=embodiment_tag)

print('\n'*2)
print("="*100)
print(f"{' Humanoid Dataset ':=^100}")
print("="*100)

print('dataset', dataset)

# print the 7th data point
resp = dataset[7]
any_describe(resp)
print(resp.keys())

import numpy as np
import matplotlib.pyplot as plt


SAVE_DIR = os.path.join(REPO_PATH, "getting_started", "sample_frames")
os.makedirs(SAVE_DIR, exist_ok=True)

for i in range(100):
    if i % 10 == 0:
        resp = dataset[i]
        frame = resp["video.ego_view"][0]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()

        print(frame.shape, frame.dtype, frame.min(), frame.max())

        #frame = np.clip(frame, 0.0, 1.0)
        save_path = os.path.join(SAVE_DIR, f"ego_view_{i:04d}.png")
        plt.imsave(save_path, frame)

print(f"Saved {len(range(0, 100, 10))} frames to {SAVE_DIR}")

