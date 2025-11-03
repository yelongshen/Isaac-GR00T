import os
from pathlib import Path

import numpy as np
import torch
import gr00t

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy

# change the following paths
MODEL_PATH = "nvidia/GR00T-N1.5-3B"

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
EMBODIMENT_TAG = "gr1"

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

from gr00t.experiment.data_config import DATA_CONFIG_MAP


data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

# print out the policy model architecture
print(policy.model)

modality_config = policy.modality_config

print(modality_config.keys())

for key, value in modality_config.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)

# Create the dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)
print(f"Dataset length: {len(dataset)}")

step_data = dataset[0]

print(step_data)

print("\n\n ====================================")
for key, value in step_data.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)


import matplotlib.pyplot as plt

traj_id = 0
max_steps = 150

state_joints_across_time = []
gt_action_joints_across_time = []
images = []

sample_images = 6

output_dir = Path(REPO_PATH) / "getting_started" / "inference_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

for step_count in range(max_steps):
    data_point = dataset.get_step_data(traj_id, step_count)
    state_joints = data_point["state.right_arm"][0]
    gt_action_joints = data_point["action.right_arm"][0]
    
   
    state_joints_across_time.append(state_joints)
    gt_action_joints_across_time.append(gt_action_joints)

    # We can also get the image data
    if step_count % (max_steps // sample_images) == 0:
        image = data_point["video.ego_view"][0]
        images.append(image)

# Size is (max_steps, num_joints == 7)
state_joints_across_time = np.array(state_joints_across_time)
gt_action_joints_across_time = np.array(gt_action_joints_across_time)


# Plot the joint angles across time
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 2 * 7))

for i, ax in enumerate(axes):
    ax.plot(state_joints_across_time[:, i], label="state joints")
    ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
    ax.set_title(f"Joint {i}")
    ax.legend()

plt.tight_layout()
joint_plot_path = output_dir / "joint_trajectories.png"
fig.savefig(joint_plot_path, dpi=200)
plt.close(fig)
print(f"Saved joint plot to {joint_plot_path}")


# Plot the images in a row
num_images = len(images)
if num_images > 0:
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(16, 4))

    # Ensure axes is iterable even when there's only one subplot
    axes_iter = axes if isinstance(axes, np.ndarray) else [axes]
    for idx, ax in enumerate(axes_iter):
        ax.imshow(images[idx])
        ax.axis("off")

    plt.tight_layout()
    grid_path = output_dir / "ego_view_grid.png"
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)
    print(f"Saved image grid to {grid_path}")

    for idx, image in enumerate(images):
        frame = image
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        frame = np.clip(frame, 0.0, 1.0)
        frame_path = output_dir / f"ego_view_frame_{idx:02d}.png"
        plt.imsave(frame_path, frame)
        print(f"Saved frame {idx} to {frame_path}")
else:
    print("No images collected to save.")

predicted_action = policy.get_action(step_data)
for key, value in predicted_action.items():
    print(key, value.shape)
