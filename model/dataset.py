"""Dataset loader for Le World Model Two Rooms dataset."""

import os

# Allow multiple processs to read the H5s, okay because we are just loading data
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import hdf5plugin  # Required for compressed HDF5 datasets
import torch
from torch.utils.data import Dataset
import numpy as np


class TwoRoomsDataset(Dataset):
    """Dataset for Two Rooms environment from HDF5 file.

    The HDF5 file contains:
    - observations: RGB images (frames)
    - actions: Environment actions
    """

    def __init__(
        self,
        h5_path: str,
        sequence_length: int = 5,
        transform=None,
        max_episodes: int = None,
    ):
        """
        Args:
            h5_path: Path to the HDF5 file
            sequence_length: Number of frames in each sequence (N)
            transform: Optional transform to apply to images
            max_episodes: Maximum number of episodes to load (for debugging)
        """
        self.h5_path = h5_path
        self.sequence_length = sequence_length
        self.transform = transform

        # Per-process file handle, opened lazily on first access so it is never
        # shared across a DataLoader-worker fork (which corrupts HDF5 reads).
        self._h5 = None

        # Open HDF5 file and load metadata
        with h5py.File(h5_path, 'r') as f:
            # Check what keys are in the file
            self.keys = list(f.keys())
            print(f"HDF5 file contains keys: {self.keys}")

            # Check for Two Rooms specific structure (flat with ep_idx/ep_offset)
            if 'pixels' in f and 'ep_idx' in f and 'ep_offset' in f:
                # Two Rooms dataset structure
                ep_indices = f['ep_idx'][:]
                ep_offsets = f['ep_offset'][:]
                ep_lengths = f['ep_len'][:]

                # Build episode information
                self.num_episodes = len(ep_offsets)
                if max_episodes:
                    self.num_episodes = min(self.num_episodes, max_episodes)

                self.episode_offsets = ep_offsets[:self.num_episodes]
                self.episode_lengths = ep_lengths[:self.num_episodes]
                self.use_flat_structure = True

            # Fallback to standard structures
            elif 'observations' in f:
                # Flat structure
                self.num_episodes = 1
                self.episode_lengths = [len(f['observations'])]
                self.episode_offsets = [0]
                self.use_flat_structure = True
            else:
                # Episode structure (e.g., 'episode_0', 'episode_1', ...)
                self.episode_keys = [k for k in self.keys if k.startswith('episode') or k.startswith('demo')]
                if max_episodes:
                    self.episode_keys = self.episode_keys[:max_episodes]
                self.num_episodes = len(self.episode_keys)

                # Get episode lengths
                self.episode_lengths = []
                for ep_key in self.episode_keys:
                    ep_len = len(f[ep_key]['observations'])
                    self.episode_lengths.append(ep_len)
                self.use_flat_structure = False

        # Build index: list of (episode_idx, start_frame) tuples
        self.index = []
        for ep_idx, ep_len in enumerate(self.episode_lengths):
            # Each episode can produce multiple sequences
            num_sequences = max(1, ep_len - sequence_length)
            for start_idx in range(num_sequences):
                self.index.append((ep_idx, start_idx))

        print(f"Loaded {self.num_episodes} episodes with {len(self.index)} total sequences")

    def __len__(self):
        return len(self.index)

    def _file(self):
        """Return this process's HDF5 handle, opening it once on first use.

        Opening per worker (instead of per __getitem__) avoids a file open/close
        syscall on every sample, and keeping it lazy keeps the dataset picklable
        so it can be sent to DataLoader workers.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __getstate__(self):
        # Don't try to pickle an open h5py handle when shipping to workers.
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def __getitem__(self, idx):
        """
        Returns:
            observations: (N+1, C, H, W) - sequence of N+1 frames
            actions: (N, action_dim) - N actions
        """
        ep_idx, start_idx = self.index[idx]

        f = self._file()
        if self.use_flat_structure:
            # Two Rooms or flat structure
            if 'pixels' in f:
                # Two Rooms dataset - use episode offsets
                ep_offset = self.episode_offsets[ep_idx]
                global_start = ep_offset + start_idx
                global_end = global_start + self.sequence_length + 1

                observations = f['pixels'][global_start:global_end]
                actions = f['action'][global_start:global_start + self.sequence_length]
            else:
                # Standard flat structure
                obs_data = f['observations']
                action_data = f['actions']
                end_idx = start_idx + self.sequence_length + 1
                observations = obs_data[start_idx:end_idx]
                actions = action_data[start_idx:start_idx + self.sequence_length]
        else:
            # Episode structure
            ep_key = self.episode_keys[ep_idx]
            obs_data = f[ep_key]['observations']
            action_data = f[ep_key]['actions']

            # Load sequence of observations (N+1 frames)
            end_idx = start_idx + self.sequence_length + 1
            observations = obs_data[start_idx:end_idx]
            actions = action_data[start_idx:start_idx + self.sequence_length]

        # Convert to torch tensors
        observations = torch.from_numpy(np.array(observations)).float()
        actions = torch.from_numpy(np.array(actions)).float()

        # Handle different image formats
        if observations.dim() == 4:  # (N+1, H, W, C) -> (N+1, C, H, W)
            if observations.shape[-1] == 3:  # Channel last
                observations = observations.permute(0, 3, 1, 2)

        # Normalize images to [0, 1] if needed
        if observations.max() > 1.0:
            observations = observations / 255.0

        # Apply transforms if provided
        if self.transform:
            observations = torch.stack([self.transform(obs) for obs in observations])

        return observations, actions


def create_dataloader(
    h5_path: str,
    batch_size: int = 32,
    sequence_length: int = 5,
    num_workers: int = 8,
    shuffle: bool = True,
    max_episodes: int = None,
    prefetch_factor: int = 4,
    drop_last: bool = True,
):
    """
    Create a dataloader for the Two Rooms dataset.

    Args:
        h5_path: Path to the HDF5 file
        batch_size: Batch size
        sequence_length: Number of frames per sequence
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        max_episodes: Maximum episodes to load (for debugging)

    Returns:
        DataLoader
    """
    dataset = TwoRoomsDataset(
        h5_path=h5_path,
        sequence_length=sequence_length,
        max_episodes=max_episodes,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return dataloader
