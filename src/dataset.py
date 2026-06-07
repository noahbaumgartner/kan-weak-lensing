from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _noise_sigma(ng: float, pixel_size_arcmin: float) -> float:
    return 0.4 / (2.0 * ng * pixel_size_arcmin**2) ** 0.5


class _MapDataset(Dataset):
    def __init__(
        self,
        kappa_flat: np.ndarray,
        mask: np.ndarray,
        labels: Optional[np.ndarray],
        input_stats: Optional[tuple[float, float]] = None,
        add_noise: bool = False,
        ng: float = 30.0,
        pixel_size_arcmin: float = 2.0,
        noise_seed: Optional[int] = None,
    ):
        self.kappa_flat = kappa_flat
        self.mask = mask.astype(bool, copy=False)
        self.labels = labels
        self.input_stats = input_stats
        self.add_noise = add_noise
        self.noise_seed = noise_seed
        self._h, self._w = mask.shape
        self._mask_f32 = self.mask.astype(np.float32)
        self._noise_sigma = _noise_sigma(ng, pixel_size_arcmin) if add_noise else 0.0

    def __len__(self) -> int:
        return self.kappa_flat.shape[0]

    def _rng(self, idx: int):
        if self.noise_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence([self.noise_seed, idx]))

    def __getitem__(self, idx: int):
        dense = np.zeros((self._h, self._w), dtype=np.float32)
        dense[self.mask] = self.kappa_flat[idx].astype(np.float32, copy=False)

        if self.add_noise:
            noise = self._rng(idx).standard_normal(dense.shape).astype(np.float32)
            noise *= self._noise_sigma
            noise *= self._mask_f32
            dense += noise

        if self.input_stats is not None:
            m, s = self.input_stats
            dense = (dense - m) / s

        x = torch.from_numpy(dense).unsqueeze(0)
        if self.labels is None:
            return x
        return x, torch.from_numpy(self.labels[idx])


class WeakLensingDataset:
    img_height = 1424
    img_width = 176
    in_chans = 1
    output_dim = 2

    def __init__(
        self,
        data_dir: str,
        n_targets: int = 2,
        val_fraction: float = 0.2,
        split_seed: int = 5566,
        noise_seed: Optional[int] = None,
        ng: float = 30.0,
        pixel_size_arcmin: float = 2.0,
        standardize: bool = True,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.n_targets = n_targets
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.noise_seed = noise_seed
        self.ng = ng
        self.pixel_size_arcmin = pixel_size_arcmin
        self.standardize = standardize

        self.kappa_file = "WIDE12H_bin2_2arcmin_kappa.npy"
        self.label_file = "label.npy"
        self.test_kappa_file = "WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
        self.mask_file = "WIDE12H_bin2_2arcmin_mask.npy"
        self.Ncosmo, self.Nsys = 101, 256

    def _load_compact(self, fname: str) -> np.ndarray:
        return np.load(os.path.join(self.data_dir, fname), mmap_mode="r")

    def _estimate_input_stats(
        self, kappa_flat: np.ndarray, mask: np.ndarray, n_sample: int = 256
    ) -> tuple[float, float]:
        n = kappa_flat.shape[0]
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(n_sample, n), replace=False)
        sigma = _noise_sigma(self.ng, self.pixel_size_arcmin)
        mask_f32 = mask.astype(np.float32)

        sum_, sum_sq, count = 0.0, 0.0, 0
        for i in idx:
            dense = np.zeros(mask.shape, dtype=np.float32)
            dense[mask] = kappa_flat[i].astype(np.float32, copy=False)
            noise = rng.standard_normal(mask.shape).astype(np.float32)
            noise *= sigma
            noise *= mask_f32
            dense += noise
            sum_ += float(dense.sum())
            sum_sq += float((dense * dense).sum())
            count += dense.size
        mean = sum_ / count
        var = sum_sq / count - mean * mean
        return mean, float(np.sqrt(max(var, 1e-12)))

    def create(self) -> dict:
        mask = np.load(os.path.join(self.data_dir, self.mask_file)).astype(bool)
        n_unmasked = int(mask.sum())

        kappa = np.ascontiguousarray(self._load_compact(self.kappa_file), dtype=np.float16)
        kappa = kappa.reshape(self.Ncosmo, self.Nsys, n_unmasked)
        labels = np.load(os.path.join(self.data_dir, self.label_file)).astype(np.float32)

        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(self.Nsys)
        n_val = int(round(self.Nsys * self.val_fraction))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])

        train_kappa = kappa[:, train_idx].reshape(-1, n_unmasked)
        val_kappa = kappa[:, val_idx].reshape(-1, n_unmasked)
        train_y = labels[:, train_idx].reshape(-1, labels.shape[-1])[:, : self.n_targets]
        val_y = labels[:, val_idx].reshape(-1, labels.shape[-1])[:, : self.n_targets]

        input_stats = self._estimate_input_stats(train_kappa, mask) if self.standardize else None

        if self.standardize:
            label_mean = train_y.mean(axis=0).astype(np.float32)
            label_std = train_y.std(axis=0).astype(np.float32)
            label_std = np.where(label_std > 1e-8, label_std, 1.0).astype(np.float32)
            train_y = (train_y - label_mean) / label_std
            val_y = (val_y - label_mean) / label_std
            label_stats = (label_mean, label_std)
        else:
            label_stats = None

        train_ds = _MapDataset(
            train_kappa, mask, train_y,
            input_stats=input_stats, add_noise=True,
            ng=self.ng, pixel_size_arcmin=self.pixel_size_arcmin,
            noise_seed=None,
        )
        val_ds = _MapDataset(
            val_kappa, mask, val_y,
            input_stats=input_stats, add_noise=True,
            ng=self.ng, pixel_size_arcmin=self.pixel_size_arcmin,
            noise_seed=31415,
        )

        test_compact = np.ascontiguousarray(self._load_compact(self.test_kappa_file), dtype=np.float16)
        if test_compact.ndim == 2:
            test_compact = test_compact.reshape(-1, n_unmasked)
        test_ds = _MapDataset(
            test_compact, mask, None,
            input_stats=input_stats, add_noise=False,
        )

        return {
            "train_input": train_ds,
            "train_label": train_y,
            "val_input": val_ds,
            "val_label": val_y,
            "test_input": test_ds,
            "test_label": None,
            "mask": mask,
            "input_stats": input_stats,
            "label_stats": label_stats,
        }
