"""
02_specimen_variant_gan.partitioning.adversarial_and_kfold
===========================================================
Các giao thức phân hoạch ngẫu nhiên phân tầng, học đối kháng và Group K-Fold:
- PP6: Stratified Random Split (Baseline)
- PP7: Adversarial Validation Split (Discriminator MLP)
- PP8: Stratified Group K-Fold Split
"""

import random
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold as SKF_Splitter
from tqdm import tqdm

from .base import (
	compute_split_counts,
	_shuffle_df,
	_split_by_groups,
)


def stratified_random_split(
	df: pd.DataFrame,
	embeddings: np.ndarray = None,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP6: Baseline — chia ngẫu nhiên giữ tỉ lệ class.
	Được cài đặt an toàn để tránh lỗi khi một số class có số lượng mẫu cực ít.
	"""
	rng = random.Random(seed)
	train_idx, val_idx, test_idx = [], [], []

	for _, group in df.groupby("label"):
		indices = sorted(group.index.tolist())
		rng.shuffle(indices)

		n_total = len(indices)
		train_count, val_count, test_count = compute_split_counts(n_total, train_ratio, val_ratio)

		train_idx.extend(indices[:train_count])
		val_idx.extend(indices[train_count : train_count + val_count])
		test_idx.extend(indices[train_count + val_count :])

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


def adversarial_validation_split(
	df: pd.DataFrame,
	embeddings: np.ndarray,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP7: Adversarial Validation — train discriminator trên subfolder embeddings để phân biệt 2 pool subfolders.
	Những subfolders mà discriminator tự tin nhất là "khác biệt" -> đưa vào test.
	Đảm bảo KHÔNG rò rỉ dữ liệu mức subfolder.
	"""
	if len(df) != embeddings.shape[0]:
		raise ValueError("Embeddings length does not match dataframe length")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rng = np.random.RandomState(seed)

	train_idx, val_idx, test_idx = [], [], []

	for label, group in tqdm(df.groupby("label"), desc="PP7-Adversarial"):
		subfolder_groups = group.groupby("subfolder")
		subfolder_names = list(subfolder_groups.groups.keys())
		n_subfolders = len(subfolder_names)

		if n_subfolders == 0:
			continue

		if n_subfolders < 3:
			groups = [subfolder_groups.get_group(sf_name).index.tolist() for sf_name in subfolder_names]
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=True)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
			continue

		subfolder_embs = []
		for sf_name in subfolder_names:
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			subfolder_embs.append(embeddings[sf_indices].mean(axis=0))
		subfolder_embs = np.array(subfolder_embs)

		perm = rng.permutation(n_subfolders)
		half = n_subfolders // 2
		pool_b_pos = perm[half:]

		X = subfolder_embs.copy()
		y = np.zeros(n_subfolders, dtype=np.float32)
		y[pool_b_pos] = 1.0

		emb_dim = X.shape[1]
		discriminator = nn.Sequential(
			nn.Linear(emb_dim, 128),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(128, 1),
			nn.Sigmoid(),
		).to(device)

		X_tensor = torch.from_numpy(X).float().to(device)
		y_tensor = torch.from_numpy(y).float().to(device)

		optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
		criterion = nn.BCELoss()

		discriminator.train()
		for _ in range(30):
			optimizer.zero_grad()
			pred = discriminator(X_tensor).squeeze()
			loss = criterion(pred, y_tensor)
			loss.backward()
			optimizer.step()

		discriminator.eval()
		with torch.no_grad():
			scores = discriminator(X_tensor).squeeze().cpu().numpy()

		difficulty_scores = np.abs(scores - 0.5)
		sorted_positions = np.argsort(-difficulty_scores)

		train_sf_count, val_sf_count, test_sf_count = compute_split_counts(n_subfolders, train_ratio, val_ratio)

		for i, pos in enumerate(sorted_positions):
			sf_name = subfolder_names[pos]
			sf_indices = subfolder_groups.get_group(sf_name).index.tolist()
			if i < test_sf_count:
				test_idx.extend(sf_indices)
			elif i < test_sf_count + val_sf_count:
				val_idx.extend(sf_indices)
			else:
				train_idx.extend(sf_indices)

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)
	return df_train, df_val, df_test


def stratified_group_kfold_split(
	df: pd.DataFrame,
	embeddings: np.ndarray = None,
	train_ratio: float = 0.60,
	val_ratio: float = 0.20,
	seed: int = 42,
	eps: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	PP8: StratifiedGroupKFold — Cân bằng cách ly nhóm subfolder và phân tầng class.
	Đối với các class có ít hơn 3 subfolders, phân rã ảnh ra và chia bằng _split_by_groups.
	Đối với các class có từ 3 subfolders trở lên, chia bằng StratifiedGroupKFold ở mức group.
	"""
	train_idx, val_idx, test_idx = [], [], []

	large_class_groups = []
	for label, group in df.groupby("label"):
		subfolder_names = group["subfolder"].unique()
		n_subfolders = len(subfolder_names)

		if n_subfolders < 3:
			groups = [group[group["subfolder"] == sf].index.tolist() for sf in subfolder_names]
			tr, va, te = _split_by_groups(groups, train_ratio, val_ratio, seed, sorted_descending=True)
			train_idx.extend(tr)
			val_idx.extend(va)
			test_idx.extend(te)
		else:
			large_class_groups.append(group)

	if large_class_groups:
		df_large = pd.concat(large_class_groups)
		test_ratio = 1.0 - train_ratio - val_ratio
		n_splits_test = max(3, int(round(1.0 / test_ratio)))

		groups = (df_large["label"] + "___" + df_large["subfolder"]).values
		labels = df_large["label"].values

		sgkf_test = SKF_Splitter(n_splits=n_splits_test, shuffle=False, random_state=None)

		trainval_indices_rel = None
		test_indices_rel = None
		for tv_idx, te_idx in sgkf_test.split(df_large.index, labels, groups):
			trainval_indices_rel = tv_idx
			test_indices_rel = te_idx
			break

		trainval_absolute_indices = df_large.index[trainval_indices_rel].tolist()
		test_absolute_indices = df_large.index[test_indices_rel].tolist()

		df_trainval = df.loc[trainval_absolute_indices]
		test_idx.extend(test_absolute_indices)

		val_fraction = val_ratio / (train_ratio + val_ratio)
		n_splits_val = max(3, int(round(1.0 / val_fraction)))

		groups_tv = (df_trainval["label"] + "___" + df_trainval["subfolder"]).values
		labels_tv = df_trainval["label"].values

		sgkf_val = SKF_Splitter(n_splits=n_splits_val, shuffle=False, random_state=None)

		train_indices_rel = None
		val_indices_rel = None
		for tr_idx, va_idx in sgkf_val.split(df_trainval.index, labels_tv, groups_tv):
			train_indices_rel = tr_idx
			val_indices_rel = va_idx
			break

		train_absolute_indices = df_trainval.index[train_indices_rel].tolist()
		val_absolute_indices = df_trainval.index[val_indices_rel].tolist()

		train_idx.extend(train_absolute_indices)
		val_idx.extend(val_absolute_indices)

	df_train = _shuffle_df(df.loc[train_idx], seed)
	df_val = _shuffle_df(df.loc[val_idx], seed)
	df_test = _shuffle_df(df.loc[test_idx], seed)

	return df_train, df_val, df_test
