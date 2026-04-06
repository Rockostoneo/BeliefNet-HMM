import argparse
import ast
from pathlib import Path

import h5py as h5
import numpy as np
from hmmlearn import hmm


def load_model(model_file: Path) -> hmm.CategoricalHMM:
	with h5.File(model_file, "r") as f:
		A = f["A"][:]
		C = f["C"][:]
		pi = f["pi"][:]

	n_components = A.shape[0]
	model = hmm.CategoricalHMM(
		n_components=n_components,
		init_params="",
		params="",
	)
	model.startprob_ = pi
	model.transmat_ = A
	model.emissionprob_ = C
	model.n_features = C.shape[1]
	return model

def to_hmmlearn_input(split: np.ndarray) -> tuple[np.ndarray, list[int] | None]:
	if split.ndim == 1:
		return split.reshape(-1, 1), [int(split.shape[0])]

	if split.ndim == 2:
		if split.shape[1] == 1:
			return split.astype(np.int64), [int(split.shape[0])]
		# Interpret each row as an independent sequence.
		n_seq, seq_len = split.shape
		X = split.reshape(-1, 1)
		lengths = [int(seq_len)] * int(n_seq)
		return X, lengths

	raise ValueError(
		f"Expected validation split with 1 or 2 dimensions, got shape {split.shape}"
	)

def evaluate(model_file: Path, validation_set) -> None:
	model = load_model(model_file)
	X, lengths = to_hmmlearn_input(validation_set)

	log_likelihood = model.score(X, lengths=lengths)
	token_nll = -log_likelihood / X.shape[0]

	print(f"Num sequences: {validation_set.shape[0] if validation_set.ndim > 1 else 1}")
	print(f"Num tokens: {X.shape[0]}")
	print(f"Validation loss (token NLL) : {token_nll:.3f}")
	print(f"Validation set shape: {validation_set.shape}")

if __name__ == "__main__":
	data_file_path = Path(__file__).resolve().parent / "../prepare_data/data_d64_m32_lamda0.9_example.h5"
	model_param_path = Path(__file__).resolve().parent / "bw_d64_m32lamda0.9_iters5_fit0_16.h5"
	with h5.File(data_file_path, "r") as f:
		params = eval(f["params"][()])
		validation_set = np.asarray(f["z_validation"][:], dtype=np.int64) # z_train, z_validation, or z_test
	evaluate(model_param_path, validation_set)
