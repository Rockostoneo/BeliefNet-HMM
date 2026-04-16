from pathlib import Path
import h5py as h5
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

def evaluate(model_file: Path, validation_set) -> None:
	model = load_model(model_file)

	X = validation_set.flatten()
	lengths = [validation_set.shape[1]] * validation_set.shape[0]

	loss = -model.score(X.reshape(-1,1), lengths) / X.size
	print(f"Validation loss (token NLL) : {loss:.3f}")

if __name__ == "__main__":
	data_file_path = Path(__file__).resolve().parent / "../prepare_data/data_d64_m32_lamda0.9_example.h5"
	model_param_path = Path(__file__).resolve().parent / "bw_d64_m32_lamda0.9_iters20_fit1_example.h5" # edit this to the model you want to evaluate
	split = "validation" # "train", "validation", or "test"
	print(f"Split: {split}")
	
	with h5.File(data_file_path, "r") as f:
		params = eval(f["params"][()])
		validation_set = f[f"z_{split}"][:]

	evaluate(model_param_path, validation_set)
