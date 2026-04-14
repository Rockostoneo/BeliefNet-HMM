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

	token_nll = -model.score(validation_set) / validation_set.size
	print(f"Validation loss (token NLL) : {token_nll:.3f}")
	print(f"Validation set shape: {validation_set.shape}")

if __name__ == "__main__":
	data_file_path = Path(__file__).resolve().parent / "../prepare_data/data_d64_m32_lamda0.9_example.h5"
	model_param_path = Path(__file__).resolve().parent / "bw_d64_m32_lamda0.9_iters20_fit1_example.h5"
	with h5.File(data_file_path, "r") as f:
		params = eval(f["params"][()])
		validation_set = f["z_validation"][:] # z_train, z_validation, or z_test
	evaluate(model_param_path, validation_set)
