# BeliefNet-HMM Instructions
This repository contains the code used to train and test the models presented in the [Belief Net paper](https://arxiv.org/abs/2511.10571).

The steps will be:
1. Set up the Python environment and install necessary packages
2. Generate training data from a synthetic Hidden Markov Model (HMM) or existing text file
3. Train, test, and extract learned parameters for each model:
     <ol type="i">
         <li>Belief Net</li>
         <li>NanoGPT</li>
         <li>Baum-Welch</li>
         <li>Spectral</li>
     </ol>

## Setting up repo and environment  
1. Install and initialize Git LFS (required for downloading large model artifacts)
```
git lfs install
```
2. For a fresh clone, clone the repository and download LFS-managed files
```
git clone https://github.com/Rockostoneo/BeliefNet-HMM.git
cd BeliefNet-HMM
git lfs pull
```
3. Create virtual environment with Python 3.13.2
```
python -m venv .venv
```
4. Activate venv (for windows, mac and linux are slightly different)

PowerShell:
```
.\.venv\Scripts\Activate.ps1
```

Command Prompt (CMD):
```
.venv\Scripts\activate.bat
```

Git Bash:
```
source .venv/Scripts/activate
```
5. Install libraries
```
pip install hmmlearn tiktoken notebook
pip install git+https://github.com/hanson-hschang/Signal-System.git
```
6. *Optional* An HDF5 file viewer is recommended for conveniently viewing raw HDF5 files (e.g. [H5Web](https://marketplace.visualstudio.com/items?itemName=h5web.vscode-h5web) or [myHDF5](https://myhdf5.hdfgroup.org/)).

## Generating Testing Data
1. From the home directory change to the `/prepare_data` directory 
```
cd prepare_data
```
2. Edit the parameters in either the `generate_data.py` or `generate_data_from_corpus.py` file. Then run
```
python generate_data.py
```
or 
```
python generate_data_from_corpus.py
```
3. *Optional* Check generated data files using an HDF5 file viewer.

## Training Belief Net
1. From the home directory change to the `/BeliefNet` directory 
```
cd BeliefNet
```
2. Edit or create a new config json file. (e.g. `config_cpu_d64_m32.json`)

3. Train a model, you can override config setting with command-line flags `--name value`.
```
python training.py config_cpu_d64_m32.json --mode train
```

4. Plot the training and evaluation loss trajectory. The plot is saved in the model directory.
```
python training.py config_cpu_d64_m32.json --mode visual
```

5. Extract the learned parameters (transition, emission, initial state). The parameters are saved in an hdf5 file in the model directory.
```
python training.py config_cpu_d64_m32.json --mode get_matricies 
```

6. Evaluate the trained model on the entire dataset or a new dataset (losses during training are only calculated over a single batch). First edit the `if __name__ == '__main__':` section in the `evaluate.py` file, then run.
```
python evaluate.py
```

## Training NanoGPT
1. From the home directory change to the `/NanoGPT` directory.
```
cd NanoGPT
```
2. Edit or create a new config file in the `/config` directory. (e.g. `train_m32.py`)

3. Train a model, you can override config setting with command-line flags `--max_iters=2000`
```
python train.py config/train_m32.py
```

4. Evaluate the trained model on the entire dataset or a new dataset. Edit the parameters in `validation.py`, then
```
python validation.py
```

## Training Baum-Welch
1. From the home directory change to the `/Baum-Welch` directory.
```
cd Baum-Welch
```

2. Train a model. Edit the parameters in the `if __name__ == '__main__':` section of the `train_baum_welch.py` file. Trained model parameters are automatically saved as HDF5 files. Then run
```
python train_baum_welch.py
```

3. Evaluate the trained model on the entire dataset or a new dataset. Edit the parameters in the `if __name__ == '__main__':` section of `eval_baum_welch.py`, then run:
```
python eval_baum_welch.py
```

## Training Spectral Algorithm
1. Follow the instructions in the `train_spectral.ipynb` Jupyter notebook.


# Citation
We ask that any publications which use `BeliefNet-HMM` cite as follows:
```
@inproceedings{
  chen2026belief,
  title={Belief Net: A Filter-Based Framework for Learning Hidden Markov Models from Observations},
  author={Reginald Zhiyan Chen and Heng-Sheng Chang and Prashant G Mehta},
  booktitle={8th Annual Learning for Dynamics and Control Conference},
  year={2026},
}
```