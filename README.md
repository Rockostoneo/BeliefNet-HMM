<div align=center>

# Belief Net - HMM

</div>

This repository contains the code used to train and test the models presented in the paper [Belief Net](https://arxiv.org/abs/2511.10571).
The steps will be:

1.  **Environment Setup**: Install Python dependencies
2.  **Data Generation**: Produce synthetic trajectories from a Hidden Markov Model (HMM) or load existing text files.
3.  **Model Evaluation**: Train, test, and extract learned parameters for the following architectures:
      * **[Belief Net](#31-training-belief-net)**: Developed via [`Signal-System`]((https://github.com/hanson-hschang/Signal-System)).
      * **[nanoGPT](#32-training-nanogpt)**: Adapted from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
      * **[Baum-Welch](#33-training-baum-welch)**: Implemented using [`hmmlearn`](https://github.com/hmmlearn/hmmlearn).
      * **[Spectral](#34-training-spectral-algorithm)**: Based on the [Hsu et al. (2009)](https://arxiv.org/abs/0811.4413) algorithm and adapted from [hanzhaoml/Spectral-learning](https://github.com/hanzhaoml/Spectral-learning)

## 1. Setting Up Repo and Environment  
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
    pip install git+https://github.com/hanson-hschang/Signal-System.git@v0.0.2
    ```

6. *Optional:* An HDF5 file viewer is recommended for conveniently viewing raw HDF5 files (e.g. [H5Web](https://marketplace.visualstudio.com/items?itemName=h5web.vscode-h5web) or [myHDF5](https://myhdf5.hdfgroup.org/)).

## 2. Generating Testing Data
1. From the home directory change to the `prepare_data/` directory 
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

## 3. Training Models

### 3.1 Training Belief Net
1. From the home directory change to the `BeliefNet/` directory 
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

### 3.2 Training NanoGPT
1. From the home directory change to the `NanoGPT/` directory.
    ```
    cd NanoGPT
    ```
2. Edit or create a new config file in the `config/` directory. (e.g. `train_m32.py`)

3. Train a model, you can override config setting with command-line flags `--max_iters=2000`
    ```
    python train.py config/train_m32.py
    ```

4. Evaluate the trained model on the entire dataset or a new dataset. Edit the parameters in `validation.py`, then
    ```
    python validation.py
    ```

### 3.3 Training Baum-Welch
1. From the home directory change to the `Baum-Welch/` directory.
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

### 3.4 Training Spectral Algorithm
1. Follow the instructions in the `train_spectral.ipynb` Jupyter notebook.


## Citation
We ask that any publications which use `Belief Net - HMM` cite as follows:
```
@inproceedings{
  chen2026differentiable,
  title={Differentiable Filtering for Learning Hidden Markov Models},
  author={Reginald Zhiyan Chen and Heng-Sheng Chang and Prashant G Mehta},
  booktitle={8th Annual Learning for Dynamics and Control Conference},
  year={2026},
}
```
