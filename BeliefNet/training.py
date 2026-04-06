from typing import Any, Callable, Dict, cast

import torch._dynamo
torch._dynamo.config.suppress_errors = True

#import os
import time
#import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import h5py 
from torch.utils.data import DataLoader, TensorDataset
from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter
from ss.estimation.filtering.hmm.learning.module import config as Config
from ss.estimation.filtering.hmm.learning.process import (
    LearningHmmFilterProcess,
)
from ss.utility.device import Device
from ss.utility.device.manager import DeviceManager
from ss.utility.learning.process.checkpoint import CheckpointInfo
from ss.utility.learning.process.config import TrainingConfig
from ss.utility.logging import Logging
from ss.utility.path import PathManager

logger = Logging.get_logger(__name__)
from utils import (
    CharTokenizer,
    UserConfig,
    get_file_name,
    get_model_path,
    get_time_log_file_path,
    load_config,
)

def reconstruct_loaders_from_hdf5(data_file,
                                  batch_size=None):
    """
    Reconstruct train_loader and eval_loader from saved HDF5 file.
    
    Args:
        data_file: Path to the HDF5 file containing the datasets
        batch_size: Batch size for loaders (if None, uses original batch size)
    
    Returns:
        train_loader, eval_loader: Reconstructed PyTorch DataLoaders
    """
    
    f = h5py.File(data_file, 'r')
    train_inputs_tensor = torch.tensor(f['z_train'][:,:-1])
    train_targets_tensor = torch.tensor(f['z_train'][:,1:])
    eval_inputs_tensor = torch.tensor(f['z_validation'][:,:-1])
    eval_targets_tensor = torch.tensor(f['z_validation'][:,1:])
    
    f.close()
    # Create TensorDatasets
    train_dataset = TensorDataset(train_inputs_tensor, train_targets_tensor)
    eval_dataset = TensorDataset(eval_inputs_tensor, eval_targets_tensor)
    
    # Determine batch size
    if batch_size is None:
        # Use the batch size from the original data
        batch_size = 1
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Reconstructed train_loader: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Reconstructed eval_loader: {len(eval_dataset)} samples, {len(eval_loader)} batches")
    
    return train_loader, eval_loader

def train(
    model_folder_path: Path,
    model_file_name: str,
    user_config: UserConfig,
    data_file: str,
):

    # Prepare data
    train_loader, eval_loader = reconstruct_loaders_from_hdf5(
        data_file=data_file,
        batch_size=user_config.batchsize)

    ## Prepare model
    config = Config.LearningHmmFilterConfig.basic_config(
        state_dim=user_config.d,
        discrete_observation_dim=user_config.m,
        dropout_rate=user_config.dropout,
    )

    learning_filter = LearningHmmFilter(config)

    # Prepare loss function
    loss_function = torch.nn.functional.cross_entropy

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        learning_filter.parameters(),
        lr=user_config.lr
    )

    # Prepare learning process
    class LearningProcess(LearningHmmFilterProcess):
        def __init__(
            self,
            module: LearningHmmFilter,
            loss_function: Callable[..., torch.Tensor],
            optimizer: torch.optim.Optimizer,
            user_config: UserConfig,
        ) -> None:
            super().__init__(module, loss_function, optimizer)
            self._user_config = user_config

        def _save_model_info(self) -> Dict[str, Any]:
            model_info = dict(
                loss_function=self._loss_function,
                optimizer=self._optimizer,
                user_config=self._user_config,
                # save extra arguments if needed
            )
            return model_info

    learning_process = LearningProcess(
        module=learning_filter,
        loss_function=loss_function,
        optimizer=optimizer,
        user_config=user_config,
        # number_of_epochs=user_config.epoch,
        # model_filename=model_filepath,
        # evaluate_model_iteration_skip=user_config.eval_interval,
        # save_model_epoch_skip=max(1, user_config.epoch//5),
    )
    training_config = TrainingConfig()
    training_config.validation.per_iteration_period = user_config.eval_interval
    training_config.termination.max_epoch = user_config.epoch
    training_config.termination.max_iteration = user_config.max_iters
    training_config.checkpoint.folderpath = model_folder_path
    training_config.checkpoint.filename = str(
        Path(model_file_name).with_suffix("")
    )
    training_config.checkpoint.per_epoch_period = max(
        1, user_config.epoch // 5
    )
    training_config.checkpoint.appendix.option = (
        training_config.checkpoint.appendix.Option.COUNTER
    )

    # device_manager = DeviceManager()
    start_time = time.perf_counter()
    # with device_manager.monitor_performance(
    #     sampling_rate=10.0, # frequency to detect machine status
    #     # result_directory=result_directory,
    # ):
    learning_process.training(train_loader, eval_loader, training_config)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(
        f"Training time: {execution_time/60:.4f} minutes for {user_config.max_iters} iterations"
    )  # {user_config.epoch} epochs")
    # execution_time_per_epoch = execution_time / user_config.epoch
    execution_time_per_iter = execution_time / user_config.max_iters
    token_per_sec = user_config.token_per_iter / execution_time_per_iter
    return execution_time_per_iter, token_per_sec

def get_matricies(
    model_folder_path,
    model_file_name,
    user_config: UserConfig,
    matrix_filename
):  
    model_filepath = model_folder_path / model_file_name
    learning_filter, model_info = LearningHmmFilter.load(
        model_filepath,
        safe_callables={
            torch.nn.functional.cross_entropy,
            torch.optim.AdamW,
            UserConfig,
            CharTokenizer,
            tiktoken.core.Encoding,
        },
        strict=False,
    )
    with learning_filter.evaluation_mode():
        A = learning_filter.transition.matrix.cpu().detach().numpy()
        C = learning_filter.emission.matrix.cpu().detach().numpy()
        pi = learning_filter.transition.initial_state.cpu().detach().numpy()
    
    
    f = h5py.File(matrix_filename, 'a')
    f.create_dataset('A', data=A)
    f.create_dataset('C', data=C)
    f.create_dataset('pi', data=pi)
    f.close()
    print(f"Matricies saved to {matrix_filename}")
    
def visualization(
    model_folder_path, model_file_name, user_config: UserConfig, saveflag=False
):
    model_filepath = model_folder_path / model_file_name
    # Plot the training and validation loss
    checkpoint_info = CheckpointInfo.load(model_filepath.with_suffix(".hdf5"))
    training_loss_trajectory = checkpoint_info["__training_loss_history__"]
    validation_loss_trajectory = checkpoint_info["__validation_loss_history__"]

    iterations = training_loss_trajectory["iteration"]
    training_loss = training_loss_trajectory["loss"]
    epochs = validation_loss_trajectory["iteration"]
    validation_loss = validation_loss_trajectory["loss_mean"]
    # validation_loss = validation_loss_trajectory["loss_std"]

    plt.plot(
        iterations,
        training_loss,
        lw=3,
        ls="-",
        color="C0",
        label="training loss",
    )
    plt.plot(
        epochs,
        validation_loss,
        lw=2,
        ls="--",
        color="C1",
        label="validation loss",
    )
    plt.text(
        epochs[0],
        validation_loss[0],
        f"{validation_loss[0]:.4f}",
        fontsize=12,
        color="C3",
        ha="left",
        va="center",
    )
    plt.text(
        epochs[-1],
        validation_loss[-1] + 0.2,
        f"{validation_loss[-1]:.4f}",
        fontsize=12,
        color="C3",
        ha="right",
    )

    resolution = 0.2
    y_min, y_max = plt.ylim()
    y_min = np.floor(y_min / resolution) * resolution
    y_max = np.ceil(y_max / resolution) * resolution
    y_range = y_max - y_min
    plt.ylim(y_min - 0.01 * y_range, y_max + 0.01 * y_range)
    # print(y_min, y_max, y_range)

    xtick_labels = [str(i) for i in epochs]
    skip = max(1, len(epochs) // 5)
    plt.xticks(epochs[::skip], xtick_labels[::skip])
    plt.xlabel("Iterations - " + str(user_config.epoch) + " epochs")
    plt.legend()
    file_name = get_file_name(
        user_config, target_variable=None, file_name="loss"
    )
    plt.title(file_name)
    if saveflag:
        plt.savefig(model_folder_path / (file_name + ".png"), dpi=300)
    # plt.show()


import json
import click

# from utils import model_parent_folder


@click.command()
@click.option(
    "--mode",
    type=click.Choice(
        ["train", "visual", "get_matricies"],
        case_sensitive=False,
    ),
    # default='visual',
)
@click.option(
    "--device",
    type=click.Choice(
        ["cpu", "gpu", "mps", "cuda"],
        case_sensitive=False,
    ),
    # default='cpu',
)
@click.option(
    "--block",
    type=click.Choice(
        ["full_matrix", "spatial_invariant", "iid"],
        case_sensitive=False,
    ),
    # default='full_matrix',
)
@click.option(
    "--initblock",
    type=click.Choice(
        [
            "normal_distribution",
            "uniform_distribution",
            "constant",
            "identity",
        ],
        case_sensitive=False,
    ),
    # default='normal_distribution',
)
@click.option(
    "--fulltoken",  ## using full gpt-2 vocabulary or character-level tokens
    type=bool,
    # is_flag=True,
    # default=True,
)
@click.option(
    "--layer",  ## number of layers
    type=int,
    # default=1
)
@click.option(
    "--d",  ## Embedding dimension
    type=int,
    # default=768
)
@click.option(
    "--lamda",  ## lamda for loss function
    type=float,
    # default=1.0,
) 
@click.option(
    "--max_iters",  ## max number of iterations
    type=int,
    # default=2000
)
@click.option(
    "--epoch",  ## number of epochs
    type=int,
    # default=1
)
@click.option(
    "--iterflag",  ## whether name the model by max_iters or by epoch (default)
    # is_flag=True,
    type=bool,
)
@click.option(
    "--context",  ## context length
    type=int,
    # default=256,
)
@click.option(
    "--stride",  ## stride
    type=int,
    # default=None, ## default is context length
)
@click.option(
    "--batchsize",  ## batch size
    type=int,
    # default=4,
)
@click.option(
    "--splitratio",  ## train-validation split ratio
    type=float,
    # default=0.9,
)
@click.option(
    "--lr",  ## learning rate
    type=float,
    # default=0.1,
)
@click.option(
    "--dropout",  ## dropout rate
    type=float,
    # default=0.2,
)
@click.option(
    "--eval_interval",  ## evaluation interval
    type=int,
    # default=200,
)
@click.option(
    "--temp",  ## temperature scaling
    type=float,
    # default=0.8,
)
@click.option(
    "--topk",  ## topk sampling
    type=int,
    # default=200,
)
@click.option(
    "--start_text",  ## start text for inference
    type=str,
    # default="\n",
)
@click.option(
    "--max_new_tokens",  ## max new tokens for inference
    type=int,
    # default=200,
)
@click.option(
    "--saveflag",  ## save figure / inference results or not
    type=bool,
    # is_flag=True,
)
@click.option(
    "--cross",  ## evaluate over hyperparameters
    # is_flag=True,
    type=click.Choice(
        ["d", "context", "batchsize", "lr", "dropout"],
        case_sensitive=False,
    ),
    # default=None,
)
@click.option(
    "--lb",  ## lower bound for hyperparameter evaluation, base 2
    type=int,
    # default=None,
)
@click.option(
    "--ub",  ## lower bound for hyperparameter evaluation, base 2
    type=int,
    # default=None,
)
@click.option(
    "--targetlist",  ## target list for hyperparameter evaluation, seperate by ',' and no space
    type=str,
    # default=None,
)
@click.option(
    "--checkpoint",  ## select model checkpoint file
    type=int,
    # default=None,
)
@click.option(
    "--verbose",
    type=bool,
    # is_flag=True,
    help="Set the verbose mode.",
)
@click.option(
    "--debug",
    type=bool,
    # is_flag=True,
    help="Set the debug mode.",
)
@click.option(
    "--model_folder_name",  ## model folder name: type everything for custom name, or '* + custom part' to append the part to the default name
    type=str,
    # default=None,
)
@click.option(
    "--model_parent_folder",  ## model parent folder
    type=str,
    # default='Model/',
)
@click.option(
    "--training_temp", type=bool  ## whether require training temperature
)
@click.option(
    "--data_file",  ## data file path for training and evaluation
    type=str,
    # default='prepare_data/fed_subword_128.h5',
)
@click.argument("config_file", type=click.Path(exists=True))

def main(config_file, **kwargs):
    # Load the configuration file
    config = load_config(config_file)
    print("Loaded configuration:")
    print(json.dumps(config, indent=4))

    # Overwrite fields if provided via Click options
    for arg in kwargs.keys():
        if kwargs[arg] is not None:  # Check if the argument was provided
            config[arg] = kwargs[arg]
            print(f"Overwriting {arg} with {kwargs[arg]}")

    if config["training_temp"]:
        config["model_folder_name"] = "*tmp"

    # # Display the updated configuration
    print("\nUpdated configuration:")
    # print(json.dumps(config, indent=4))

    print(
        f"mode: {config['mode']}",
        ", device:",
        config["device"],
        ", block:",
        config["block"],
        ", initial A:",
        config["initblock"],
        ", fulltoken:",
        config["fulltoken"],
        ", layer:",
        config["layer"],
    )
    print(
        "max_iterations:",
        config["max_iters"],
        ", epoch:",
        config["epoch"],
        f"embedding dim: {config['d']}",
        ", context length:",
        config["context"],
        ", stride:",
        config["stride"],
        ", batch size:",
        config["batchsize"],
        ", train ratio:",
        config["splitratio"],
        ", learning rate:",
        config["lr"],
        ", dropout rate:",
        config["dropout"],
    )
    if config["cross"] is not None:
        print("cross evaluation")
    else:
        print("single evaluation")
    print(
        "model checkpoint:",
        config["checkpoint"],
        ", save figure/output?",
        config["saveflag"],
        ".",
    )
    
    datafilepath = config["data_file"]
    print(f"Data file: {datafilepath}")
    with h5py.File(datafilepath, 'r') as f:
        dict = eval(f['params'][()])
    n = dict['n_train'] + dict['n_validation']
    observation_dim = dict['m'] 
    lamda = dict['lamda']

    path_manager = PathManager(__file__)
    Logging.basic_config(
        filename=path_manager.logging_filepath,
        verbose=config["verbose"],
        debug=config["debug"],
    )

    if config["device"] == "cpu":
        _device = Device.CPU
    elif config["device"] == "gpu" or "cuda":
        _device = Device.CUDA
    elif config["device"] == "mps":
        _device = Device.MPS
    else:
        raise ValueError(f"Unknown device: {config['device']}")
    device_manager = DeviceManager(_device)
    print(f"actual device used: {device_manager.device}")

    ### acquire text data
    print("Tokens:", n*config["context"])
    print("Unique tokens", observation_dim)
    tokenizer = tiktoken.get_encoding("gpt2") #useless but keep for compatibility

    ### count token/iter, iter/epoch
    token_per_iter = config["batchsize"] * config["context"]
    iter_per_epoch = int(
        np.ceil(
            int((n*config["context"]) * config["splitratio"])
            / config["batchsize"]
            / config["context"]
        )
    )
    print("Tokens per iteration:", token_per_iter)
    print("Iterations per epoch:", iter_per_epoch)
    if config["iterflag"]:
        epoch_new = int(np.ceil(config["max_iters"] / iter_per_epoch))
        print(config["max_iters"], "iterations needs", epoch_new, "epochs")
        if config["epoch"] != epoch_new:
            config["epoch"] = epoch_new
            print(
                "Ignore epoch, training according to max_iters. Epoch number set to",
                epoch_new,
            )
    else:
        config["max_iters"] = int(np.ceil(config["epoch"] * iter_per_epoch))
        print(
            "Ignore max_iters, training according to epoch number. Max iterations set to",
            config["max_iters"],
        )

    user_config = UserConfig(
        block=config["block"],
        initblock=config["initblock"],
        fulltoken=config["fulltoken"],
        layer=config["layer"],
        max_iters=config["max_iters"],
        epoch=config["epoch"],
        iterflag=config["iterflag"],
        d=config["d"],
        context=config["context"],
        stride=config["stride"],
        batchsize=config["batchsize"],
        lr=config["lr"],
        dropout=config["dropout"],
        splitratio=config["splitratio"],
        device=config["device"],
        m=observation_dim,
        tokenizer=tokenizer,
        eval_interval=config["eval_interval"],
        temp=config["temp"],
        topk=config["topk"],
        start_text=config["start_text"],
        max_new_tokens=config["max_new_tokens"],
        token_per_iter=token_per_iter,
        iter_per_epoch=iter_per_epoch,
        model_parent_folder=config["model_parent_folder"],
        training_temp=config["training_temp"],
    )

    (
        model_folder_path,
        model_file_name,
        model_filepath,
        model_folder_path_name,
        model_folder_name,
    ) = get_model_path(
        path_manager,
        user_config,
        checkpoint=config["checkpoint"],
        auto_create=(config["mode"] == "train"),
        model_folder_name=config["model_folder_name"],
    )
    print(
        "model_folder_name:", model_folder_path_name + model_folder_name, "\n"
    )

    if config["mode"] == "train":
        log_file = get_time_log_file_path(user_config)
        execution_time_per_iter, token_per_sec = train(
            model_folder_path, model_file_name, user_config, data_file=datafilepath
        )
        with open(log_file, "a") as f:
            f.write(
                f"{execution_time_per_iter}, {token_per_sec}, {config['layer']}, {config['d']}, {config['context']}, {config['batchsize']}, {config['lr']}, {config['dropout']} \n"
            )
    elif config["mode"] == "visual":
        visualization(
            model_folder_path,
            model_file_name,
            user_config,
            saveflag=config["saveflag"],
            )
    elif config["mode"] == "get_matricies":
        get_matricies(
            model_folder_path,
            model_file_name,
            user_config,
            matrix_filename=f"learning_hmm_d{user_config.d}_m{user_config.m}_lamda{lamda}_batchsize{config['batchsize']}_iters{config['max_iters']}.h5",
        )
    
    if not config["saveflag"]:
        plt.show()


if __name__ == "__main__":
    main()