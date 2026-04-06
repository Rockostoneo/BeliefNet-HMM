from typing import Union, Optional, Any
from pathlib import Path
from ss.utility.path import PathManager
import os
import json
import torch
import tiktoken
from dataclasses import dataclass
# from ss.utility.learning.mode import LearningMode
from ss.utility.learning.process import BaseLearningProcess
from ss.estimation.filtering.hmm.learning.module import LearningHmmFilter

# model_parent_folder = 'Model/'

def load_config(file_path):
    """
    Load configuration from a JSON file.
    
    Configuration options:
    - mode: 'visual', 'train', 'inference', 'inspect'
    - device: 'cpu', 'gpu', 'mps', 'cuda'
    - block: 'full_matrix', 'spatial_invariant', 'iid'
    - initblock: 'normal_distribution', 'uniform_distribution', 'constant', 'identity'
    - fulltoken: boolean flag
    - layer: integer (number of layers)
    - d: integer (embedding dimension)
    - max_iters: integer or null (max number of iterations)
    - epoch: integer (number of epochs)
    - iterflag: boolean flag
    - context: integer (context length)
    - stride: integer or null (stride, default is context length)
    - batchsize: integer (batch size)
    - splitratio: float (train-validation split ratio)
    - lr: float (learning rate)
    - dropout: float (dropout rate)
    - eval_interval: integer (evaluation interval)
    - temp: float (temperature scaling)
    - topk: integer (topk sampling)
    - start_text: string (start text for inference)
    - max_new_tokens: integer (max new tokens for inference)
    - saveflag: boolean flag
    - cross: null or one of 'd', 'context', 'batchsize', 'lr', 'dropout'
    - lb: integer or null (lower bound for hyperparameter evaluation)
    - ub: integer or null (upper bound for hyperparameter evaluation)
    - targetlist: string or null (target list for hyperparameter evaluation)
    - checkpoint: integer or null (model checkpoint file)
    - verbose: boolean flag
    - debug: boolean flag
    - model_folder_name: string or null (custom model folder name)
    - model_parent_folder: string (parent folder for model)
    - training_temp: boolean flag
    """
    with open(file_path, "r") as config_file:
        return json.load(config_file)

class CharTokenizer:
    def __init__(self, text_data):
        self.unique_chars = sorted(list(set(text_data)))
        print('Vocabulary size:', len(self.unique_chars))
        print('Unique characters:', ''.join(self.unique_chars))
        self.i2s = {i: s for i, s in enumerate(self.unique_chars)}
        self.s2i = {s: i for i, s in enumerate(self.unique_chars)}

    def encode(self, s): # encoder: take a string, output a list of integers
        return [self.s2i[c] for c in s]

    def decode(self, x): # decoder: take a list of integers, output a string
        return ''.join([self.i2s[i] for i in x])
    
@dataclass
class UserConfig:
    block: str
    initblock: str
    fulltoken: bool
    layer: int
    max_iters: int
    epoch: int
    iterflag: bool
    d: int
    context: int
    stride: int
    batchsize: int
    lr: float
    dropout: float
    splitratio: float
    device: str
    m: int
    tokenizer: Union[CharTokenizer, tiktoken.core.Encoding]
    eval_interval: int
    temp: float
    topk: int
    start_text: str
    max_new_tokens: int
    token_per_iter: int
    iter_per_epoch: int
    model_parent_folder: str
    training_temp: bool

    def __post_init__(self):
        # Initialize a dictionary to store previous values
        self._previous_values = {}

    def update_target(self, target_variable: str, target_value):
        """
        Updates the specified field in the dataclass if target_variable is valid.
        Saves the previous value for restoration.
        """
        if target_variable is None:
            return  # Do nothing if target_variable is None

        # Check if the target_variable is a valid field in the dataclass
        if target_variable in self.__annotations__:
            # Save the current value before updating
            if target_variable not in self._previous_values:
                self._previous_values[target_variable] = getattr(self, target_variable)
            # Update the field with the new value
            setattr(self, target_variable, target_value)
        else:
            raise ValueError(f"Invalid target_variable: {target_variable}. Must be one of {list(self.__annotations__.keys())}")

    def restore(self, target_variable: str):
        """
        Restores the specified field to its previous value.
        """
        if target_variable is None:
            return  # Do nothing if target_variable is None

        # Check if the target_variable has a previous value stored
        if target_variable in self._previous_values:
            # Restore the previous value
            setattr(self, target_variable, self._previous_values[target_variable])
            # Remove the previous value from the dictionary
            del self._previous_values[target_variable]
        else:
            raise ValueError(f"No previous value found for {target_variable}. Cannot restore.")

def get_model_path(path_manager: PathManager, user_config: UserConfig, checkpoint=None, auto_create=False, model_folder_name=None) -> tuple[Path, str, Path, str, Any | str]:
        model_folder_path_name = user_config.model_parent_folder + user_config.block + '/'
        if model_folder_name is None or model_folder_name == 'None' or model_folder_name[0] == '*':
            if model_folder_name is not None and model_folder_name[0] == '*':
                suffix = model_folder_name[1:]
            else:
                suffix = ''
            model_folder_name = ''
            # if user_config.device != 'cpu':
            model_folder_name += user_config.device + '_'
            if user_config.initblock != 'normal_distribution':
                if user_config.initblock == 'uniform_distribution':
                    model_folder_name += f"unif_"
                elif user_config.initblock == 'constant':
                    model_folder_name += f"const_"
                elif user_config.initblock == 'identity':
                    model_folder_name += f"eye_"
            if user_config.fulltoken:
                model_folder_name += f"fulltoken_"
            else:
                model_folder_name += f"char_"
            if not user_config.iterflag: # user_config.epoch != 50 and user_config.epoch is not None:
                model_folder_name += f"e{user_config.epoch}_"
            else:
                model_folder_name += f"i{user_config.max_iters}_"
            if user_config.layer > 1:
                model_folder_name += f"l{user_config.layer}_"
            if user_config.context != 256:
                model_folder_name += f"c{user_config.context}_"
            if user_config.batchsize != 4:
                model_folder_name += f"b{user_config.batchsize}_"
            if user_config.lr != 0.1:
                model_folder_name += f"lr{user_config.lr}_"
            if user_config.dropout != 0.2:
                model_folder_name += f"dr{user_config.dropout}_"
            model_folder_name += f"d{user_config.d}"
            if suffix != '':
                model_folder_name += '_' + suffix

        model_folder_path = path_manager.get_directory(model_folder_path_name+model_folder_name, auto_create=auto_create)

        if checkpoint is None:
            model_file_name = "learning_filter.pt"
        else:
            model_file_name = f"learning_filter_checkpoint_{checkpoint:02d}.pt"
        model_filepath = model_folder_path / model_file_name

        return model_folder_path, model_file_name, model_filepath, model_folder_path_name, model_folder_name

def get_file_name(user_config: UserConfig, target_variable: Optional[str] =None, inference=False, file_name='') -> str:
    file_prefix = ''
    # if user_config.device != 'cpu':
    file_prefix += user_config.device + '_'
    if user_config.initblock != 'normal_distribution':
        if user_config.initblock == 'uniform_distribution':
            file_prefix += f"unif_"
        elif user_config.initblock == 'constant':
            file_prefix += f"const_"
        elif user_config.initblock == 'identity':
            file_prefix += f"eye_"
    if user_config.fulltoken:
        file_prefix += f"fulltoken_"
    else:
        file_prefix += f"char_"
    if target_variable != 'layer':
        file_prefix += f"l{user_config.layer}_"
    if not user_config.iterflag:
        file_prefix += f"e{user_config.epoch}_"
    else:
        file_prefix += f"i{user_config.max_iters}_"
    if target_variable != 'context':
        file_prefix += f"c{user_config.context}_"
    if target_variable != 'batchsize':
        file_prefix += f"b{user_config.batchsize}_"
    if target_variable != 'lr':
        file_prefix += f"lr{user_config.lr}_"
    if target_variable != 'dropout':
        file_prefix += f"dr{user_config.dropout}_"
    if target_variable != 'd':
        file_prefix += f"d{user_config.d}_"
    if inference:
        file_prefix += 'temp' + str(user_config.temp) + '_topk' + str(user_config.topk) + '_'
    file_path_name = file_prefix + file_name
    if target_variable is not None:
        file_path_name += '_' + target_variable
    return file_path_name

def get_time_log_file_path(user_config: UserConfig):
    prefix = user_config.device + '_' # '' if user_config.device == 'cpu' else user_config.device + '_'
    prefix += 'char_' if not user_config.fulltoken else 'fulltoken_'
    # if not user_config.iterflag: ## this is temporary, need to be removed
    prefix += 'full_matrix_' if user_config.block == 'full_matrix' else 'spatial_invariant_'
    log_file_name = "execution_times.txt"
    log_file_path = user_config.model_parent_folder + user_config.block + '/' + prefix + log_file_name
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            # f.write("execution_time_per_epoch [s], layer, epoch, d, context, batchsize, lr, dropout\n")
            f.write("execution_time_per_iter [s], token_per_sec, layer, d, context, batchsize, lr, dropout\n")
    return log_file_path

def token_ids_to_text(obsv_state, tokenizer: Union[CharTokenizer, tiktoken.core.Encoding]):
    # indices = obsv_state.tolist()
    # return tokenizer.decode(indices)
    return tokenizer.decode(obsv_state)

def text_to_token_ids(text, tokenizer: Union[CharTokenizer, tiktoken.core.Encoding]):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

# ### new text generator, consider top-k sampling and temp scaling.
# def generate_text(learning_filter, context_size, start_text, tokenizer, max_new_tokens=100, temp=0.0, top_k=None, filter_reset=True):
#     indices = text_to_token_ids(start_text, tokenizer)
#     if not filter_reset:
#         learning_filter.update(indices)
#     # For-loop is the same as before: Get logits, and only focus on last time step
#     with Mode.inference(learning_filter):
#         for _ in range(max_new_tokens):
#             if filter_reset:
#                 idx_cond = indices[:, -context_size:]
#                 learning_filter.reset()
#                 learning_filter.update(idx_cond)
#             learning_filter.estimate()
#             estimated_next_observation_probability = (learning_filter.predicted_next_observation_probability).unsqueeze(0)
#             logits = torch.log(estimated_next_observation_probability + 1e-16)  # (batch_size, context_len, vocab_size)

#             # New: Filter logits with top_k sampling
#             if top_k is not None:
#                 # Keep only top_k values
#                 top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#                 min_val = top_logits[:, -1]
#                 logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

#             # New: Apply temp scaling
#             if temp > 0.0:
#                 logits = logits / temp

#                 # Apply softmax to get probabilities
#                 probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

#                 # Sample from the distribution
#                 idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

#             # Otherwise same as before: get idx of the vocab entry with the highest logits value
#             else:
#                 idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        
#             # idx_next = learning_filter.predict()

#             if not filter_reset:
#                 learning_filter.update(idx_next)
#             # Same as before: append sampled index to the running sequence
#             indices = torch.cat((indices, idx_next), dim=1)  # (batch_size, num_tokens+1)
#     return indices.squeeze(0).numpy()

### compact text generator
def generate_text(learning_filter: LearningHmmFilter, context_size, start_text, tokenizer, max_new_tokens=100, filter_reset=True):
    indices = text_to_token_ids(start_text, tokenizer)
    if not filter_reset:
        learning_filter.update(indices)
    # For-loop is the same as before: Get logits, and only focus on last time step
    with BaseLearningProcess.inference_mode(learning_filter):
        for _ in range(max_new_tokens):
            if filter_reset:
                idx_cond = indices[:, -context_size:]
                learning_filter.reset()
                learning_filter.update(idx_cond)

            idx_next = learning_filter.predict().unsqueeze(0)  # (batch_size, 1)
            
            if not filter_reset:
                learning_filter.update(idx_next)
            indices = torch.cat((indices, idx_next), dim=1)  # (batch_size, num_tokens+1)
    return indices.squeeze(0).numpy()

