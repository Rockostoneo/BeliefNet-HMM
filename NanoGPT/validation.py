import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import h5py
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cpu' 
ptdtype = torch.bfloat16
ctx = nullcontext() 

#Edit these to point to the right files and directories for your setup
data_file_path = '../prepare_data/data_d64_m32_lamda0.9_example.h5'
out_dir = 'm32_d64_1l1h_example'
split = 'test' # 'train' or 'validation' or 'test'

# model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
device = 'cpu'
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Load state dict into model
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()

def get_batch(data_file_path):
    data_file = h5py.File(data_file_path, 'r') 
    eval_split = f'z_{split}'
    inputs = data_file[eval_split][:,:-1] 
    targets = data_file[eval_split][:,1:]
    data_file.close()
    x = torch.from_numpy(inputs.astype(np.int64))  
    y = torch.from_numpy(targets.astype(np.int64))  
    x, y = x.to(device), y.to(device)
    return x, y

# Get validation batch and compute loss
x, y = get_batch(data_file_path)
with torch.no_grad():
    logits, loss = model(x, y)
    
print(f'Validation loss: {loss.item():.4f}')
print(f'Validation perplexity: {torch.exp(loss).item():.4f}')