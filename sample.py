"""
Code adapted from https://github.com/karpathy/nanoGPT
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model.gpt2 import GPT
from model.gpt2 import GPTConfig
import argparse
from utility import load_yaml_config, update_default_configs, print_color

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    return args

config = {
    'init_from': 'resume',    # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    'out_dir': 'out',         # ignored if init_from is not 'resume'
    'start': "\n",            # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    'num_samples': 10,        # number of samples to draw
    'max_new_tokens': 500,    # number of tokens generated in each sample
    'temperature': 0.8,       # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    'top_k': 200,             # retain only the top_k most likely tokens, clamp others to have 0 probability
    'seed': 1337,
    'device': 'cuda',         # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', # 'float32' or 'bfloat16' or 'float16'
    'compile': False          # use PyTorch 2.0 to compile the model to be faster
}

args = parse_arguments()
# Load and merge YAML configuration if provided
yaml_config = load_yaml_config(args.config)
config = update_default_configs(config, yaml_config)



torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in config['device'] else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if config['init_from'] == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=config['device'])
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif config['init_from'].startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(config['init_from'], dict(dropout=0.0))

model.eval()
model.to(config['device'])
if config['compile']:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if config['init_from'] == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if config['start'].startswith('FILE:'):
    with open(config['start'][5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(config['start'])
x = (torch.tensor(start_ids, dtype=torch.long, device=config['device'])[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(config['num_samples']):
            y = model.generate(x, config['max_new_tokens'], temperature=config['temperature'], top_k=config['top_k'])
            print(decode(y[0].tolist()))
            print('---------------')
