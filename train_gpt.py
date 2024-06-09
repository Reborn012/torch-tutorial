# Code adapted from https://github.com/karpathy/nanoGPT

import argparse
import os
import pickle
import numpy as np
import math
import time
from contextlib import nullcontext
import logging

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.gpt2 import GPT
from model.gpt2 import GPTConfig as ModelConfig
from utility import load_yaml_config, update_default_configs, print_color

class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.block_size = block_size
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get one block of data
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return x, y
    
def init_dataloader(config, ddp):
    data_dir = os.path.join('data', config['dataset'])
    train_file_path = os.path.join(data_dir, 'train.bin')
    val_file_path = os.path.join(data_dir, 'val.bin')

    # Create datasets
    train_dataset = TextDataset(train_file_path, config['block_size'])
    val_dataset = TextDataset(val_file_path, config['block_size'])

    # Create samplers
    train_sampler = DistributedSampler(train_dataset) if ddp else None
    val_sampler = DistributedSampler(val_dataset) if ddp else None

    # Create data loaders
    device = config['device']
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler, 
                                pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=(val_sampler is None), sampler=val_sampler, 
                                pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, sampler=val_sampler)

    return train_loader, val_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    return args

def initialize_ddp(backend):
    """
    Initialize the Distributed Data Parallel (DDP) environment.

    Args:
        backend (str): The backend to use for distributed training (e.g., 'nccl', 'gloo').

    Returns:
        tuple: Contains ddp_rank, ddp_local_rank, ddp_world_size, and device information.
            - ddp_rank (int): The global rank of the process.
            - ddp_local_rank (int): The local rank of the process on the node.
            - ddp_world_size (int): The total number of processes.
            - device (str): The CUDA device to be used by this process.
    """
    # Initialize the process group for distributed training
    init_process_group(backend=backend)

    # Retrieve the unique global rank of the process from the environment variable
    ddp_rank = int(os.environ['RANK'])

    # Retrieve the local rank (GPU no. in local node) of the process on the node from the environment variable
    ddp_local_rank = int(os.environ['LOCAL_RANK'])

    # Retrieve the total number of processes from the environment variable
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    # Determine the CUDA device to be used by this process based on the local rank
    device = f'cuda:{ddp_local_rank}'

    # Set the CUDA device for this process
    torch.cuda.set_device(device)

    return ddp_rank, ddp_local_rank, ddp_world_size, device

def setup_master_process(ddp_rank):
    # Determine if the current process is the master process.
    master_process = ddp_rank == 0  # Check if this is the master process
    seed_offset = ddp_rank  # Set random seed offset based on the global rank
    return master_process, seed_offset

def setup_single_process():
    # Setup for a single-process (non-distributed) training.
    master_process = True  # Single process is always the master
    seed_offset = 0  # Random seed offset is 0
    ddp_world_size = 1  # World size is 1 for single-process training
    return master_process, seed_offset, ddp_world_size

def initialize_model(model_args, init_from, out_dir, device, compile_model=True):
    if init_from == 'scratch':
        print_color("Initializing a new model from scratch", 'yellow')
        if model_args['vocab_size'] is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = 50304
        gptconf = ModelConfig(**model_args) # Update the model configuration based on the provided arguments
        model = GPT(gptconf)
    elif init_from == 'resume':
        print_color(f"Resuming training from {out_dir}", 'yellow')
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = ModelConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        print_color(f"Initializing from OpenAI GPT-2 weights: {init_from}", 'yellow')
        override_args = dict(dropout=model_args['dropout'])
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    else:
        raise ValueError("Unknown value for init_from")
    # crop down the bock size (i.e., context length) if necessary
    block_size = model_args['block_size']
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size
    model.to(device)
    if compile_model:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    return model, checkpoint if init_from == 'resume' else None

@torch.no_grad()
def estimate_loss(model, eval_iters, ctx, config, train_loader, val_dataloader):
    out = {}
    model.eval()

    device = config['device']
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        dataloader = train_loader if split == 'train' else val_dataloader
        for batch_idx, (X, Y) in enumerate(dataloader):
            if batch_idx >= eval_iters:
                break
            if device_type == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            else:
                X, Y = X.to(device), Y.to(device)

            with ctx:
                logits, loss = model(X, Y)
            losses[batch_idx] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup), based on nanoGPT
def get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def create_logger(log_dir, config):
    model_name = config['model']

    logger = logging.getLogger(f'Training Log')
    logger.setLevel(logging.INFO)  # Set to INFO to avoid logging debug-level messages
    file_handler = logging.FileHandler(f'{log_dir}/{model_name}.log', mode='w')
    formatter = logging.Formatter('%(message)s')  # Simplified format without time or debug level
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def main():
    args = parse_arguments()

    # Default configuration, generated based on train.py from nanoGPT repo
    # Can be overridden by the YAML configuration
    config = {
        "model": "gpt2", # model name, used for logging
        "out_dir": 'out',
        "eval_interval": 2000,
        "log_interval": 1,
        "eval_iters": 200,
        "eval_only": False, # if True, script exits right after the first eval
        "always_save_checkpoint": True, # if True, always save a checkpoint after each eval
        "init_from": 'scratch', # 'scratch' or 'resume' or 'gpt2*'
        # data
        "dataset": 'openwebtext',
        "gradient_accumulation_steps": 1,   # used to simulate larger batch sizes 
                                            # in local device (CPU or GPU)
        "batch_size": 12,   # if gradient_accumulation_steps > 1, this is the micro-batch size
        "block_size": 1024, # context length
        "vocab_size": None, # DO NOT SET MANUALLY, will be inferred from the dataset
        # model
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.0, # for pretraining 0 is good, for finetuning try 0.1+
        "bias": False,  # do we use bias inside LayerNorm and Linear layers?
        # training
        "num_epochs": 1,   # number of epochs to train for
        "max_iters": -1,  # maximum number of iterations to train for
                            # default is -1, meaning num_epochs will be used; if set, overrides num_epochs
        # adamw optimizer
        "learning_rate": 6e-4,  # max learning rate
        "weight_decay": 1e-1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,   # clip gradients at this value, or disable if == 0.0
        # learning rate decay settings
        "decay_lr": True,   # whether to decay the learning rate
        "warmup_iters": 2000,   # how many steps to warm up for
        "lr_decay_iters": None,     # DO NOT SET MANUALLY, will be set automatically;
                                    # should be ~= max_iters per Chinchilla
        "min_lr": 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        # DDP settings
        "backend": 'nccl',  # 'nccl', 'gloo', etc. 'nccl' for NVLink and 'gloo' for 'PCI-e' communication
        # system
        "device": 'cuda',   # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks. 
                            # Only use for single GPU training. For multi-GPU training, override with DDP. 
        "dtype": 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        "compile": True    # use PyTorch 2.0 to compile the model to be faster; 
    }

    # Load and merge YAML configuration if provided
    yaml_config = load_yaml_config(args.config)
    config = update_default_configs(config, yaml_config)

    # Attempt to derive vocab_size from the dataset
    data_dir = os.path.join('data', config['dataset'])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta.get('vocab_size')
        if meta_vocab_size:
            config['vocab_size'] = meta_vocab_size
            print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    print_color(config, 'yellow')

    # Initial distributed training setup, currently using DDP only
    # @TODO: Implement FSDP
    ddp = int(os.environ.get('RANK', -1)) != -1     # check if we are training distributedly
    if ddp:
        ddp_rank, ddp_local_rank, ddp_world_size, device = initialize_ddp(config['backend'])
        master_process, seed_offset = setup_master_process(ddp_rank)
        config["device"] = device   # override the device with the DDP device
    else:
        master_process, seed_offset, ddp_world_size = setup_single_process()
        device = config['device']

    config['tokens_per_iter'] = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size']
    print(f"tokens per iteration will be: {config['tokens_per_iter']:,}")

    if master_process:
        # Create the output directory if it does not exist
        os.makedirs(config['out_dir'], exist_ok=True)
    torch.manual_seed(1337 + seed_offset)   # fix the random seed based on ranking

    # Enable faster training with mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn 
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    # When training on GPU, use automatic mixed precision (AMP) to speed up training
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    logger = create_logger(config['out_dir'], config)

    # Initialize DataLoader
    train_loader, val_loader = init_dataloader(config, ddp)
    max_iters = len(train_loader) * config['num_epochs']
    config['lr_decay_iters'] = max_iters  # set the lr decay iters based on the max iters


    # Model initialization
    model_args = dict(n_layer=config['n_layer'], n_head=config['n_head'], n_embd=config['n_embd'], block_size=config['block_size'],
                      bias=config['bias'], vocab_size=config['vocab_size'], dropout=config['dropout'])
    model, checkpoint = initialize_model(model_args, config['init_from'], config['out_dir'], device, config['compile'])
    
    # Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))

    # Optimizer
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
    if checkpoint is not None:  # if resuming training
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory
    
    # Wrap model into DDP container if training distributedly
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Training loop
    print_color("starting training...", 'yellow')
    t0 = time.time()
    iter_num, running_mfu = 0, -1.0
    best_val_loss = 1e9
    if checkpoint is not None:
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    raw_model = model.module if ddp else model  # unwrap DDP container if needed

    for epoch in range(config['num_epochs']):
        logger.info(
            f"epoch {epoch:05d} "
            f"| num_steps_per_epoch {len(train_loader)} "
            )
        for batch_idx, (X, Y) in enumerate(train_loader):
            if device_type == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            else:
                X, Y = X.to(device), Y.to(device)
            
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num, config['warmup_iters'], config['learning_rate'], config['lr_decay_iters'], config['min_lr']) if config['decay_lr'] else config['learning_rate']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % config['eval_interval'] == 0 and master_process:
                losses = estimate_loss(model, config['eval_iters'], ctx, config, train_loader, val_loader)
                print(
                            f"epoch {epoch:05d} "
                            f"| step {iter_num:010d} "
                            f"| train loss {losses['train']:.4f} "
                            f"| val loss {losses['val']:.4f}"
                        )
                logger.info(
                            f"epoch {epoch:05d} "
                            f"| step {iter_num:010d} "
                            f"| train loss {losses['train']:.4f} "
                            f"| val loss {losses['val']:.4f}"
                        )
                if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        print(f"saving checkpoint to {config['out_dir']}")
                        torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
            if iter_num == 0 and config['eval_only']:
                break

            if config['max_iters'] != -1 and iter_num >= config['max_iters']:
                print_color("reached max iterations, stopping training", 'red')
                break

            # Forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(config['gradient_accumulation_steps']):
                if ddp:
                    # From NanoGPT:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / config['gradient_accumulation_steps'] # scale the loss to account for gradient accumulation
                                                                        # normalize the gradient under the hood
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradients if necessary
            if config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % config['log_interval'] == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * config['gradient_accumulation_steps']
                if iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(
                    f"epoch {epoch:05d} "
                    f"| step {iter_num:010d} "
                    f"| loss {lossf:.4f} "
                    f"| time {dt * 1000:.2f}ms "
                    f"| mfu {running_mfu * 100:.2f}%"
                )
                logger.info(
                    f"epoch {epoch:05d} "
                    f"| step {iter_num:010d} "
                    f"| loss {lossf:.4f} "
                    f"| time {dt * 1000:.2f}ms "
                    f"| mfu {running_mfu * 100:.2f}%"
                )
            iter_num += 1
    
    if ddp:
        destroy_process_group()
    

if __name__ == '__main__':
    main()
