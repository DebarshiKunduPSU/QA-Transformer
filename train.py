

import os
import math
import requests
import torch
import tiktoken
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import GPTModel, GPTDatasetV1, generate_text_simple, set_tensor_parallel_group


def ddp_init():
    # requires torchrun
    if 'RANK' not in os.environ:
        raise RuntimeError('Run with torchrun. Example: torchrun --standalone --nproc_per_node=8 gpt_train_quantum_dp_tp.py')

    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    return rank, world_size, local_rank, device


def build_dp_tp_groups(rank: int, world_size: int, tp_size: int):
    if world_size % tp_size != 0:
        raise ValueError(f'WORLD_SIZE ({world_size}) must be divisible by tp_size ({tp_size}).')

    dp_size = world_size // tp_size
    dp_index = rank // tp_size
    tp_index = rank % tp_size

    # TP groups: contiguous blocks
    tp_groups = []
    for d in range(dp_size):
        ranks = list(range(d * tp_size, (d + 1) * tp_size))
        tp_groups.append(dist.new_group(ranks=ranks))

    # DP groups: same tp_index across blocks
    dp_groups = []
    for t in range(tp_size):
        ranks = [d * tp_size + t for d in range(dp_size)]
        dp_groups.append(dist.new_group(ranks=ranks))

    tp_group = tp_groups[dp_index]
    dp_group = dp_groups[tp_index]

    return {
        'tp_size': tp_size,
        'dp_size': dp_size,
        'tp_index': tp_index,
        'dp_index': dp_index,
        'tp_group': tp_group,
        'dp_group': dp_group,
    }


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def ddp_mean(x: float, device: torch.device, group) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    t /= dist.get_world_size(group)
    return t.item()


def evaluate_loss(data_loader, model, device, num_batches=None):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            loss = calc_loss_batch(inp, tgt, model, device)
            tot += loss.item()
            n += 1
    model.train()
    return tot / max(n, 1)


def tp_allreduce_grads(model, tp_group):
    # SUM across TP ranks because each TP rank computed disjoint query-token rows
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=tp_group)


def main():
    rank, world_size, local_rank, device = ddp_init()
    is_global_main = (rank == 0)

    # Choose TP size. With 8 A100s, a good default is TP=4, DP=2.
    # You can override via environment variable: TP_SIZE=2/4/8
    tp_size = int(os.environ.get('TP_SIZE', '4'))
    groups = build_dp_tp_groups(rank, world_size, tp_size)

    tp_group = groups['tp_group']
    dp_group = groups['dp_group']
    dp_size = groups['dp_size']
    dp_index = groups['dp_index']
    tp_index = groups['tp_index']

    # Let the model know which group to use for token-parallel attention
    set_tensor_parallel_group(tp_group)

    # RNG strategy:
    # - Same seed for all TP ranks inside a DP replica (keeps replicated ops consistent)
    # - Different seed across DP replicas (good stochasticity)
    torch.manual_seed(1234 + dp_index)
    torch.cuda.manual_seed_all(1234 + dp_index)

    # Download data once
    file_path = 'the-verdict.txt'
    url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt'

    if is_global_main and not os.path.exists(file_path):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(r.text)

    dist.barrier()

    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    tokenizer = tiktoken.get_encoding('gpt2')

    # TINY config — adjust carefully; quantum attention cost scales ~T^2
    GPT_CONFIG = {
        'vocab_size': 50257,
        'context_length': 64,
        'emb_dim': 128,
        'n_heads': 8,
        'n_layers': 4,
        'drop_rate': 0.1,
        'qkv_bias': False,

        # quantum
        'attn_type': 'quantum',
        'use_quantum_pos': True,
        'quantum_angle_scale': math.pi,
        'quantum_n_qubits': 2,
        'quantum_heads': 1,

        # TRUE tensor-parallel acceleration of the score matrix
        'quantum_tensor_parallel': True,
        'tp_disable_attn_dropout': True,
    }

    OTHER = {
        'learning_rate': 5e-4,
        'num_epochs': 2,
        'batch_size': 2,
        'weight_decay': 0.1,
        'eval_freq': 10,
        'eval_iter': 2,
    }

    context_len = GPT_CONFIG['context_length']
    split_idx = int(0.90 * len(text_data))

    train_dataset = GPTDatasetV1(text_data[:split_idx], tokenizer, max_length=context_len, stride=context_len)
    val_dataset = GPTDatasetV1(text_data[split_idx:], tokenizer, max_length=context_len, stride=context_len)

    # Data parallel sharding only over DP replicas (dp_size). TP ranks inside the same DP replica share data.
    train_sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_index, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dp_size, rank=dp_index, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=OTHER['batch_size'], sampler=train_sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=OTHER['batch_size'], sampler=val_sampler,
                            num_workers=0, pin_memory=True, drop_last=False)

    # Model
    model = GPTModel(GPT_CONFIG).to(device)

    # Make sure params are identical across ALL ranks (global broadcast)
    # (helps avoid any accidental init mismatch across TP ranks)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # DDP across DP group only
    # IMPORTANT: we do TP gradient SUM manually after DDP's DP averaging.
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, process_group=dp_group,
                    broadcast_buffers=False, find_unused_parameters=False)

    opt = torch.optim.AdamW(ddp_model.parameters(), lr=OTHER['learning_rate'], weight_decay=OTHER['weight_decay'])

    def gen_sample(start_context: str):
        # Only run sample generation on one representative rank to avoid spam.
        # Pick global rank 0.
        if not is_global_main:
            return
        ddp_model.eval()
        with torch.no_grad():
            idx = text_to_token_ids(start_context, tokenizer).to(device)
            out = generate_text_simple(ddp_model.module, idx=idx, max_new_tokens=50, context_size=context_len)
            print(token_ids_to_text(out, tokenizer).replace('\n', ' '))
        ddp_model.train()

    # Train
    global_step = 0
    for epoch in range(OTHER['num_epochs']):
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        for inp, tgt in train_loader:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            loss = calc_loss_batch(inp, tgt, ddp_model, device)
            loss.backward()

            # Ensure DDP's DP reductions are done before TP sum
            dist.barrier(group=dp_group)

            # SUM grads across TP ranks (disjoint token-rows)
            tp_allreduce_grads(ddp_model.module, tp_group)

            opt.step()

            if global_step % OTHER['eval_freq'] == 0:
                # Evaluate only on one representative TP rank per DP replica to save compute:
                # tp_index==0 are the leaders of each TP group. We then average across DP leaders.
                if tp_index == 0:
                    tr = evaluate_loss(train_loader, ddp_model, device, num_batches=OTHER['eval_iter'])
                    va = evaluate_loss(val_loader, ddp_model, device, num_batches=OTHER['eval_iter'])
                    tr = ddp_mean(tr, device, group=dp_group)
                    va = ddp_mean(va, device, group=dp_group)
                    if is_global_main:
                        print(f'Ep {epoch+1} step {global_step:06d} | train {tr:.3f} | val {va:.3f}')

        gen_sample('Every effort moves you')

    if is_global_main:
        torch.save(ddp_model.module.state_dict(), 'model_dp_tp.pth')

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
