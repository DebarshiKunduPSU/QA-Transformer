import os
import math
import requests
import torch
import tiktoken
import torch.distributed as dist
import matplotlib.pyplot as plt

from previous_chapters_quantum_tp import GPTModel, GPTDatasetV1, generate_text_simple


def is_distributed_env() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def setup_dist():
    if not is_distributed_env():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return True, rank, world_size, local_rank, device


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=40, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


def average_grads(model, world_size: int):
    """Synchronize gradients across ranks (needed because we are NOT using DDP)."""
    for p in model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def main():
    dist_on, rank, world_size, local_rank, device = setup_dist()
    is_main = (rank == 0)

    # NOTE: tensor-parallel token slicing requires ALL ranks to see the SAME batch
    # (we are parallelizing the T x T score matrix, not sharding the data).
    # So: no DistributedSampler here.

    torch.manual_seed(123)

    # Download data once
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if is_main and not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)

    if dist_on:
        dist.barrier()

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")

    GPT_CONFIG = {
        "vocab_size": 50257,
        "context_length": 64,   # keep small; score matrix is O(T^2)
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,

        # quantum
        "attn_type": "quantum",
        "use_quantum_pos": True,
        "quantum_angle_scale": math.pi,
        "quantum_n_qubits": 2,

        # huge speed win: only 1 head quantum
        "quantum_heads": 1,

        # TRUE multi-GPU acceleration: parallelize quantum score matrix over token rows
        "quantum_tensor_parallel": True,
    }

    OTHER = {
        "lr": 5e-4,
        "weight_decay": 0.1,
        "batch_size": 2,
        "epochs": 3,
    }

    dataset = GPTDatasetV1(
        txt=text_data,
        tokenizer=tokenizer,
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
    )

    # Same order on all ranks; shuffle False to keep perfectly aligned batches
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=OTHER["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    model = GPTModel(GPT_CONFIG).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=OTHER["lr"], weight_decay=OTHER["weight_decay"])

    start_context = "Every effort moves you"

    model.train()
    step = 0
    for epoch in range(OTHER["epochs"]):
        for input_batch, target_batch in loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optim.zero_grad(set_to_none=True)

            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )

            loss.backward()

            if dist_on:
                # Make sure all ranks take the same step
                average_grads(model, world_size)

            optim.step()
            step += 1

            if is_main and step % 10 == 0:
                print(f"epoch {epoch+1} step {step}: loss {loss.item():.4f}")

        if is_main:
            generate_and_print_sample(model, tokenizer, device, start_context)

    if is_main:
        torch.save(model.state_dict(), "model_tp.pth")

    if dist_on:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
