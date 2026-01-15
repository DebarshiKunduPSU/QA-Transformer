# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

import tiktoken
import torch
import torch.distributed as dist
from torch.autograd import Function as _TorchAutogradFunction

def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def _tp_get_rank_world(group=None):
    if not _dist_is_initialized():
        return 0, 1
    return dist.get_rank(group=group), dist.get_world_size(group=group)

def _tp_token_partition(num_tokens: int, rank: int, world_size: int):
    """Contiguous block partition of [0, num_tokens) across world_size ranks."""
    base = num_tokens // world_size
    rem = num_tokens % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end

class AllGatherTokens(_TorchAutogradFunction):
    """Autograd-safe all_gather along token dimension (dim=1) for (B, T_local, C)."""

    @staticmethod
    def forward(ctx, x, group=None):
        if not _dist_is_initialized():
            ctx.start = 0
            ctx.end = x.shape[1]
            return x

        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)

        local_len = torch.tensor([x.shape[1]], device=x.device, dtype=torch.int64)
        lens = [torch.zeros_like(local_len) for _ in range(world)]
        dist.all_gather(lens, local_len, group=group)
        lens = [int(t.item()) for t in lens]
        max_len = max(lens)

        if x.shape[1] < max_len:
            pad = torch.zeros((x.shape[0], max_len - x.shape[1], x.shape[2]),
                              device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, pad], dim=1)
        else:
            x_pad = x

        gathered = [torch.empty_like(x_pad) for _ in range(world)]
        dist.all_gather(gathered, x_pad, group=group)

        chunks = [g[:, :lens[r], :] for r, g in enumerate(gathered)]
        out = torch.cat(chunks, dim=1)

        start = sum(lens[:rank])
        end = start + lens[rank]
        ctx.start = start
        ctx.end = end
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output[:, ctx.start:ctx.end, :].contiguous(), None


# Optional: set a custom tensor-parallel process group (for DP+TP setups).
TP_GROUP = None

def set_tensor_parallel_group(group):
    global TP_GROUP
    TP_GROUP = group

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.tensor_parallel = bool(tensor_parallel)
        self.tp_group = tp_group if tp_group is not None else TP_GROUP
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, _ = x.shape
    
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)
    
        # -> (B, heads, T, head_dim)
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    
        # Tensor-parallel over query tokens (rows of the score matrix)
        if self.tensor_parallel and _dist_is_initialized():
            rank, world = _tp_get_rank_world(group=self.tp_group)
            q_start, q_end = _tp_token_partition(num_tokens, rank, world)
        else:
            q_start, q_end = 0, num_tokens
    
        mask_bool_full = self.mask.bool()[:num_tokens, :num_tokens]
        mask_local = mask_bool_full[q_start:q_end, :]
    
        # Precompute absolute positional angles for keys/queries if enabled
        pos_full = None
        if self.quantum_heads > 0:
            if self.qattn[0].use_quantum_pos:
                pos_full = self.qattn[0].sinusoidal_pos_angles(num_tokens, self.n_qubits, x.device)  # (T, n_qubits)
                pos_full = pos_full.unsqueeze(0).expand(b, -1, -1)  # (B,T,n_qubits)
            else:
                pos_full = torch.zeros((b, num_tokens, self.n_qubits), device=x.device, dtype=torch.float32)
    
        head_outs = []
    
        # Quantum heads
        for h in range(self.quantum_heads):
            qh_local = q[:, h, q_start:q_end, :]      # (B, Tq, head_dim)
            kh_full  = k[:, h, :, :]                  # (B, T, head_dim)
            vh_full  = v[:, h, :, :]                  # (B, T, head_dim)
    
            q_proj = self._proj_angles(qh_local, h, "q")
            k_proj = self._proj_angles(kh_full,  h, "k")
    
            q_ang = self._to_angles(q_proj).to(torch.float32)   # (B, Tq, n_qubits)
            k_ang = self._to_angles(k_proj).to(torch.float32)   # (B, T,  n_qubits)
    
            pos1_local = pos_full[:, q_start:q_end, :] if pos_full is not None else None
            pos2_full  = pos_full
    
            scores_local = self.qattn[h](q_ang, k_ang, pos1=pos1_local, pos2=pos2_full).to(torch.float32)  # (B,Tq,T)
            scores_local = scores_local.masked_fill(mask_local, -1e9)
    
            attn_weights = torch.softmax(scores_local / self.softmax_temp, dim=-1)
            attn_weights = self.dropout(attn_weights)
    
            ctx_local = attn_weights @ vh_full  # (B,Tq,head_dim)
            head_outs.append(ctx_local)
    
        # Classical heads (cheap) — we still compute only local query rows for correct TP behavior
        if self.quantum_heads < self.num_heads:
            for h in range(self.quantum_heads, self.num_heads):
                qh_local = q[:, h, q_start:q_end, :]
                kh_full  = k[:, h, :, :]
                vh_full  = v[:, h, :, :]
    
                scores_local = qh_local @ kh_full.transpose(1, 2)  # (B,Tq,T)
                scores_local = scores_local.masked_fill(mask_local, -1e9)
    
                attn_weights = torch.softmax(scores_local / math.sqrt(self.head_dim), dim=-1)
                attn_weights = self.dropout(attn_weights)
    
                ctx_local = attn_weights @ vh_full
                head_outs.append(ctx_local)
    
        # concat heads: (B, Tq, C)
        context_local = torch.cat(head_outs, dim=-1)
    
        # output projection is per-token; do it locally then gather (less communication)
        context_local = self.out_proj(context_local)
    
        # gather token blocks back to full sequence if tensor-parallel
        if self.tensor_parallel and _dist_is_initialized():
            context_full = AllGatherTokens.apply(context_local, self.tp_group)  # (B, T, C)
            return context_full
    
        return context_local


#####################################
# Quantum Attention (Custom)
#####################################
# This section adds a quantum-computed attention *score* matrix (Q,K -> scores)
# while keeping the rest of the Transformer block fully classical.
#
# How it works (high level):
#   - Classical linear layers produce Q, K, V
#   - For each head, we map Q and K vectors to rotation angles
#   - A quantum kernel computes scores_ij = |<psi(Q_i,pos_i;Wq) | psi(K_j,pos_j;Wk)>|^2
#   - Classical softmax + masking produce attention weights
#   - Classical weighted sum mixes V
#
# NOTE: This is extremely compute-heavy for large context lengths. Start tiny.

import math
import numpy as np
from torch.autograd import Function


class QuantumFunction(Function):
    """
    Quantum attention kernel via |<psi_k(x,pos_k;Wk) | psi_q(y,pos_q;Wq)>|^2.

    The circuit is built in Qiskit and contracted using cuQuantum (CircuitToEinsum + Network).
    """

    _circuit_cache = {}   # {n_dim: {"query_pos": qc, "key_pos": qc}}
    _tensor_cache = {}    # {(n_dim, device_id): (exp, base_oper, idx_map)}

    # ---------- small helpers ----------
    @staticmethod
    def _rz(theta: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[torch.exp(-0.5j * theta), 0.0j],
             [0.0j, torch.exp(0.5j * theta)]],
            dtype=torch.complex64,
            device=theta.device
        )

    @staticmethod
    def _ry(theta: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        return torch.tensor(
            [[c, -s],
             [s,  c]],
            dtype=torch.complex64,
            device=theta.device
        )

    @staticmethod
    def _allclose(a: torch.Tensor, b: torch.Tensor, atol=1e-6, rtol=1e-6) -> bool:
        return torch.allclose(a, b, atol=atol, rtol=rtol)

    # ---------- circuits with position parameters ----------
    @staticmethod
    def make_bsp_query_pos(n_dim: int):
        if n_dim in QuantumFunction._circuit_cache and "query_pos" in QuantumFunction._circuit_cache[n_dim]:
            return QuantumFunction._circuit_cache[n_dim]["query_pos"]

        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector
        except ImportError as e:
            raise ImportError("Quantum attention requires qiskit to be installed.") from e

        p_data = ParameterVector("p_data", n_dim)
        p_pos  = ParameterVector("p_pos",  n_dim)
        p_Wq   = ParameterVector("p_W_query", n_dim)

        qc = QuantumCircuit(n_dim)
        qc.h(range(n_dim))

        for q in range(n_dim):
            qc.rz(p_data[q], q)
            qc.ry(p_data[q], q)

        for q in range(n_dim):
            qc.rz(p_pos[q], q)
            qc.ry(p_pos[q], q)

        for i in range(n_dim - 1):
            qc.cx(i, i + 1)

        for q in range(n_dim):
            qc.rz(p_data[q], q)
        for q in range(n_dim):
            qc.rz(p_pos[q], q)

        for q in range(n_dim):
            qc.rz(p_Wq[q], q)
            qc.ry(p_Wq[q], q)

        for i in range(n_dim - 1):
            qc.cx(i, i + 1)

        for q in range(n_dim):
            qc.rz(p_Wq[q], q)

        QuantumFunction._circuit_cache.setdefault(n_dim, {})["query_pos"] = qc
        return qc

    @staticmethod
    def make_bsp_key_pos(n_dim: int):
        if n_dim in QuantumFunction._circuit_cache and "key_pos" in QuantumFunction._circuit_cache[n_dim]:
            return QuantumFunction._circuit_cache[n_dim]["key_pos"]

        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector
        except ImportError as e:
            raise ImportError("Quantum attention requires qiskit to be installed.") from e

        p_data = ParameterVector("p_data", n_dim)
        p_pos  = ParameterVector("p_pos",  n_dim)
        p_Wk   = ParameterVector("p_W_key", n_dim)

        qc = QuantumCircuit(n_dim)
        qc.h(range(n_dim))

        for q in range(n_dim):
            qc.rz(p_data[q], q)
            qc.ry(p_data[q], q)

        for q in range(n_dim):
            qc.rz(p_pos[q], q)
            qc.ry(p_pos[q], q)

        for i in range(n_dim - 1):
            qc.cx(i, i + 1)

        for q in range(n_dim):
            qc.rz(p_data[q], q)
        for q in range(n_dim):
            qc.rz(p_pos[q], q)

        for q in range(n_dim):
            qc.rz(p_Wk[q], q)
            qc.ry(p_Wk[q], q)

        for i in range(n_dim - 1):
            qc.cx(i, i + 1)

        for q in range(n_dim):
            qc.rz(p_Wk[q], q)

        QuantumFunction._circuit_cache.setdefault(n_dim, {})["key_pos"] = qc
        return qc

    @staticmethod
    def build_qsvm_qc_pos(n_dim, qc_query, qc_key,
                         y_data, y_pos, x_data, x_pos,
                         W_query, W_key):
        try:
            from qiskit import QuantumCircuit
        except ImportError as e:
            raise ImportError("Quantum attention requires qiskit to be installed.") from e

        q_data_params = [p for p in qc_query.parameters if p.name.startswith("p_data")]
        q_pos_params  = [p for p in qc_query.parameters if p.name.startswith("p_pos")]
        q_w_params    = [p for p in qc_query.parameters if p.name.startswith("p_W_query")]

        k_data_params = [p for p in qc_key.parameters if p.name.startswith("p_data")]
        k_pos_params  = [p for p in qc_key.parameters if p.name.startswith("p_pos")]
        k_w_params    = [p for p in qc_key.parameters if p.name.startswith("p_W_key")]

        q_map = {}
        q_map.update(dict(zip(q_data_params, y_data)))
        q_map.update(dict(zip(q_pos_params,  y_pos)))
        q_map.update(dict(zip(q_w_params,    W_query)))

        k_map = {}
        k_map.update(dict(zip(k_data_params, x_data)))
        k_map.update(dict(zip(k_pos_params,  x_pos)))
        k_map.update(dict(zip(k_w_params,    W_key)))

        assigned_query = qc_query.assign_parameters(q_map)
        assigned_key_inv = qc_key.assign_parameters(k_map).inverse()

        kernel_qc = QuantumCircuit(n_dim)
        kernel_qc.append(assigned_query, range(n_dim))
        kernel_qc.append(assigned_key_inv, range(n_dim))
        return kernel_qc

    @staticmethod
    def convert_to_tensor(circuit, n_dim: int):
        try:
            from cuquantum import CircuitToEinsum
        except ImportError as e:
            raise ImportError("Quantum attention requires cuQuantum (cuquantum) to be installed.") from e

        converter = CircuitToEinsum(circuit, dtype="complex64", backend="torch")
        a = str(0).zfill(n_dim)
        exp, oper = converter.amplitude(a)
        return exp, oper

    @staticmethod
    def _build_operand_index_map(n_dim: int, exp, oper):
        # Move operands to CPU for matching
        oper_cpu = [t.detach().cpu() if torch.is_tensor(t) else t for t in oper]

        qc_q = QuantumFunction.make_bsp_query_pos(n_dim)
        qc_k = QuantumFunction.make_bsp_key_pos(n_dim)

        base = np.linspace(0.13, 1.37, n_dim)
        q_data = base + 0.11
        q_pos  = base + 0.77
        q_w    = base + 1.41

        k_data = base + 2.05
        k_pos  = base + 2.79
        k_w    = base + 3.33

        full = QuantumFunction.build_qsvm_qc_pos(
            n_dim, qc_q, qc_k,
            y_data=q_data, y_pos=q_pos,
            x_data=k_data, x_pos=k_pos,
            W_query=q_w, W_key=k_w,
        )

        exp2, oper2 = QuantumFunction.convert_to_tensor(full, n_dim)
        oper2_cpu = [t.detach().cpu() if torch.is_tensor(t) else t for t in oper2]

        def find_indices(mat_cpu: torch.Tensor):
            hits = []
            for i, o in enumerate(oper2_cpu):
                if torch.is_tensor(o) and o.shape == (2, 2) and QuantumFunction._allclose(o, mat_cpu):
                    hits.append(i)
            return hits

        EXPECT = {"rz": 2, "ry": 1}

        idx_map = {k: [[] for _ in range(n_dim)] for k in [
            "q_data_rz", "q_data_ry", "q_pos_rz", "q_pos_ry", "q_w_rz", "q_w_ry",
            "k_data_rz", "k_data_ry", "k_pos_rz", "k_pos_ry", "k_w_rz", "k_w_ry",
        ]}

        for q in range(n_dim):
            th = torch.tensor(q_data[q], dtype=torch.float32)
            idx_map["q_data_rz"][q] = find_indices(QuantumFunction._rz(th).cpu())
            idx_map["q_data_ry"][q] = find_indices(QuantumFunction._ry(th).cpu())

            th = torch.tensor(q_pos[q], dtype=torch.float32)
            idx_map["q_pos_rz"][q] = find_indices(QuantumFunction._rz(th).cpu())
            idx_map["q_pos_ry"][q] = find_indices(QuantumFunction._ry(th).cpu())

            th = torch.tensor(q_w[q], dtype=torch.float32)
            idx_map["q_w_rz"][q] = find_indices(QuantumFunction._rz(th).cpu())
            idx_map["q_w_ry"][q] = find_indices(QuantumFunction._ry(th).cpu())

            if len(idx_map["q_data_rz"][q]) != EXPECT["rz"] or len(idx_map["q_data_ry"][q]) != EXPECT["ry"]:
                raise RuntimeError(f"Query data gate match failed at qubit {q}")
            if len(idx_map["q_pos_rz"][q]) != EXPECT["rz"] or len(idx_map["q_pos_ry"][q]) != EXPECT["ry"]:
                raise RuntimeError(f"Query pos gate match failed at qubit {q}")
            if len(idx_map["q_w_rz"][q]) != EXPECT["rz"] or len(idx_map["q_w_ry"][q]) != EXPECT["ry"]:
                raise RuntimeError(f"Query W gate match failed at qubit {q}")

        for q in range(n_dim):
            th = torch.tensor(-k_data[q], dtype=torch.float32)
            idx_map["k_data_rz"][q] = find_indices(QuantumFunction._rz(th).cpu())
            idx_map["k_data_ry"][q] = find_indices(QuantumFunction._ry(th).cpu())

            th = torch.tensor(-k_pos[q], dtype=torch.float32)
            idx_map["k_pos_rz"][q] = find_indices(QuantumFunction._rz(th).cpu())
            idx_map["k_pos_ry"][q] = find_indices(QuantumFunction._ry(th).cpu())

            th = torch.tensor(-k_w[q], dtype=torch.float32)
            idx_map["k_w_rz"][q] = find_indices(QuantumFunction._rz(th).cpu())
            idx_map["k_w_ry"][q] = find_indices(QuantumFunction._ry(th).cpu())

            if len(idx_map["k_data_rz"][q]) != EXPECT["rz"] or len(idx_map["k_data_ry"][q]) != EXPECT["ry"]:
                raise RuntimeError(f"Key data gate match failed at qubit {q}")
            if len(idx_map["k_pos_rz"][q]) != EXPECT["rz"] or len(idx_map["k_pos_ry"][q]) != EXPECT["ry"]:
                raise RuntimeError(f"Key pos gate match failed at qubit {q}")
            if len(idx_map["k_w_rz"][q]) != EXPECT["rz"] or len(idx_map["k_w_ry"][q]) != EXPECT["ry"]:
                raise RuntimeError(f"Key W gate match failed at qubit {q}")

        return exp2, oper2, idx_map

    @staticmethod
    def _apply_weights_inplace(op_list, idx_map, W_query: torch.Tensor, W_key: torch.Tensor):
        for q in range(W_query.shape[0]):
            th_q = W_query[q]
            rz_q = QuantumFunction._rz(th_q)
            ry_q = QuantumFunction._ry(th_q)
            for idx in idx_map["q_w_rz"][q]:
                op_list[idx] = rz_q
            for idx in idx_map["q_w_ry"][q]:
                op_list[idx] = ry_q

            th_k = -W_key[q]  # key side is inverted in the kernel circuit
            rz_k = QuantumFunction._rz(th_k)
            ry_k = QuantumFunction._ry(th_k)
            for idx in idx_map["k_w_rz"][q]:
                op_list[idx] = rz_k
            for idx in idx_map["k_w_ry"][q]:
                op_list[idx] = ry_k

    @staticmethod
    def _apply_data_pos_inplace(op_list, idx_map,
                               y_data: torch.Tensor, x_data: torch.Tensor,
                               y_pos: torch.Tensor,  x_pos: torch.Tensor):
        n_dim = y_data.shape[0]

        for q in range(n_dim):
            th = y_data[q]
            rz = QuantumFunction._rz(th)
            ry = QuantumFunction._ry(th)
            for idx in idx_map["q_data_rz"][q]:
                op_list[idx] = rz
            for idx in idx_map["q_data_ry"][q]:
                op_list[idx] = ry

            th = y_pos[q]
            rz = QuantumFunction._rz(th)
            ry = QuantumFunction._ry(th)
            for idx in idx_map["q_pos_rz"][q]:
                op_list[idx] = rz
            for idx in idx_map["q_pos_ry"][q]:
                op_list[idx] = ry

        for q in range(n_dim):
            th = -x_data[q]  # inverted
            rz = QuantumFunction._rz(th)
            ry = QuantumFunction._ry(th)
            for idx in idx_map["k_data_rz"][q]:
                op_list[idx] = rz
            for idx in idx_map["k_data_ry"][q]:
                op_list[idx] = ry

            th = -x_pos[q]  # inverted
            rz = QuantumFunction._rz(th)
            ry = QuantumFunction._ry(th)
            for idx in idx_map["k_pos_rz"][q]:
                op_list[idx] = rz
            for idx in idx_map["k_pos_ry"][q]:
                op_list[idx] = ry

    @staticmethod
    def forward(ctx, data1_batch, data2_batch, pos1_batch, pos2_batch, W_query, W_key, shift):
        if not data1_batch.is_cuda:
            raise RuntimeError("Quantum attention (cuQuantum) currently requires CUDA tensors.")

        n_dim = W_query.shape[0]
        device_id = int(data1_batch.device.index)

        batch_size = data1_batch.shape[0]
        seq_len_q = data1_batch.shape[1]
        seq_len_k = data2_batch.shape[1]

        cache_key = (n_dim, device_id)

        if cache_key not in QuantumFunction._tensor_cache:
            qc_q = QuantumFunction.make_bsp_query_pos(n_dim)
            qc_k = QuantumFunction.make_bsp_key_pos(n_dim)

            dummy = np.zeros(n_dim, dtype=np.float64)
            dummy_wq = np.linspace(0.21, 1.21, n_dim)
            dummy_wk = np.linspace(1.77, 2.77, n_dim)

            full = QuantumFunction.build_qsvm_qc_pos(
                n_dim, qc_q, qc_k,
                y_data=dummy, y_pos=dummy,
                x_data=dummy, x_pos=dummy,
                W_query=dummy_wq, W_key=dummy_wk,
            )

            exp0, oper0 = QuantumFunction.convert_to_tensor(full, n_dim)
            exp_tag, oper_tag, idx_map = QuantumFunction._build_operand_index_map(n_dim, exp0, oper0)
            oper_tag = [t.to(data1_batch.device) if torch.is_tensor(t) else t for t in oper_tag]
            QuantumFunction._tensor_cache[cache_key] = (exp_tag, oper_tag, idx_map)

        exp, base_oper, idx_map = QuantumFunction._tensor_cache[cache_key]

        oper_weight = base_oper.copy()
        QuantumFunction._apply_weights_inplace(
            oper_weight, idx_map,
            W_query.to(data1_batch.device), W_key.to(data1_batch.device)
        )

        try:
            from cuquantum import Network, NetworkOptions
        except ImportError as e:
            raise ImportError("Quantum attention requires cuQuantum (cuquantum) to be installed.") from e

        options = NetworkOptions(blocking="auto", device_id=device_id)

        kernel_matrix = torch.zeros(
            (batch_size, seq_len_q, seq_len_k),
            dtype=torch.float32,
            device=data1_batch.device
        )

        with Network(exp, *oper_weight, options=options) as tn:
            tn.contract_path()

            for b in range(batch_size):
                data1_seq = data1_batch[b]   # (Lq, n_dim)
                data2_seq = data2_batch[b]   # (Lk, n_dim)
                pos1_seq  = pos1_batch[b]
                pos2_seq  = pos2_batch[b]

                for i1 in range(seq_len_q):
                    for i2 in range(seq_len_k):
                        op = oper_weight.copy()
                        QuantumFunction._apply_data_pos_inplace(
                            op, idx_map,
                            y_data=data1_seq[i1],
                            x_data=data2_seq[i2],
                            y_pos=pos1_seq[i1],
                            x_pos=pos2_seq[i2],
                        )
                        tn.reset_operands(*op)
                        amp = tn.contract()
                        kernel_matrix[b, i1, i2] = (amp.abs() ** 2)

            tn.free()

        ctx.save_for_backward(data1_batch, data2_batch, pos1_batch, pos2_batch, W_query, W_key)
        ctx.shift = shift
        return kernel_matrix

    @staticmethod
    def backward(ctx, grad_output):
        # SPSA gradients (expensive but hardware-friendly)
        data1, data2, pos1, pos2, W_query, W_key = ctx.saved_tensors
        shift = ctx.shift
        SPSA_ITERS = 4

        def spsa_grad_weights(param, other_param, pname):
            grad_est = torch.zeros_like(param)
            for _ in range(SPSA_ITERS):
                delta = (torch.randint(0, 2, size=param.shape, device=param.device).float() * 2 - 1)
                param_plus  = param + shift * delta
                param_minus = param - shift * delta

                if pname == "W_query":
                    out_plus  = QuantumFunction.apply(data1, data2, pos1, pos2, param_plus, other_param, shift)
                    out_minus = QuantumFunction.apply(data1, data2, pos1, pos2, param_minus, other_param, shift)
                else:
                    out_plus  = QuantumFunction.apply(data1, data2, pos1, pos2, other_param, param_plus, shift)
                    out_minus = QuantumFunction.apply(data1, data2, pos1, pos2, other_param, param_minus, shift)

                grad = (out_plus - out_minus) / (2.0 * shift)
                grad_est += (grad_output * grad).sum() * delta / SPSA_ITERS
            return grad_est

        def spsa_grad_tensor(tensor, name):
            grad_est = torch.zeros_like(tensor)
            for _ in range(SPSA_ITERS):
                delta = (torch.randint(0, 2, size=tensor.shape, device=tensor.device).float() * 2 - 1)
                t_plus  = tensor + shift * delta
                t_minus = tensor - shift * delta

                if name == "data1":
                    out_plus  = QuantumFunction.apply(t_plus, data2, pos1, pos2, W_query, W_key, shift)
                    out_minus = QuantumFunction.apply(t_minus, data2, pos1, pos2, W_query, W_key, shift)
                elif name == "data2":
                    out_plus  = QuantumFunction.apply(data1, t_plus, pos1, pos2, W_query, W_key, shift)
                    out_minus = QuantumFunction.apply(data1, t_minus, pos1, pos2, W_query, W_key, shift)
                elif name == "pos1":
                    out_plus  = QuantumFunction.apply(data1, data2, t_plus, pos2, W_query, W_key, shift)
                    out_minus = QuantumFunction.apply(data1, data2, t_minus, pos2, W_query, W_key, shift)
                else:  # "pos2"
                    out_plus  = QuantumFunction.apply(data1, data2, pos1, t_plus, W_query, W_key, shift)
                    out_minus = QuantumFunction.apply(data1, data2, pos1, t_minus, W_query, W_key, shift)

                grad = (out_plus - out_minus) / (2.0 * shift)
                grad_reduced = (grad_output * grad).sum(dim=-1, keepdim=True)
                grad_est += grad_reduced * delta / SPSA_ITERS

            return grad_est

        data1_grad = spsa_grad_tensor(data1, "data1") if ctx.needs_input_grad[0] else None
        data2_grad = spsa_grad_tensor(data2, "data2") if ctx.needs_input_grad[1] else None
        pos1_grad  = spsa_grad_tensor(pos1,  "pos1")  if ctx.needs_input_grad[2] else None
        pos2_grad  = spsa_grad_tensor(pos2,  "pos2")  if ctx.needs_input_grad[3] else None
        Wq_grad    = spsa_grad_weights(W_query, W_key, "W_query") if ctx.needs_input_grad[4] else None
        Wk_grad    = spsa_grad_weights(W_key, W_query, "W_key")   if ctx.needs_input_grad[5] else None

        return data1_grad, data2_grad, pos1_grad, pos2_grad, Wq_grad, Wk_grad, None


class QuantumAttention(nn.Module):
    """Returns quantum attention scores (B, Lq, Lk)."""

    def __init__(self, n_dim: int, use_quantum_pos: bool = True):
        super().__init__()
        self.n_dim = n_dim
        self.use_quantum_pos = use_quantum_pos

        # Trainable circuit weights
        self.W_query = nn.Parameter(torch.rand(n_dim) * 2 * math.pi)
        self.W_key   = nn.Parameter(torch.rand(n_dim) * 2 * math.pi)

        self.shift = torch.tensor(math.pi / 2)

    @staticmethod
    def sinusoidal_pos_angles(seq_len: int, n_dim: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (seq,1)
        i = torch.arange(n_dim, device=device, dtype=torch.float32).unsqueeze(0)     # (1,n_dim)
        angle_rates = 1.0 / (10000 ** (2 * torch.floor(i / 2) / n_dim))
        angles = pos * angle_rates
        pe = torch.zeros(seq_len, n_dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return math.pi * pe  # radians

    def forward(self, data1, data2, pos1=None, pos2=None):
        """
        data1,data2: (B, L, n_dim) angles (float32)
        pos1,pos2: optional (B, L, n_dim) absolute-position angles.
        """
        b, s1, _ = data1.shape
        _, s2, _ = data2.shape
    
        if pos1 is None or pos2 is None:
            if self.use_quantum_pos:
                pos1_ = self.sinusoidal_pos_angles(s1, self.n_dim, data1.device).unsqueeze(0).expand(b, -1, -1)
                pos2_ = self.sinusoidal_pos_angles(s2, self.n_dim, data2.device).unsqueeze(0).expand(b, -1, -1)
            else:
                pos1_ = torch.zeros((b, s1, self.n_dim), device=data1.device, dtype=torch.float32)
                pos2_ = torch.zeros((b, s2, self.n_dim), device=data2.device, dtype=torch.float32)
        else:
            pos1_ = pos1
            pos2_ = pos2
    
        # Keep everything float32; QuantumFunction itself uses complex64 + float32
        return QuantumFunction.apply(
            data1.to(torch.float32), data2.to(torch.float32),
            pos1_.to(torch.float32), pos2_.to(torch.float32),
            self.W_query.to(torch.float32), self.W_key.to(torch.float32),
            self.shift.to(torch.float32),
        )


class QuantumMultiHeadAttention(nn.Module):
    """
    Drop-in replacement for MultiHeadAttention:

      - Q,K,V projections are classical
      - attention scores per head come from a quantum kernel running on `n_qubits`
      - masking, softmax, and V-mixing are classical

    Speed notes:
      - Quantum attention cost scales ~ O(B * heads * T^2) * (cost_per_contraction).
      - Keep context_length small for experiments, and consider using only a few quantum heads/layers.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False,
                 use_quantum_pos: bool = True,
                 angle_scale: float = math.pi,
                 n_qubits: int | None = None,
                 shared_angle_proj: bool = True,
                 softmax_temp: float | None = None,
                 quantum_heads: int | None = None,
                 tensor_parallel: bool = False,
                 tp_group=None):
        """
        Args:
            n_qubits: number of qubits used by the quantum kernel. If None, defaults to head_dim.
            shared_angle_proj: if True, one angle projection is shared across heads (faster + fewer params).
            softmax_temp: if provided, divides scores by this value before softmax. If None, uses sqrt(n_qubits).
            quantum_heads: if set, only the first `quantum_heads` heads use quantum scores; remaining heads use
                          classical dot-product scores (big speed win).
        """
        super().__init__()
        self.tensor_parallel = bool(tensor_parallel)
        self.tp_group = tp_group if tp_group is not None else TP_GROUP
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.n_qubits = int(n_qubits) if n_qubits is not None else int(self.head_dim)
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be >= 1")

        self.angle_scale = float(angle_scale)
        self.softmax_temp = float(softmax_temp) if softmax_temp is not None else math.sqrt(self.n_qubits)

        # how many heads use quantum
        self.quantum_heads = int(quantum_heads) if quantum_heads is not None else self.num_heads
        self.quantum_heads = max(0, min(self.quantum_heads, self.num_heads))

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # Project head_dim -> n_qubits to get quantum rotation angles.
        if shared_angle_proj:
            self.angle_proj_q = nn.Linear(self.head_dim, self.n_qubits, bias=True)
            self.angle_proj_k = nn.Linear(self.head_dim, self.n_qubits, bias=True)
            self._per_head_angle_proj = False
        else:
            self.angle_proj_q = nn.ModuleList([nn.Linear(self.head_dim, self.n_qubits, bias=True)
                                               for _ in range(num_heads)])
            self.angle_proj_k = nn.ModuleList([nn.Linear(self.head_dim, self.n_qubits, bias=True)
                                               for _ in range(num_heads)])
            self._per_head_angle_proj = True

        # Quantum scorers only for the heads that use quantum
        self.qattn = nn.ModuleList([
            QuantumAttention(self.n_qubits, use_quantum_pos=use_quantum_pos)
            for _ in range(self.quantum_heads)
        ])

    def _to_angles(self, x: torch.Tensor) -> torch.Tensor:
        return self.angle_scale * torch.tanh(x)

    def _proj_angles(self, x: torch.Tensor, h: int, which: str) -> torch.Tensor:
        if self._per_head_angle_proj:
            proj = self.angle_proj_q[h] if which == "q" else self.angle_proj_k[h]
            return proj(x)
        else:
            proj = self.angle_proj_q if which == "q" else self.angle_proj_k
            return proj(x)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # -> (B, heads, T, head_dim)
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        head_outs = []

        # Heads 0..quantum_heads-1 use quantum scores
        for h in range(self.quantum_heads):
            qh = q[:, h, :, :]  # (B,T,head_dim)
            kh = k[:, h, :, :]
            vh = v[:, h, :, :]

            q_proj = self._proj_angles(qh, h, "q")
            k_proj = self._proj_angles(kh, h, "k")

            q_ang = self._to_angles(q_proj).to(torch.float32)
            k_ang = self._to_angles(k_proj).to(torch.float32)

            scores = self.qattn[h](q_ang, k_ang).to(torch.float32)  # (B,T,T)

            scores = scores.masked_fill(mask_bool, -1e9)
            attn_weights = torch.softmax(scores / self.softmax_temp, dim=-1)
            attn_weights = self.dropout(attn_weights)

            ctx = attn_weights @ vh
            head_outs.append(ctx)

        # Remaining heads use classical dot-product scores
        if self.quantum_heads < self.num_heads:
            for h in range(self.quantum_heads, self.num_heads):
                qh = q[:, h, :, :]
                kh = k[:, h, :, :]
                vh = v[:, h, :, :]

                scores = qh @ kh.transpose(1, 2)
                scores = scores.masked_fill(mask_bool, -1e9)

                attn_weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
                attn_weights = self.dropout(attn_weights)
                ctx = attn_weights @ vh
                head_outs.append(ctx)

        context_vec = torch.cat(head_outs, dim=-1)
        context_vec = self.out_proj(context_vec)
        return context_vec

#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        attn_type = cfg.get("attn_type", "classical")
        if attn_type == "quantum":
            self.att = QuantumMultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"],
                use_quantum_pos=cfg.get("use_quantum_pos", True),
                angle_scale=cfg.get("quantum_angle_scale", math.pi),
                n_qubits=cfg.get("quantum_n_qubits", None),
                shared_angle_proj=cfg.get("quantum_shared_angle_proj", True),
                softmax_temp=cfg.get("quantum_softmax_temp", None),
                quantum_heads=cfg.get("quantum_heads", None),
                tensor_parallel=cfg.get("quantum_tensor_parallel", False),
            )
        else:
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
