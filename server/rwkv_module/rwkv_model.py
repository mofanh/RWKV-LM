import numpy as np
import types, torch, copy, time
from typing import List
import os
from torch.nn import functional as F
import torch.nn as nn

# 使用JIT编译
MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

# 设置默认数据类型
DTYPE = torch.half

# 从RWKV-v7目录复制CUDA扩展库
RWKV_PATH = os.environ.get("RWKV_PATH", "/root/megrez-tmp/RWKV-LM/RWKV-v7")
HEAD_SIZE = 64  # 默认值，会被替换

# 确保CUDA扩展已经编译好并可用
def ensure_cuda_extension():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建cuda文件夹
    cuda_dir = os.path.join(current_dir, "cuda")
    os.makedirs(cuda_dir, exist_ok=True)
    
    # 复制CUDA文件
    cuda_files = ["wkv7s_op.cpp", "wkv7s.cu", "wkv7_op.cpp", "wkv7.cu"]
    for file in cuda_files:
        src = os.path.join(RWKV_PATH, "cuda", file)
        dst = os.path.join(cuda_dir, file)
        if not os.path.exists(dst) and os.path.exists(src):
            import shutil
            print(f"复制CUDA文件: {src} -> {dst}")
            shutil.copy2(src, dst)
    
    # 尝试导入编译好的扩展
    try:
        # 检查是否已经有编译好的库
        if not hasattr(torch.ops, 'wkv7s'):
            from torch.utils.cpp_extension import load
            # 编译扩展
            load(
                name="wkv7s",
                sources=[
                    os.path.join(cuda_dir, "wkv7s_op.cpp"),
                    os.path.join(cuda_dir, "wkv7s.cu")
                ],
                is_python_module=False,
                verbose=True,
                extra_cuda_cflags=[
                    "-res-usage",
                    "--use_fast_math",
                    "-O3",
                    "-Xptxas -O3",
                    "--extra-device-vectorization",
                    f"-D_N_={HEAD_SIZE}",
                ],
            )
        return True
    except Exception as e:
        print(f"CUDA扩展编译失败: {e}")
        return False

class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r, w, k, v, a, b])
            assert all(x.is_contiguous() for x in [r, w, k, v, a, b])
            y = torch.empty(
                (T, C),
                device=k.device,
                dtype=DTYPE,
                requires_grad=False,
                memory_format=torch.contiguous_format,
            )
            torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y

def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7.apply(state, r, w, k, v, a, b)

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        global HEAD_SIZE
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        HEAD_SIZE = args.head_size  # 更新全局HEAD_SIZE
        
        # 确保CUDA扩展可用
        ensure_cuda_extension()
        
        self.eval()
        
        # 加载模型权重
        self.z = torch.load(args.MODEL_NAME + ".pth", map_location="cuda")
        z = self.z
        self.n_head, self.head_size = z["blocks.0.att.r_k"].shape

        keys = list(z.keys())
        for k in keys:
            if (
                "key.weight" in k
                or "value.weight" in k
                or "receptance.weight" in k
                or "output.weight" in k
                or "head.weight" in k
            ):
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith("att.r_k"):
                z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        z["emb.weight"] = F.layer_norm(
            z["emb.weight"],
            (args.n_embd,),
            weight=z["blocks.0.ln0.weight"],
            bias=z["blocks.0.ln0.bias"],
        )
        z["blocks.0.att.v0"] = z["blocks.0.att.a0"]  # actually ignored
        z["blocks.0.att.v1"] = z["blocks.0.att.a1"]  # actually ignored
        z["blocks.0.att.v2"] = z["blocks.0.att.a2"]  # actually ignored

    def forward(self, idx, state, full_output=False):
        if state == None:
            state = [None for _ in range(self.args.n_layer * 3)]
            for i in range(self.args.n_layer):  # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i * 3 + 0] = torch.zeros(
                    self.args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda"
                )
                state[i * 3 + 1] = torch.zeros(
                    (self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size),
                    dtype=torch.float,
                    requires_grad=False,
                    device="cuda",
                )
                state[i * 3 + 2] = torch.zeros(
                    self.args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda"
                )

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    @MyFunction
    def forward_one(self, idx: int, state: List[torch.Tensor]):
        with torch.no_grad():
            z = self.z
            x = z["emb.weight"][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f"blocks.{i}."
                att = f"blocks.{i}.att."
                ffn = f"blocks.{i}.ffn."

                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln1.weight"],
                    bias=z[bbb + "ln1.bias"],
                )

                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_one(
                    i,
                    self.n_head,
                    self.head_size,
                    xx,
                    state[i * 3 + 0],
                    v_first,
                    state[i * 3 + 1],
                    z[att + "x_r"],
                    z[att + "x_w"],
                    z[att + "x_k"],
                    z[att + "x_v"],
                    z[att + "x_a"],
                    z[att + "x_g"],
                    z[att + "w0"],
                    z[att + "w1"],
                    z[att + "w2"],
                    z[att + "a0"],
                    z[att + "a1"],
                    z[att + "a2"],
                    z[att + "v0"],
                    z[att + "v1"],
                    z[att + "v2"],
                    z[att + "g1"],
                    z[att + "g2"],
                    z[att + "k_k"],
                    z[att + "k_a"],
                    z[att + "r_k"],
                    z[att + "receptance.weight"],
                    z[att + "key.weight"],
                    z[att + "value.weight"],
                    z[att + "output.weight"],
                    z[att + "ln_x.weight"],
                    z[att + "ln_x.bias"],
                )
                x = x + xx

                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln2.weight"],
                    bias=z[bbb + "ln2.bias"],
                )

                xx, state[i * 3 + 2] = RWKV_x070_CMix_one(
                    xx,
                    state[i * 3 + 2],
                    z[ffn + "x_k"],
                    z[ffn + "key.weight"],
                    z[ffn + "value.weight"],
                )
                x = x + xx

            x = F.layer_norm(
                x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]
            )
            x = x @ z["head.weight"]
            return x, state

    @MyFunction
    def forward_seq(
        self, idx: List[int], state: List[torch.Tensor], full_output: bool = False
    ):
        with torch.no_grad():
            z = self.z
            x = z["emb.weight"][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f"blocks.{i}."
                att = f"blocks.{i}.att."
                ffn = f"blocks.{i}.ffn."

                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln1.weight"],
                    bias=z[bbb + "ln1.bias"],
                )

                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_seq(
                    i,
                    self.n_head,
                    self.head_size,
                    xx,
                    state[i * 3 + 0],
                    v_first,
                    state[i * 3 + 1],
                    z[att + "x_r"],
                    z[att + "x_w"],
                    z[att + "x_k"],
                    z[att + "x_v"],
                    z[att + "x_a"],
                    z[att + "x_g"],
                    z[att + "w0"],
                    z[att + "w1"],
                    z[att + "w2"],
                    z[att + "a0"],
                    z[att + "a1"],
                    z[att + "a2"],
                    z[att + "v0"],
                    z[att + "v1"],
                    z[att + "v2"],
                    z[att + "g1"],
                    z[att + "g2"],
                    z[att + "k_k"],
                    z[att + "k_a"],
                    z[att + "r_k"],
                    z[att + "receptance.weight"],
                    z[att + "key.weight"],
                    z[att + "value.weight"],
                    z[att + "output.weight"],
                    z[att + "ln_x.weight"],
                    z[att + "ln_x.bias"],
                )
                x = x + xx

                xx = F.layer_norm(
                    x,
                    (self.n_embd,),
                    weight=z[bbb + "ln2.weight"],
                    bias=z[bbb + "ln2.bias"],
                )

                xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(
                    xx,
                    state[i * 3 + 2],
                    z[ffn + "x_k"],
                    z[ffn + "key.weight"],
                    z[ffn + "value.weight"],
                )
                x = x + xx

            if not full_output:
                x = x[-1, :]
            x = F.layer_norm(
                x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]
            )
            x = x @ z["head.weight"]
            return x, state

@MyStatic
def RWKV_x070_TMix_one(
    layer_id: int,
    H: int,
    N: int,
    x,
    x_prev,
    v_first,
    state,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    w0,
    w1,
    w2,
    a0,
    a1,
    a2,
    v0,
    v1,
    v2,
    g1,
    g2,
    k_k,
    k_a,
    r_k,
    R_,
    K_,
    V_,
    O_,
    ln_w,
    ln_b,
):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = (
        x + xx * x_r,
        x + xx * x_w,
        x + xx * x_k,
        x + xx * x_v,
        x + xx * x_a,
        x + xx * x_g,
    )

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float()))  # 0.606531 = exp(-0.5)

    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    xx = state.to(dtype=x.dtype) @ r.view(H, N, 1)

    xx = torch.nn.functional.group_norm(
        xx.view(1, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5
    ).view(H * N)
    xx = xx + ((r * k * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(
        H * N
    )
    return (xx * g) @ O_, x, state, v_first


@MyStatic
def RWKV_x070_TMix_seq(
    layer_id: int,
    H: int,
    N: int,
    x,
    x_prev,
    v_first,
    state,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    w0,
    w1,
    w2,
    a0,
    a1,
    a2,
    v0,
    v1,
    v2,
    g1,
    g2,
    k_k,
    k_a,
    r_k,
    R_,
    K_,
    V_,
    O_,
    ln_w,
    ln_b,
):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
    xr, xw, xk, xv, xa, xg = (
        x + xx * x_r,
        x + xx * x_w,
        x + xx * x_k,
        x + xx * x_v,
        x + xx * x_a,
        x + xx * x_g,
    )

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T, H, N), dim=-1, p=2.0).view(
        T, H * N
    )
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    # 直接使用循环来替换 RWKV7_OP
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float()))  # 0.606531 = exp(-0.5)
    xx = torch.empty_like(x)
    for t in range(T):
        r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
        vk = v_.view(H,N,1) @ k_.view(H,1,N)
        ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
        state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
        xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    xx = torch.nn.functional.group_norm(
        xx.view(T, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5
    ).view(T, H * N)
    xx = xx + (
        (r * k * r_k).view(T, H, N).sum(dim=-1, keepdim=True) * v.view(T, H, N)
    ).view(T, H * N)
    return (xx * g) @ O_, x[-1, :], state, v_first

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1, :]

@MyStatic
def sample_logits(logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)

    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

class RWKV_TOKENIZER:
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]

    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []  # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted_tokens += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(
            range(len(sorted_tokens))
        ):  # reverse order - match longer tokens first
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode("utf-8")

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()