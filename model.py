"""
算法岗必刷题单

Assembled from your step-by-step solutions.
"""

# Step 1 - __init__ (not yet solved)
# TODO: implement

# Step 2 - __init__
import torch
from typing import Tuple

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        # 防止exp溢出，标准sigmoid
        return 1.0 / (1.0 + torch.exp(-z))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        n, d = X.shape
        # fit内部初始化weights全0，bias=0
        self.weights = torch.zeros(d, dtype=X.dtype, device=X.device)
        self.bias = torch.tensor(0.0, dtype=X.dtype, device=X.device)

        for _ in range(self.epochs):
            # z = Xw + b
            z = torch.matmul(X, self.weights) + self.bias
            y_hat = self._sigmoid(z)

            # 梯度，严格按题面公式，系数1/n
            grad_w = (1.0 / n) * torch.matmul(X.T, (y_hat - y))
            grad_b = (1.0 / n) * torch.sum(y_hat - y)

            # 批量梯度下降更新
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        proba = self.predict_proba(X)
        return (proba >= threshold).to(torch.float32)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_pred = self.predict(X)
        # tn, fp, fn, tp
        tn = torch.sum((y == 0) & (y_pred == 0))
        fp = torch.sum((y == 0) & (y_pred == 1))
        fn = torch.sum((y == 1) & (y_pred == 0))
        tp = torch.sum((y == 1) & (y_pred == 1))

        conf_matrix = torch.tensor([[tn, fp], [fn, tp]], dtype=torch.float32)
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total
        return accuracy, conf_matrix

# Step 3 - my_kmeans
def my_kmeans(data, num_clusters, init_centers, max_iters):
    points = [list(p) for p in data]
    centroids = [list(c) for c in init_centers]
    dim = len(points[0]) if points else 0
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(num_clusters)]
        for p in points:
            best = 0
            best_dist = float('inf')
            for i, c in enumerate(centroids):
                # 使用平方距离避免开方
                dist = sum((p[d] - c[d])**2 for d in range(dim))
                if dist < best_dist:
                    best_dist = dist
                    best = i
            clusters[best].append(p)
        
        new_centroids = []
        for i in range(num_clusters):
            if clusters[i]:
                mean = [sum(p[d] for p in clusters[i]) / len(clusters[i]) for d in range(dim)]
                new_centroids.append(mean)
            else:
                new_centroids.append(centroids[i][:])
        
        # 收敛检查
        converged = True
        for i in range(num_clusters):
            for d in range(dim):
                if abs(new_centroids[i][d] - centroids[i][d]) > 1e-12:
                    converged = False
                    break
            if not converged:
                break
        centroids = new_centroids
        if converged:
            break
    
  
    # 如果维度不固定，可以通用：result = [tuple(round(coord, 4) for coord in c) for c in centroids]
    return np.round(np.asarray(centroids, dtype=float), 4)

# Step 4 - pca
import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(data, dtype=float)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    C = np.cov(Z, rowvar=False, ddof=0)
    w, v = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:k]
    V = v[:, idx].copy()
    for j in range(V.shape[1]):
        for t in V[:, j]:
            if abs(t) > 1e-10:
                if t < 0:
                    V[:, j] *= -1
                break
    return np.round(V, 4)

# Step 5 - K_Nearest_Neighbors
def K_Nearest_Neighbors(
    points: list[tuple[float, ...]],
    query_point: tuple[float, ...],
    k: int
) -> list[tuple[float, ...]]:
  
    def _as_tuple(p):
        if isinstance(p, (int, float)):
            return (p,)
        return tuple(p)

    q = _as_tuple(query_point)

    # 计算每个点的平方欧氏距离 + 原始索引
    indexed = []
    for idx, point in enumerate(points):
        p = _as_tuple(point)
        dist_sq = sum((pi - qi) ** 2 for pi, qi in zip(p, q))
        indexed.append((dist_sq, idx, point))

    # 排序：距离升序；同距离按原索引升序，保留先后顺序
    indexed.sort(key=lambda x: (x[0], x[1]))

    # 取前 k 个点返回
    return [item[2] for item in indexed[:k]]

# Step 6 - auc_rank
import numpy as np

def auc_rank(y_scores: np.ndarray, y_true: np.ndarray) -> float:
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    # 边界处理：无正样本或无负样本时直接返回0
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # 按预测分数升序排序，得到对应索引
    sorted_indices = np.argsort(y_scores)
    sorted_scores = y_scores[sorted_indices]
    
    # 定位同分样本的分组边界
    diff = np.diff(sorted_scores)
    change_points = np.where(diff != 0)[0] + 1
    boundaries = np.concatenate([[0], change_points, [len(sorted_scores)]])
    
    # 初始化秩数组，存储每个样本的秩（同分取平均秩）
    ranks = np.empty_like(y_scores, dtype=np.float64)
    
    # 为每个同分组分配平均秩（秩从1开始计数）
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        avg_rank = (start + end + 1) / 2.0
        ranks[sorted_indices[start:end]] = avg_rank
    
    # 计算所有正样本的秩和
    rank_sum = np.sum(ranks[y_true == 1])
    
    # 代入 Mann-Whitney U 公式计算 AUC
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc

# Step 7 - sigmoid (not yet solved)
# TODO: implement

# Step 8 - relu (not yet solved)
# TODO: implement

# Step 9 - 交叉熵损失 (not yet solved)
# TODO: implement

# Step 10 - single_neuron_model (not yet solved)
# TODO: implement

# Step 11 - adam_optimizer (not yet solved)
# TODO: implement

# Step 12 - AdamW 优化器 (not yet solved)
# TODO: implement

# Step 13 - residual_block (not yet solved)
# TODO: implement

# Step 14 - __init__ (not yet solved)
# TODO: implement

# Step 15 - example_kernel (not yet solved)
# TODO: implement

# Step 16 - __init__ (not yet solved)
# TODO: implement

# Step 17 - __init__ (not yet solved)
# TODO: implement

# Step 18 - __init__ (not yet solved)
# TODO: implement

# Step 19 - kl_divergence (not yet solved)
# TODO: implement

# Step 20 - focal_loss
import torch

def focal_loss(
    probs: torch.Tensor, targets: torch.Tensor, gamma: float, alpha: float
) -> float:
    # pt: y=1取probs; y=0取1‑probs
    pt = torch.where(targets == 1, probs, 1.0 - probs)
    # α_t:正类α，负类1‑α
    alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)

    fl = - alpha_t * torch.pow((1.0 - pt), gamma) * torch.log(pt)
    mean_fl = torch.mean(fl)
    # 返回平均损失，保留4位小数，转python float
    return round(float(mean_fl), 4)

# Step 21 - triplet_loss
import numpy as np

def triplet_loss(
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    margin: float,
) -> float:
    dist_ap = np.linalg.norm(anchor - positive, ord=2)
    dist_an = np.linalg.norm(anchor - negative, ord=2)
    loss_val = dist_ap - dist_an + margin
    loss = max(0.0, loss_val)
    return float(loss)

# Step 22 - hinge_loss
import numpy as np

def hinge_loss(scores: np.ndarray, y: np.ndarray) -> float:
    term = 1 - y * scores
    loss_each = np.maximum(0, term)
    avg_loss = np.mean(loss_each)
    return float(avg_loss)

# Step 23 - InfoNCE Loss (not yet solved)
# TODO: implement

# Step 24 - bpr_loss
import numpy as np

def bpr_loss(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    diff = pos_scores - neg_scores
    sigma = 1 / (1 + np.exp(-diff))
    log_sigma = np.log(sigma)
    loss = -np.mean(log_sigma)
    return round(loss, 4)

# Step 25 - lambda_loss (not yet solved)
# TODO: implement

# Step 26 - dropout_forward (not yet solved)
# TODO: implement

# Step 27 - __init__
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))

    def _norm(self, x):
        mean_square = x.float().pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_square + self.eps)
        return x.float() * rsqrt

    def forward(self, x):
        normed_x = self._norm(x)
        return normed_x.type_as(x) * self.gamma

# Step 28 - mlp_forward (not yet solved)
# TODO: implement

# Step 29 - masked_attention
import torch

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = K.size(-1)
    logits = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    logits = logits + mask
    attn_weight = torch.softmax(logits, dim=-1)
    output = torch.matmul(attn_weight, V)
    return torch.round(output * 10) / 10

# Step 30 - __init__
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        # Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q/k/v: [B, H, S, D]；返回 (output, attn_weights)
        B, H, S, D = q.shape
        # QK^T / sqrt(d_k)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

        # mask中0表示屏蔽，置为‑inf
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -torch.inf)

        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# Step 31 - sliding_window_self_attention
import torch
import math
from typing import Union

def sliding_window_self_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    # Q,K,V: (..., L, D)；每个位置 i 只看 |i-j| <= window_size
    # scores = (Q @ K^T) / sqrt(D)，窗外置 -inf，再 Softmax @ V
    Q = Q.to(dtype=torch.float64)
    K = K.to(dtype=torch.float64)
    V = V.to(dtype=torch.float64)
    L, D = Q.shape[-2], Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
    idx = torch.arange(L, device=Q.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > int(window_size)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return attn @ V

# Step 32 - __init__ (not yet solved)
# TODO: implement

# Step 33 - pos_encoding
import numpy as np
from typing import Union

def pos_encoding(position: int, d_model: int) -> Union[np.ndarray, int]:
    if position == 0 or d_model <= 0:
        return -1

    d = d_model
    pe = np.zeros((position, d), dtype=np.float16)
    pos = np.arange(position)[:, np.newaxis]
    # i：维对下标 0,1,...,d/2‑1；数组存储2*i
    pair_i = np.arange(0, d, 2)
    denom = np.power(10000.0, pair_i / d)

    pe[:, 0::2] = np.sin(pos / denom)
    pe[:, 1::2] = np.cos(pos / denom)
    return pe

# Step 34 - ffn_token_kernel
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def ffn_token_kernel(
    x_ptr, wu_ptr, bu_ptr, wd_ptr, bd_ptr, y_ptr,
    D, H,
    stride_xn, stride_xd,
    stride_wu_h, stride_wu_d,
    stride_wd_d, stride_wd_h,
    stride_yn, stride_yd,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_H)
    mask_d = offs_d < D
    mask_h = offs_h < H
    x = tl.load(x_ptr + pid * stride_xn + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)
    wu = tl.load(
        wu_ptr + offs_h[:, None] * stride_wu_h + offs_d[None, :] * stride_wu_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    bu = tl.load(bu_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    hid = tl.sum(wu * x[None, :], axis=1) + bu
    hid = tl.maximum(hid, 0.0)
    wd = tl.load(
        wd_ptr + offs_d[:, None] * stride_wd_d + offs_h[None, :] * stride_wd_h,
        mask=mask_d[:, None] & mask_h[None, :],
        other=0.0,
    ).to(tl.float32)
    bd = tl.load(bd_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    y = tl.sum(wd * hid[None, :], axis=1) + bd
    tl.store(y_ptr + pid * stride_yn + offs_d * stride_yd, y, mask=mask_d)


class FFN(nn.Module):
    def __init__(self, model_dim: int, intermediate_dim: int):
        super().__init__()
        self.w_up = nn.Linear(model_dim, intermediate_dim)
        self.w_down = nn.Linear(intermediate_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_device = x.device
        orig_dtype = x.dtype
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
            self.cuda()
        w_up = self.w_up.weight.contiguous()
        b_up = self.w_up.bias.contiguous()
        w_down = self.w_down.weight.contiguous()
        b_down = self.w_down.bias.contiguous()
        B, S, D = x.shape
        H = int(self.w_up.out_features)
        n = B * S
        x2 = x.view(n, D)
        y2 = torch.empty((n, D), device=x.device, dtype=torch.float32)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_H = triton.next_power_of_2(H)
        ffn_token_kernel[(n,)](
            x2, w_up, b_up, w_down, b_down, y2,
            D, H,
            x2.stride(0), x2.stride(1),
            w_up.stride(0), w_up.stride(1),
            w_down.stride(0), w_down.stride(1),
            y2.stride(0), y2.stride(1),
            BLOCK_D=BLOCK_D,
            BLOCK_H=BLOCK_H,
        )
        return y2.view(B, S, D).to(device=orig_device, dtype=orig_dtype)

# Step 35 - kv_cache_attention
import torch

def kv_cache_attention(
    Q: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    start_pos: int,
    new_k: torch.Tensor,
    new_v: torch.Tensor
) -> torch.Tensor:
    # 1. 将新KV写入对应缓存位置
    length = new_k.shape[0]
    cache_k[start_pos: start_pos + length] = new_k
    cache_v[start_pos: start_pos + length] = new_v

    # 2. 取出全部有效key、value
    K = cache_k[: start_pos + length]
    V = cache_v[: start_pos + length]

    # 3. 缩放点积注意力计算
    d = Q.shape[-1]
    attn_score = Q @ K.T / torch.sqrt(torch.tensor(d, dtype=torch.float32))
    attn_weight = torch.softmax(attn_score, dim=-1)
    attn_out = attn_weight @ V

    # 4. 保留4位小数
    attn_out = torch.round(attn_out, decimals=4)
    return attn_out

# Step 36 - lora_forward (not yet solved)
# TODO: implement

# Step 37 - __init__ (not yet solved)
# TODO: implement

# Step 38 - online_softmax_attention (not yet solved)
# TODO: implement

# Step 39 - greedy_decode
import numpy as np

def greedy_decode(logits_steps, eos_id=None):
    out = []
    for logits in logits_steps:
        z = np.asarray(logits, dtype=float)
        tid = int(np.argmax(z))
        out.append(tid)
        if eos_id is not None and tid == int(eos_id):
            break
    return out

# Step 40 - beam_search_decode
import numpy as np

def beam_search_decode(logits_steps, beam_width, length_penalty=1.0, eos_id=None):
    beams = [([], 0.0)]
    alpha = float(length_penalty)
    k = int(beam_width)

    def score(item):
        seq, sum_lp = item
        L = max(len(seq), 1)
        return sum_lp / (L ** alpha)

    for logits in logits_steps:
        z = np.asarray(logits, dtype=float)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        lp = np.log(p + 1e-12)
        cands = []
        for seq, sum_lp in beams:
            if eos_id is not None and len(seq) > 0 and seq[-1] == int(eos_id):
                cands.append((seq, sum_lp))
                continue
            for i, v in enumerate(lp):
                cands.append((seq + [int(i)], sum_lp + float(v)))
        cands.sort(key=lambda it: (-score(it), it[0]))
        beams = cands[:k]
    beams.sort(key=lambda it: (-score(it), it[0]))
    return beams[0][0]

# Step 41 - 温度采样 (not yet solved)
# TODO: implement

# Step 42 - Top-K 采样 (not yet solved)
# TODO: implement

# Step 43 - top_p_sample
import torch

def top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    x = logits.to(dtype=torch.float64)
    z = x - x.max()
    p = torch.exp(z)
    p = p / p.sum()

    order = torch.argsort(-p)
    cum = 0.0
    keep = []
    for i in order.tolist():
        keep.append(int(i))
        cum += float(p[i])
        if cum >= float(top_p):
            break

    masked = torch.full_like(x, float('-inf'))
    for i in keep:
        masked[i] = x[i]

    m = masked.max()
    e = torch.exp(masked - m)
    e = torch.where(torch.isfinite(e), e, torch.zeros_like(e))
    return torch.round(e / e.sum(), decimals=4).tolist()

# Step 44 - ppo_clip_ratio
import numpy as np

def ppo_clip_ratio(ratio, clip_eps):
    lower = 1 - clip_eps
    upper = 1 + clip_eps
    clipped = np.clip(ratio, lower, upper)
    return round(clipped, 4)

# Step 45 - dpo_loss
import numpy as np
import math

def dpo_loss(pi_logratio: np.ndarray, ref_logratio: np.ndarray, beta: float) -> np.ndarray:
    x = beta * (pi_logratio - ref_logratio)
    sig = 1 / (1 + np.exp(-x))
    loss = -np.log(sig)
    return np.round(loss, 4)

# Step 46 - compute_efficiency
import torch

def compute_efficiency(n_experts, k_active, d_in, d_out):
    # 返回相对稠密激活的计算节省比例（%）；保留 1 位小数
    return round((1 - k_active / n_experts) * 100, 1)

# Step 47 - bpe_merge (not yet solved)
# TODO: implement

# Step 48 - paged_attention_blocks (not yet solved)
# TODO: implement

# Step 49 - allreduce_mean
import torch

def allreduce_mean(grads: torch.Tensor) -> torch.Tensor:
    # 沿第0维求平均，N个卡的梯度做all‑reduce均值
    mean_val = torch.mean(grads, dim=0)
    # 保留4位小数
    result = torch.round(mean_val * 10000) / 10000
    return result

# Step 50 - huber_loss
import math

def huber_loss(y_true, y_pred, delta):
    e = abs(y_true - y_pred)
    if e <= delta:
        loss = 0.5 * (e ** 2)
    else:
        loss = delta * (e - 0.5 * delta)
    return float(loss)

# Step 51 - precompute_freqs_cis
import torch

def precompute_freqs_cis(head_dim, seq_len, theta_base=10000.0):
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs_cis), torch.sin(freqs_cis)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(xq, xk, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out

# Step 52 - __init__
import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_q, x_kv):
        Q = self.W_q(x_q)
        K = self.W_k(x_kv)
        V = self.W_v(x_kv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

# Step 53 - __init__
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.W_o(out)

# Step 54 - __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, mask=None, use_cache=False):
        B, L, D = x.shape
        Q = self.Wq(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        if use_cache:
            if self.cache_k is not None:
                self.cache_k = torch.cat([self.cache_k, K], dim=-2)
                self.cache_v = torch.cat([self.cache_v, V], dim=-2)
            else:
                self.cache_k = K
                self.cache_v = V
            K = self.cache_k
            V = self.cache_v
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.Wo(out)

# Step 55 - __init__
import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiQueryAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, self.head_dim)
        self.W_v = nn.Linear(embed_dim, self.head_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.W_o(out)

# Step 56 - mla_attention
import torch
import math

def mla_attention(q_c, q_r, k_c, k_r, v_c):
    Q = torch.cat([q_c, q_r], dim=-1)
    k_r_expanded = k_r.expand(-1, q_c.size(1), -1, -1)
    K = torch.cat([k_c, k_r_expanded], dim=-1)
    scale = math.sqrt(q_c.size(-1) + q_r.size(-1))
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_c)

# Step 57 - __init__
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """题目给定，勿改。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        Q = self.W_q(q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """题目给定，勿改。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.dropout(self.self_attention(x, x, x, mask))
        x = self.norm1(x + attn)
        ff = self.dropout(self.feed_forward(x))
        x = self.norm2(x + ff)
        return x

# Step 58 - __init__
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """题目给定，勿改。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        S = k.size(1)
        Q = self.W_q(q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """题目给定，勿改。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
    ) -> torch.Tensor:
        sa = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(sa))
        ca = self.cross_attention(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(ca))
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff))
        return x

# Step 59 - _linear_attn_kv_kernel
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _linear_attn_kv_kernel(
    K_ptr, V_ptr, KV_ptr, KSUM_ptr,
    L, D, stride_kl, stride_kd, stride_vl, stride_vd,
    stride_kvd, stride_kvdv, stride_ks,
    BLOCK_D: tl.constexpr,
):
   
    pid_d = tl.program_id(0)
    pid_dv = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_dv = pid_dv * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_dv = offs_dv < D

    acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    ksum_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for l in range(0, L):
        k = tl.load(K_ptr + l * stride_kl + offs_d * stride_kd, mask=mask_d, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + l * stride_vl + offs_dv * stride_vd, mask=mask_dv, other=0.0).to(tl.float32)
        acc += k[:, None] * v[None, :]
        if pid_dv == 0:
            ksum_acc += k

    tl.store(
        KV_ptr + offs_d[:, None] * stride_kvd + offs_dv[None, :] * stride_kvdv,
        acc,
        mask=mask_d[:, None] & mask_dv[None, :],
    )
    if pid_dv == 0:
        tl.store(KSUM_ptr + offs_d * stride_ks, ksum_acc, mask=mask_d)


@triton.jit
def _linear_attn_out_kernel(
    Q_ptr, KV_ptr, KSUM_ptr, O_ptr,
    L, D, eps,
    stride_ql, stride_qd, stride_kvd, stride_kvdv, stride_ks, stride_ol, stride_od,
    BLOCK_D: tl.constexpr,
):
    """对单个 batch：O = Q @ KV / (Q @ k_sum + eps)；Q 已做 φ。"""
    pid_l = tl.program_id(0)
    if pid_l >= L:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q = tl.load(Q_ptr + pid_l * stride_ql + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
    ksum = tl.load(KSUM_ptr + offs_d * stride_ks, mask=mask_d, other=0.0).to(tl.float32)
    den = tl.maximum(tl.sum(q * ksum), eps)

    out = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_dv = d0 + tl.arange(0, BLOCK_D)
        mask_dv = offs_dv < D
        # KV[d, dv]；对 d 维累加 q[d] * KV[d, dv]
        kv = tl.load(
            KV_ptr + offs_d[:, None] * stride_kvd + offs_dv[None, :] * stride_kvdv,
            mask=mask_d[:, None] & mask_dv[None, :],
            other=0.0,
        ).to(tl.float32)
        out += tl.sum(q[:, None] * kv, axis=0)

    tl.store(O_ptr + pid_l * stride_ol + offs_d * stride_od, out / den, mask=mask_d)


def linear_attention(Q, K, V, eps: float = 1e-6):
    """接口与旧题一致：Q,K,V (..., L, D) -> (..., L, D)。"""
    orig_shape = Q.shape
    L, D = int(orig_shape[-2]), int(orig_shape[-1])
    Qf = F.elu(Q.to(torch.float32)).add_(1.0).reshape(-1, L, D).contiguous()
    Kf = F.elu(K.to(torch.float32)).add_(1.0).reshape(-1, L, D).contiguous()
    Vf = V.to(torch.float32).reshape(-1, L, D).contiguous()
    B = Qf.shape[0]
    Out = torch.empty_like(Qf)

    BLOCK = triton.next_power_of_2(D)
    assert BLOCK <= 128, "本题 Triton 参考实现要求 D<=128"

    for b in range(B):
        Kv = torch.empty((D, D), device=Q.device, dtype=torch.float32)
        Ksum = torch.empty((D,), device=Q.device, dtype=torch.float32)
        grid_kv = (triton.cdiv(D, BLOCK), triton.cdiv(D, BLOCK))
        _linear_attn_kv_kernel[grid_kv](
            Kf[b], Vf[b], Kv, Ksum,
            L, D,
            Kf.stride(1), Kf.stride(2),
            Vf.stride(1), Vf.stride(2),
            Kv.stride(0), Kv.stride(1), Ksum.stride(0),
            BLOCK_D=BLOCK,
        )
        _linear_attn_out_kernel[(L,)](
            Qf[b], Kv, Ksum, Out[b],
            L, D, float(eps),
            Qf.stride(1), Qf.stride(2),
            Kv.stride(0), Kv.stride(1), Ksum.stride(0),
            Out.stride(1), Out.stride(2),
            BLOCK_D=BLOCK,
        )

    return Out.reshape(orig_shape).to(Q.dtype)

# Step 60 - __init__ (not yet solved)
# TODO: implement

# Step 61 - ring_attention
import math
import torch

def ring_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_devices: int,
) -> torch.Tensor:
    """
    Q/K/V: (B, S, D)，S 可被 num_devices 整除。
    返回与标准 softmax(QK^T / sqrt(D)) V 同形同值的 (B, S, D)。
    """
    b, s, d = Q.shape
    n = int(num_devices)
    chunk = s // n
    scale = 1.0 / math.sqrt(d)

    q_blks = list(Q.split(chunk, dim=1))
    k_blks = list(K.split(chunk, dim=1))
    v_blks = list(V.split(chunk, dim=1))

    outs = []
    for i in range(n):
        qi = q_blks[i]
        # m, l, oi：online Softmax 未归一化累加
        m = torch.full((b, chunk), float("-inf"), dtype=Q.dtype, device=Q.device)
        l = torch.zeros(b, chunk, dtype=Q.dtype, device=Q.device)
        oi = torch.zeros(b, chunk, d, dtype=Q.dtype, device=Q.device)

        for step in range(n):
            j = (i + step) % n
            scores = torch.matmul(qi, k_blks[j].transpose(-2, -1)) * scale
            m2 = torch.maximum(m, scores.amax(dim=-1))
            p = torch.exp(scores - m2.unsqueeze(-1))
            a = torch.exp(m - m2)
            l = a * l + p.sum(dim=-1)
            oi = a.unsqueeze(-1) * oi + torch.matmul(p, v_blks[j])
            m = m2

        outs.append(oi / l.unsqueeze(-1))

    return torch.cat(outs, dim=1)

# Step 62 - token_accuracy
import torch

def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:

    pred = torch.argmax(logits, dim=-1)
    mask = targets != ignore_index
    correct = (pred == targets) & mask
    count_valid = mask.sum().item()
    if count_valid == 0:
        return 0.0
    return correct.sum().item() / count_valid
