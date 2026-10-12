"""Clone-resilient PLA model.

Snapshot encoder: per-access-point CSI token (conv over antennas x sub-bands) + AP identity + power share,
then set attention across the distributed APs (cell-free) -> AP tokens and a snapshot embedding.
Channel memory: causal attention over [history ..., query] with
  * PaTH position encoding (Yang et al., NeurIPS 2025): logits q_i^T (H_i H_{i-1} ... H_{j+1}) k_j,
    H_s = I - beta_s w_s w_s^T, data-dependent Householder-like transforms accumulated along the path;
  * forget gate (Forgetting Transformer, Lin et al., ICLR 2025): + sum_{s=j+1}^{i} log f_s.
Ablation switches: pos in {path, rope, none}, forget on/off, spatial in {attn, mean}, K (history length).
Heads: spoof logit at the query position; latent channel-prediction head (predict query embedding from history).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class APTokenizer(nn.Module):
    def __init__(self, M, Fb, d, A):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, (3, 5), padding=(1, 2)), nn.GELU(),
            nn.Conv2d(32, 64, (3, 5), stride=(1, 2), padding=(1, 2)), nn.GELU(),
            nn.Conv2d(64, 64, (3, 5), stride=(2, 2), padding=(1, 2)), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 4)))
        self.proj = nn.Linear(64 * 8, d)
        self.ap_emb = nn.Parameter(torch.randn(A, d) * 0.02)
        self.pw = nn.Linear(1, d)

    def forward(self, Xn, lp):                     # Xn: B,T,A,M,F,2 ; lp: B,T,A
        B, T, A, M, Fb, _ = Xn.shape
        x = Xn.reshape(B * T * A, M, Fb, 2).permute(0, 3, 1, 2)
        mag = (x ** 2).sum(1, keepdim=True).sqrt()
        x = torch.cat([x, mag], 1) * math.sqrt(A * M * Fb)       # unit-norm snapshot -> O(1) entries
        h = self.proj(self.conv(x).flatten(1)).view(B, T, A, -1)
        return h + self.ap_emb[None, None, :A] + self.pw(lp[..., None] / 20.0)


class SetBlock(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        h = self.n1(x); x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class PathForgetAttention(nn.Module):
    def __init__(self, d, nh, pos="path", forget=True):
        super().__init__()
        self.nh, self.dh, self.pos, self.forget = nh, d // nh, pos, forget
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        if pos == "path":
            self.w = nn.Linear(d, d)            # Householder directions per head
            self.beta = nn.Linear(d, nh)
        if forget:
            self.f = nn.Linear(d, nh)
            nn.init.constant_(self.f.bias, 3.0)  # start close to "remember"
        self.last_attn = None

    def _rope(self, x):
        T, dh = x.shape[-2], x.shape[-1]
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
        ang = torch.arange(T, device=x.device).float()[:, None] * inv[None]
        c, s = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, x):                        # x: B,T,d (causal)
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)   # B,h,T,dh
        if self.pos == "rope":
            q, k = self._rope(q), self._rope(k)
        if self.pos == "path":
            w = F.normalize(self.w(x).view(B, T, self.nh, self.dh).transpose(1, 2), dim=-1)  # B,h,T,dh
            beta = 2 * torch.sigmoid(self.beta(x)).transpose(1, 2)                             # B,h,T
            # transformed keys: Kt[i, j] = H_i ... H_{j+1} k_j  (j < i), Kt[i, i] = k_i
            cols, cur = [], None
            for i in range(T):
                if cur is not None:
                    wi, bi = w[:, :, i], beta[:, :, i]
                    cur = cur - bi[..., None, None] * wi[:, :, None] * (cur * wi[:, :, None]).sum(-1, keepdim=True)
                    cur = torch.cat([cur, k[:, :, i:i + 1]], 2)
                else:
                    cur = k[:, :, :1]
                cols.append(cur)
            logits = torch.full((B, self.nh, T, T), float("-inf"), device=x.device)
            for i in range(T):
                logits[:, :, i, :i + 1] = (q[:, :, i:i + 1] * cols[i]).sum(-1)
            logits = logits / math.sqrt(self.dh)
        else:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            logits = logits.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf"))
        if self.forget:
            lf = F.logsigmoid(self.f(x)).transpose(1, 2)              # B,h,T
            c = lf.cumsum(-1)
            logits = logits + (c[..., :, None] - c[..., None, :])   # sum_{s=j+1}^{i} log f_s
        a = logits.softmax(-1)
        self.last_attn = a.detach()
        y = (a @ v).transpose(1, 2).reshape(B, T, d)
        return self.o(y)


class MemBlock(nn.Module):
    def __init__(self, d, nh, pos, forget):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = PathForgetAttention(d, nh, pos, forget)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ff(self.n2(x))


class CloneGuard(nn.Module):
    def __init__(self, A, M, Fb, d=128, nh=4, n_set=2, n_mem=2, pos="path", forget=True, spatial="attn"):
        super().__init__()
        self.tok = APTokenizer(M, Fb, d, A)
        self.spatial = spatial
        self.set = nn.ModuleList([SetBlock(d, nh) for _ in range(n_set)]) if spatial == "attn" else None
        self.pool_q = nn.Parameter(torch.randn(d) * 0.02)
        self.mem = nn.ModuleList([MemBlock(d, nh, pos, forget) for _ in range(n_mem)])
        self.qflag = nn.Parameter(torch.randn(d) * 0.02)     # marks the query (untrusted) position
        self.nf = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.pred = nn.Linear(d, d)
        self.last_pool = None

    def snapshot(self, Xn, lp):
        h = self.tok(Xn, lp)                                  # B,T,A,d
        B, T, A, d = h.shape
        h = h.reshape(B * T, A, d)
        if self.set is not None:
            for blk in self.set: h = blk(h)
        w = (h @ self.pool_q / math.sqrt(d)).softmax(-1)      # attention pooling over APs
        self.last_pool = w.view(B, T, A).detach()
        z = (w[..., None] * h).sum(1)
        return z.view(B, T, d)

    def forward_from_z(self, z):
        """memory + heads on precomputed snapshot embeddings z (B,T,d); the run-time authenticator caches the
        embeddings of accepted snapshots and tokenises only the query"""
        x = z.clone()
        x[:, -1] = x[:, -1] + self.qflag
        for blk in self.mem: x = blk(x)
        x = self.nf(x)
        hq, hprev = x[:, -1], x[:, -2]
        logit = self.head(torch.cat([hq, hq - hprev], -1)).squeeze(-1)
        return logit, self.pred(hprev), z[:, -1]

    def forward(self, Xn, lp):
        z = self.snapshot(Xn, lp)                             # B,T,d   (T = K+1, last = query)
        x = z.clone()
        x[:, -1] = x[:, -1] + self.qflag
        for blk in self.mem: x = blk(x)
        x = self.nf(x)
        hq, hprev = x[:, -1], x[:, -2]
        logit = self.head(torch.cat([hq, hq - hprev], -1)).squeeze(-1)
        zpred = self.pred(hprev)                               # predicts the query embedding from history
        return logit, zpred, z[:, -1]
