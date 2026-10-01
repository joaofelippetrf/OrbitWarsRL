"""Forward TRANSFORMER + features novas em NUMPY puro (para o deploy Kaggle).
Validado vs o JAX (jax_ppo.PolicyTF / features / pair_features) a ~1e-6.

Mapa dos tensores (PolicyTF d=112, 2 layers, 4 heads):
  Dense_0: embed 14->112
  bloco k: LayerNorm_(2k), MHA_k, LayerNorm_(2k+1), FFN[Dense inner 112->224, outer 224->112]
  Dense_5/6: encoder global (118->112, 112->112)
  Dense_7/8 (h), Dense_9/10 (k): pointer (outer 112->112, inner 224->112)
  Dense_11/12: pointer-bias (3->8, 8->1)
  Dense_13 noop, Dense_14 value, Dense_15 frac
"""
import math
import numpy as np

CENTER = 50.0; SUN_R = 10.0; ROT_LIMIT = 50.0; MAX_SPEED = 6.0
LOG1000 = math.log(1000.0); SUN_BUF = 2.0; MARGIN = 1.1


def _relu(x): return np.maximum(x, 0.0)
def _lin(x, W, name): return x @ W[f"params/{name}/kernel"] + W[f"params/{name}/bias"]

def layernorm(x, scale, bias, eps=1e-6):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * scale + bias

def mha(y, mask, W, pre, heads=4):
    P, d = y.shape; hd = d // heads
    Q = np.einsum('pd,dhf->phf', y, W[f"params/{pre}/query/kernel"]) + W[f"params/{pre}/query/bias"]
    K = np.einsum('pd,dhf->phf', y, W[f"params/{pre}/key/kernel"])   + W[f"params/{pre}/key/bias"]
    V = np.einsum('pd,dhf->phf', y, W[f"params/{pre}/value/kernel"]) + W[f"params/{pre}/value/bias"]
    s = np.einsum('qhf,khf->hqk', Q, K) / math.sqrt(hd)
    s = np.where(mask[None, None, :], s, -1e30)
    s = s - s.max(-1, keepdims=True); w = np.exp(s); w = w / w.sum(-1, keepdims=True)
    o = np.einsum('hqk,khf->qhf', w, V)
    return np.einsum('qhf,hfd->qd', o, W[f"params/{pre}/out/kernel"]) + W[f"params/{pre}/out/bias"]

def forward_tf(W, feat, pm, glob, pair, heads=4, layers=2):
    P = feat.shape[0]
    x = _lin(feat, W, "Dense_0")
    blocks = [("LayerNorm_0","LayerNorm_1","MultiHeadDotProductAttention_0","Dense_2","Dense_1"),
              ("LayerNorm_2","LayerNorm_3","MultiHeadDotProductAttention_1","Dense_4","Dense_3")]
    for (l1, l2, mh, fin, fout) in blocks[:layers]:
        y = layernorm(x, W[f"params/{l1}/scale"], W[f"params/{l1}/bias"])
        x = x + mha(y, pm, W, mh, heads)
        y = layernorm(x, W[f"params/{l2}/scale"], W[f"params/{l2}/bias"])
        x = x + _lin(_relu(_lin(y, W, fin)), W, fout)
    m = pm[:, None].astype(np.float64)
    pmean = (x * m).sum(0) / max(m.sum(), 1.0)
    g = _relu(_lin(np.concatenate([pmean, glob]), W, "Dense_5"))
    g = _lin(g, W, "Dense_6")
    cat = np.concatenate([x, np.broadcast_to(g[None, :], (P, g.shape[0]))], -1)
    h = _lin(_relu(_lin(cat, W, "Dense_8")), W, "Dense_7")
    k = _lin(_relu(_lin(cat, W, "Dense_10")), W, "Dense_9")
    scores = (h @ k.T) / math.sqrt(h.shape[1])
    scores = scores + _lin(_relu(_lin(pair, W, "Dense_11")), W, "Dense_12")[..., 0]
    noop = _lin(h, W, "Dense_13")
    logits = np.concatenate([noop, scores], -1)
    value = _lin(g, W, "Dense_14")[0]
    frac = _lin(h, W, "Dense_15")
    return logits, frac, value


# --- features novas (14 por planeta) + pair (3) -- arrays ja na perspectiva canonica ---
def _speed(n):
    s = np.clip(np.log(np.maximum(n, 1.0)) / LOG1000, 0, 1)
    return np.minimum(1.0 + (MAX_SPEED - 1.0) * s**1.5, MAX_SPEED)

def _ships_needed(tsh, tpr, tow, dist):
    prod = (tow != -1).astype(float); guess = tsh + 1.0
    for _ in range(2):
        eta = dist / _speed(guess); defenders = tsh + prod * tpr * eta; guess = defenders + 1.0
    return np.ceil((defenders + 1.0) * MARGIN)

def _seg_center(ax, ay, bx, by):
    vx, vy = ax - bx, ay - by; l2 = vx*vx + vy*vy
    t = np.where(l2 == 0, 0.0, np.clip(((CENTER-ax)*(bx-ax)+(CENTER-ay)*(by-ay))/np.where(l2==0,1,l2), 0, 1))
    return np.hypot(CENTER-(ax+t*(bx-ax)), CENTER-(ay+t*(by-ay)))

def features14(po, px, py, pr, ps, pp, porb, pv, step, me=0):
    mine = (po==me)&pv; enemy = (po!=me)&(po!=-1)&pv; neutral = (po==-1)&pv
    rentab = pp/(ps+1.0)
    dij = np.hypot(px[:,None]-px[None,:], py[:,None]-py[None,:])
    dist_enemy = np.where(enemy[None,:], dij, 1e9).min(1)
    dist_enemy = np.where(dist_enemy > 1e8, 140.0, dist_enemy)
    turns_rest = max(500.0-step, 1.0); boost = max(0.0, 1.0-step/40.0)
    value_rest = pp*turns_rest*(1.0+boost)
    return np.stack([mine.astype(float), enemy.astype(float), neutral.astype(float),
        px/100, py/100, pr/3, np.log1p(np.maximum(ps,0))/7, pp/5, porb.astype(float),
        np.hypot(px-CENTER, py-CENTER)/70, np.arctan2(py-CENTER, px-CENTER)/math.pi,
        np.minimum(rentab, 6.0)/3, dist_enemy/70, np.log1p(value_rest)/9], -1)

def pair_features14(px, py, pr, ps, pp, po):
    xi=px[:,None]; yi=py[:,None]; xj=px[None,:]; yj=py[None,:]
    dist = np.hypot(xi-xj, yi-yj); segc = _seg_center(xi, yi, xj, yj)
    need = _ships_needed(ps[None,:], pp[None,:], po[None,:], dist)
    sun = (segc < SUN_R+SUN_BUF).astype(float)
    return np.stack([dist/70, np.log1p(need)/7, sun], -1)
