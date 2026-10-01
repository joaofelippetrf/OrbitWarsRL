"""PPO 100% JAX para Orbit Wars, usando o motor rapido `jax_env`.

Tudo roda no device (sem Python no loop quente): features, mascaras, decode da
acao (ponteiro->jogada), politica (flax), rollout (lax.scan), GAE e update (optax).

Politica = DeepSets + pointer (cada planeta meu escolhe um alvo ou no-op), igual
em espirito ao policy.py (torch), mas em flax. Angulo e nº de naves saem da
geometria (intercept preditivo + ships_needed) reescrita em JAX.

v1: oponente = GREEDY em JAX (alvo valido mais proximo). Self-play (oponente =
snapshot congelado) via --selfplay. Avaliacao = win-rate vs greedy.

Uso:
    python3 jax_ppo.py --iters 40 --games 64 --horizon 256
"""
import argparse
import math
import os
import pickle
import threading
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.linen as nn
from jax.scipy.stats import beta as _beta_dist
from jax.scipy.special import betaln as _betaln, digamma as _digamma

# ---------------------------------------------------------------------------
# Avaliacao contra o motor REAL (kaggle engine) + heuristicas (agent7, etc.)
# ---------------------------------------------------------------------------
def _obs_to_jax_state1(obs, me=0):
    """Converte uma observacao do motor kaggle num J.State com batch=1.
    me: indice do jogador atual (0 ou 1); remapeia owners para perspectiva 0.
    me==1: rotaciona 180 (pos = 100-pos, ang+pi) p/ perspectiva CANONICA -- a
    mesma do flip_state do treino; quem chama corrige os angulos com +pi."""
    import jax_env as _J
    planets = obs["planets"]
    omega   = float(obs.get("angular_velocity", 0.03))
    step_n  = int(obs.get("step", 0))
    flip = (me == 1)
    n = min(len(planets), _J.P_MAX)
    pv = np.zeros(_J.P_MAX, bool);          po = np.full(_J.P_MAX, -1, np.int32)
    px = np.zeros(_J.P_MAX, np.float32);    py = np.zeros(_J.P_MAX, np.float32)
    pr = np.zeros(_J.P_MAX, np.float32);    ps = np.zeros(_J.P_MAX, np.float32)
    pp = np.zeros(_J.P_MAX, np.float32);    porb = np.zeros(_J.P_MAX, bool)
    for i, pl in enumerate(planets[:_J.P_MAX]):
        _id, owner, x, y, r, ships, prod = pl
        pv[i]   = True
        if owner is None or owner == -1:
            po[i] = -1
        elif owner == me:
            po[i] = 0   # sempre vejo meus planetas como owner=0
        else:
            po[i] = 1   # inimigo = owner=1
        px[i]   = (100.0 - x) if flip else x
        py[i]   = (100.0 - y) if flip else y
        pr[i]   = r
        ps[i]   = ships;  pp[i] = prod
        porb[i] = (math.hypot(x - _J.CENTER, y - _J.CENTER) + r) < _J.ROT_LIMIT
    # frotas em voo (necessario p/ as features inbound do v3; antes eram zeradas).
    fleets = obs.get("fleets", []) or []
    fa = np.zeros(_J.F_MAX, bool);          fo = np.zeros(_J.F_MAX, np.int32)
    fx = np.zeros(_J.F_MAX, np.float32);    fy = np.zeros(_J.F_MAX, np.float32)
    fang = np.zeros(_J.F_MAX, np.float32);  fsh = np.zeros(_J.F_MAX, np.float32)
    for i, fl in enumerate(fleets[:_J.F_MAX]):
        _fid, fown, fxx, fyy, fa_ang, _from, fships = fl
        fa[i] = True
        fo[i] = 0 if fown == me else 1     # mesma perspectiva dos planetas
        fx[i] = (100.0 - fxx) if flip else fxx
        fy[i] = (100.0 - fyy) if flip else fyy
        fang[i] = (fa_ang + math.pi) if flip else fa_ang
        fsh[i] = fships
    b = lambda a, dt: jnp.asarray(a.astype(dt)[None])
    return _J.State(
        p_valid=b(pv,bool),       p_owner=b(po,np.int32),
        p_x=b(px,np.float32),     p_y=b(py,np.float32),
        p_r=b(pr,np.float32),     p_ships=b(ps,np.float32),
        p_prod=b(pp,np.float32),  p_orbit=b(porb,bool),
        f_active=b(fa,bool),      f_owner=b(fo,np.int32),
        f_x=b(fx,np.float32),     f_y=b(fy,np.float32),
        f_ang=b(fang,np.float32), f_ships=b(fsh,np.float32),
        omega=jnp.array([omega],np.float32),
        step =jnp.array([step_n],np.int32))


_kaggle_infer_cache = {}  # (model_id) -> jit'd _infer fn; avoids recompile per eval cycle

def make_kaggle_agent(params, model):
    """Encapsula a politica JAX como callable (obs_kaggle -> moves).
    Usa o mesmo decode preditivo do jax_env (intercept + ships_needed)."""
    import jax_env as _J

    model_id = id(model)
    if model_id not in _kaggle_infer_cache:
        def _infer_impl(p, st):
            _, _, _, _, ang, ships, src, _ = policy_act(p, model, st, 0,
                                                         jax.random.PRNGKey(0), sample=False)
            return ang, ships, src
        _kaggle_infer_cache[model_id] = jax.jit(_infer_impl)
    _infer = _kaggle_infer_cache[model_id]

    def agent(obs, conf=None):
        me = int(obs.get("player", 0)) if hasattr(obs, "get") else int(getattr(obs, "player", 0))
        st = _obs_to_jax_state1(obs, me=me)
        ang_j, ships_j, src_j = _infer(params, st)
        ang_np   = np.array(ang_j[0])    # [P_MAX]
        ships_np = np.array(ships_j[0])  # [P_MAX]
        src_np   = np.array(src_j[0])    # [P_MAX] bool
        # me==1: o estado foi rotacionado 180 (perspectiva canonica) -> o angulo
        # gerado esta no frame rotacionado; corrige de volta com +pi.
        ang_off = math.pi if me == 1 else 0.0
        planets  = obs["planets"]
        moves = []
        for i in range(min(len(planets), _J.P_MAX)):
            if not src_np[i] or ships_np[i] < 1:
                continue
            planet_id = planets[i][0]    # id real (nao o indice)
            moves.append([planet_id, float(ang_np[i]) + ang_off, int(ships_np[i])])
        return moves
    return agent


def eval_vs_heuristic(params, model, n_games, mode="agent7", seed_base=8000):
    """Avalia a politica JAX contra agent7/starter/random usando o motor real.
    Lento (~10s/partida), use com --eval-heur-every alto.
    Retorna win-rate em [0,1] (empate = 0.5)."""
    import owenv
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent, starter_agent
    import features as _feat

    opp_fns = {"agent7": owenv.SUB.agent7, "starter": starter_agent, "random": random_agent}
    if mode == "dias":
        # bot LP do competidor GabrielDiasV (../dias/main.py)
        import importlib.util, os
        _dp = os.path.join(os.path.dirname(__file__), "..", "dias", "main.py")
        _sp = importlib.util.spec_from_file_location("dias_main", _dp)
        _dm = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_dm)
        opp_fns["dias"] = lambda obs: _dm.agent(obs)
    opp_fn  = opp_fns[mode]
    my_agent = make_kaggle_agent(params, model)
    wins = 0.0
    for k in range(n_games):
        g   = owenv.OWGame()
        obs = g.reset(seed=seed_base + k)
        done = False
        while not done:
            m0 = my_agent(obs)
            m1 = opp_fn(g.obs(1))
            obs, _, done = g.step(m0, m1)
        my, en = _feat.totals(obs, 0)
        wins += 1.0 if my > en else (0.5 if my == en else 0.0)
    return wins / n_games


def save_checkpoint(path, params, opt_state):
    """Salva pesos + estado do Adam (momentum/variancia) para retomada exata."""
    ckpt = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def load_checkpoint(path):
    """Carrega checkpoint completo. Retorna (params, opt_state) no device."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    # compatibilidade com checkpoints antigos (so tinham params)
    if isinstance(ckpt, dict) and "params" in ckpt:
        return jax.device_put(ckpt["params"]), jax.device_put(ckpt["opt_state"])
    return jax.device_put(ckpt), None   # arquivo antigo: opt_state None


def merge_params(fresh, loaded):
    """Sobrepoe os tensores de `loaded` em `fresh` quando o caminho/shape batem.
    Tensores novos (ex: head de fracao recem-adicionada) ficam com a init fresca.
    Caso especial: o kernel do EMBED (Dense_0) pode ter crescido no eixo de
    entrada (features novas APPENDADAS, ex: 16->20 c/ --feat-eta) -> copia as
    linhas antigas e ZERA as novas (a rede ignora as colunas novas no init =
    warm-start funcional intacto). Nao conta como 'carregado' de proposito:
    forca o reset do optimizer (o opt_state antigo tem shapes velhos).
    Retorna (params, n_carregados, n_total)."""
    fresh_flat = flax.traverse_util.flatten_dict(fresh)
    loaded_flat = flax.traverse_util.flatten_dict(loaded)
    merged = dict(fresh_flat)
    n = 0
    for k, v in loaded_flat.items():
        if k not in merged:
            continue
        if merged[k].shape == v.shape:
            merged[k] = v
            n += 1
        elif (k == ("params", "Dense_0", "kernel") and merged[k].ndim == 2
              and merged[k].shape[1] == v.shape[1]
              and merged[k].shape[0] > v.shape[0]):
            grown = np.zeros(merged[k].shape, np.asarray(v).dtype)
            grown[: v.shape[0]] = np.asarray(v)
            merged[k] = jnp.asarray(grown)
            print(f"merge: embed expandido {v.shape[0]}->{merged[k].shape[0]} feats "
                  f"(linhas novas zeradas)", flush=True)
    return flax.traverse_util.unflatten_dict(merged), n, len(fresh_flat)


# mantido para compatibilidade com checkpoints legados
def save_params(path, params):
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)


def load_params(path):
    with open(path, "rb") as f:
        return jax.device_put(pickle.load(f))

import jax_env as J
from jax_env import P_MAX, CENTER, SUN_R, ROT_LIMIT, MAX_SPEED, reset_batch, step_batched


def flip_state(st):
    """Rotaciona o estado 180 em torno do centro e troca owners 0<->1.
    O jogo e 180-simetrico, entao o player 1 visto por flip_state fica na MESMA
    perspectiva canonica que o player 0 (base em Q1). Como a politica so treina
    no slot 0, isso deixa o oponente self-play jogar tao bem quanto o player 0.
    Angulos de saida precisam de +pi (feito por quem chama).
    FROTAS tambem sao flipadas (pos 180, ang+pi, owner swap): com --feat-inbound
    a politica le f_* via inbound_ships; sem o flip o oponente self-play via
    frotas em posicoes/donos errados (bug que zoava as features inbound dele)."""
    swap = jnp.where(st.p_owner == 0, 1, jnp.where(st.p_owner == 1, 0, st.p_owner))
    fswap = jnp.where(st.f_owner == 0, 1, jnp.where(st.f_owner == 1, 0, st.f_owner))
    return st._replace(p_x=100.0 - st.p_x, p_y=100.0 - st.p_y, p_owner=swap,
                       f_x=100.0 - st.f_x, f_y=100.0 - st.f_y,
                       f_ang=st.f_ang + jnp.pi, f_owner=fswap)

PF = 14           # features por planeta (11 base + rentab/dist_inimigo/valor_restante)
# v3: se ligado, features() anexa +2 colunas (friend/enemy inbound) -> 16 feats.
# Flag de modulo (lido em trace-time pelo JIT). Default OFF: v1/v2 (14) intactos.
_FEAT_INBOUND = False
# v5: +4 colunas de inbound BUCKETIZADO por ETA (friend/enemy chegando em <=6 e
# <=18 turnos, cumulativo) -> 20 feats. Da nocao de TEMPO as frotas em voo (uma
# frota a 2 turnos e outra a 30 contavam igual). Exige _FEAT_INBOUND.
_FEAT_ETA = False
ETA_BUCKETS = (6.0, 18.0)
GF = 6            # features globais
PAIRF = 3         # features por-par (dist, ships_needed, sun) -> bias no pointer
# Head de fracao: cada planeta-fonte escolhe quanto das naves livres enviar.
FRAC_BINS = jnp.array([0.25, 0.5, 0.75, 1.0], jnp.float32)
N_FRAC = int(FRAC_BINS.shape[0])
GARRISON = 1.0
MAX_DIST = 60.0
MARGIN = 1.1
SUN_BUF = 2.0
LOG1000 = math.log(1000.0)


# ---------------------------------------------------------------------------
# Features / mascaras / geometria -- batched [B,P], lendo o State do jax_env.
# ---------------------------------------------------------------------------
def _speed(ships):
    s = 1.0 + (MAX_SPEED - 1.0) * (jnp.log(jnp.maximum(ships, 1.0)) / LOG1000) ** 1.5
    return jnp.minimum(s, MAX_SPEED)


def inbound_ships(st, me):
    """Naves em VOO convergindo para cada planeta, separadas por dono. [B,P] x2.
    Mesma geometria da heuristica pilkwang (dot>0 & perp<r+buf = frota mira o planeta).
    E o que a rede nao via -> causava overcommit (mandar leva 2x antes da 1a chegar)."""
    fx = st.f_x[:, :, None]; fy = st.f_y[:, :, None]           # [B,F,1]
    fdx = jnp.cos(st.f_ang)[:, :, None]; fdy = jnp.sin(st.f_ang)[:, :, None]
    dpx = st.p_x[:, None, :] - fx; dpy = st.p_y[:, None, :] - fy  # [B,F,P]
    dot = dpx * fdx + dpy * fdy
    perp = jnp.sqrt(jnp.maximum(dpx * dpx + dpy * dpy - dot * dot, 0.0))
    heading = st.f_active[:, :, None] & (dot > 0.0) & (perp < st.p_r[:, None, :] + 3.0)
    is_mine = (st.f_active & (st.f_owner == me))[:, :, None]
    is_enemy = (st.f_active & (st.f_owner != me) & (st.f_owner != -1))[:, :, None]
    fs = st.f_ships[:, :, None]
    friend = jnp.sum(jnp.where(heading & is_mine, fs, 0.0), 1)   # [B,P]
    enemy = jnp.sum(jnp.where(heading & is_enemy, fs, 0.0), 1)   # [B,P]
    return friend, enemy


def inbound_feats(st, me):
    """Colunas de inbound p/ as FEATURES, [B,P,K] (contagens cruas; features()
    aplica log1p). K=2 (friend,enemy totais) ou 6 com _FEAT_ETA: +friend/enemy
    chegando em <=ETA_BUCKETS turnos (cumulativo: 'logo' implica 'medio').
    Mesma geometria do inbound_ships; separado p/ o decode (que so usa o total)
    nao pagar os buckets."""
    fx = st.f_x[:, :, None]; fy = st.f_y[:, :, None]             # [B,F,1]
    fdx = jnp.cos(st.f_ang)[:, :, None]; fdy = jnp.sin(st.f_ang)[:, :, None]
    dpx = st.p_x[:, None, :] - fx; dpy = st.p_y[:, None, :] - fy   # [B,F,P]
    dot = dpx * fdx + dpy * fdy
    perp = jnp.sqrt(jnp.maximum(dpx * dpx + dpy * dpy - dot * dot, 0.0))
    heading = st.f_active[:, :, None] & (dot > 0.0) & (perp < st.p_r[:, None, :] + 3.0)
    is_mine = (st.f_active & (st.f_owner == me))[:, :, None]
    is_enemy = (st.f_active & (st.f_owner != me) & (st.f_owner != -1))[:, :, None]
    fs = st.f_ships[:, :, None]
    cols = [jnp.sum(jnp.where(heading & is_mine, fs, 0.0), 1),     # [B,P] friend total
            jnp.sum(jnp.where(heading & is_enemy, fs, 0.0), 1)]    # [B,P] enemy total
    if _FEAT_ETA:
        dist = jnp.sqrt(jnp.maximum(dpx * dpx + dpy * dpy, 0.0))   # [B,F,P]
        eta = dist / _speed(st.f_ships)[:, :, None]
        for t in ETA_BUCKETS:
            soon = heading & (eta <= t)
            cols.append(jnp.sum(jnp.where(soon & is_mine, fs, 0.0), 1))
            cols.append(jnp.sum(jnp.where(soon & is_enemy, fs, 0.0), 1))
    return jnp.stack(cols, -1)                                     # [B,P,K]


def features(st, me, inbound=None):
    owner, x, y = st.p_owner, st.p_x, st.p_y
    mine = (owner == me) & st.p_valid
    enemy = (owner != me) & (owner != -1) & st.p_valid
    neutral = (owner == -1) & st.p_valid
    # --- features de engenharia (inductive bias) por planeta ---
    # 1) rentabilidade: producao por nave de defesa (alvo barato e rico = alto)
    rentab = st.p_prod / (st.p_ships + 1.0)                       # [B,P]
    # 3) distancia ao INIMIGO mais proximo (fronteira/disputa)
    dij = jnp.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])  # [B,P,P]
    big = jnp.where(enemy[:, None, :], dij, 1e9)
    dist_enemy = jnp.min(big, -1)                                # [B,P]
    dist_enemy = jnp.where(dist_enemy > 1e8, 140.0, dist_enemy)  # sem inimigo -> longe
    # 4) valor restante = prod * turnos_restantes, com ENFASE no inicio (boost early)
    step = st.step.astype(jnp.float32)[:, None]                  # [B,1]
    turns_rest = jnp.maximum(500.0 - step, 1.0)
    boost = jnp.maximum(0.0, 1.0 - step / 40.0)                  # [B,1] alto so cedo
    value_rest = st.p_prod * turns_rest * (1.0 + boost)          # [B,P]
    feat = jnp.stack([
        mine.astype(jnp.float32), enemy.astype(jnp.float32), neutral.astype(jnp.float32),
        x / 100.0, y / 100.0, st.p_r / 3.0,
        jnp.log1p(jnp.maximum(st.p_ships, 0.0)) / 7.0, st.p_prod / 5.0,
        st.p_orbit.astype(jnp.float32),
        jnp.hypot(x - CENTER, y - CENTER) / 70.0,
        jnp.arctan2(y - CENTER, x - CENTER) / math.pi,
        jnp.minimum(rentab, 6.0) / 3.0,                          # 12: rentabilidade
        dist_enemy / 70.0,                                       # 13: dist inimigo proximo
        jnp.log1p(value_rest) / 9.0,                             # 14: valor restante (early-boosted)
    ], -1)                                                       # [B,P,PF]
    if _FEAT_INBOUND:
        # colunas de inbound no FIM (14,15 = friend/enemy totais; com _FEAT_ETA
        # +4 buckets por tempo de chegada). Anexar no fim mantem as anteriores
        # alinhadas c/ os pesos antigos (warm-start; embed expande c/ linha zero).
        # inbound pode vir PRE-COMPUTADO (do rollout): no update o estado e reconstruido
        # SEM frotas, entao recomputar daria 0 -> gradiente morto. Por isso passamos o
        # inbound salvo no traj. No rollout (inbound=None) computa do estado real.
        ib = inbound_feats(st, me) if inbound is None else inbound
        feat = jnp.concatenate([feat, jnp.log1p(ib) / 7.0], -1)
    pm = st.p_valid
    # globais (player me)
    my_sh = jnp.sum(jnp.where(mine, st.p_ships, 0.0), -1)
    en_sh = jnp.sum(jnp.where(enemy, st.p_ships, 0.0), -1)
    my_pr = jnp.sum(jnp.where(mine, st.p_prod, 0.0), -1)
    en_pr = jnp.sum(jnp.where(enemy, st.p_prod, 0.0), -1)
    glob = jnp.stack([
        st.step.astype(jnp.float32) / 500.0, st.omega * 20.0,
        jnp.log1p(my_sh) / 9.0, jnp.log1p(en_sh) / 9.0,
        my_pr / 30.0, en_pr / 30.0,
    ], -1)                                                       # [B,GF]
    return feat, pm, glob


def pair_features(st, me):
    """Features POR-PAR (fonte i -> alvo j), [B,P,P,3], p/ bias no pointer:
    1) distancia fonte->alvo  2) ships_needed na chegada  3) rota cruza o sol."""
    xi = st.p_x[:, :, None]; yi = st.p_y[:, :, None]
    xj = st.p_x[:, None, :]; yj = st.p_y[:, None, :]
    dist = jnp.hypot(xi - xj, yi - yj)                          # [B,P,P]
    segc = _seg_dist_center(xi, yi, xj, yj)
    tsh = st.p_ships[:, None, :]; tpr = st.p_prod[:, None, :]; tow = st.p_owner[:, None, :]
    need = _ships_needed(tsh, tpr, tow, dist)                   # [B,P,P]
    sun = (segc < SUN_R + SUN_BUF).astype(jnp.float32)
    return jnp.stack([dist / 70.0, jnp.log1p(need) / 7.0, sun], -1)  # [B,P,P,3]


def features_legacy(st, me):
    """As 11 features ORIGINAIS (pre feature-engineering). Para rodar checkpoints
    antigos (ex: pool10, d=64) como OPONENTE, que esperam 11 entradas e nao tem
    pointer-bias. Mesmos globais (6)."""
    owner, x, y = st.p_owner, st.p_x, st.p_y
    mine = (owner == me) & st.p_valid
    enemy = (owner != me) & (owner != -1) & st.p_valid
    neutral = (owner == -1) & st.p_valid
    feat = jnp.stack([
        mine.astype(jnp.float32), enemy.astype(jnp.float32), neutral.astype(jnp.float32),
        x / 100.0, y / 100.0, st.p_r / 3.0,
        jnp.log1p(jnp.maximum(st.p_ships, 0.0)) / 7.0, st.p_prod / 5.0,
        st.p_orbit.astype(jnp.float32),
        jnp.hypot(x - CENTER, y - CENTER) / 70.0,
        jnp.arctan2(y - CENTER, x - CENTER) / math.pi,
    ], -1)
    pm = st.p_valid
    my_sh = jnp.sum(jnp.where(mine, st.p_ships, 0.0), -1)
    en_sh = jnp.sum(jnp.where(enemy, st.p_ships, 0.0), -1)
    my_pr = jnp.sum(jnp.where(mine, st.p_prod, 0.0), -1)
    en_pr = jnp.sum(jnp.where(enemy, st.p_prod, 0.0), -1)
    glob = jnp.stack([
        st.step.astype(jnp.float32) / 500.0, st.omega * 20.0,
        jnp.log1p(my_sh) / 9.0, jnp.log1p(en_sh) / 9.0,
        my_pr / 30.0, en_pr / 30.0,
    ], -1)
    return feat, pm, glob


def _seg_dist_center(ax, ay, bx, by):
    vx, vy = ax - bx, ay - by
    l2 = vx * vx + vy * vy
    t = ((CENTER - ax) * (bx - ax) + (CENTER - ay) * (by - ay)) / jnp.where(l2 == 0, 1.0, l2)
    t = jnp.where(l2 == 0, 0.0, jnp.clip(t, 0.0, 1.0))
    return jnp.hypot(CENTER - (ax + t * (bx - ax)), CENTER - (ay + t * (by - ay)))


def masks(st, me):
    owner = st.p_owner
    mine = (owner == me) & st.p_valid
    free = jnp.maximum(st.p_ships - GARRISON, 0.0)
    src = mine & (free >= 1.0)                                   # [B,P]
    xi = st.p_x[:, :, None]; yi = st.p_y[:, :, None]            # [B,P,1]
    xj = st.p_x[:, None, :]; yj = st.p_y[:, None, :]            # [B,1,P]
    dist = jnp.hypot(xi - xj, yi - yj)                          # [B,P,P]
    segc = _seg_dist_center(xi, yi, xj, yj)                     # [B,P,P]
    eye = jnp.eye(P_MAX, dtype=bool)[None]
    tgt = (src[:, :, None] & st.p_valid[:, None, :] & (~eye)
           & (dist <= MAX_DIST) & (segc >= SUN_R + SUN_BUF))
    return src, tgt, free


def _ships_needed(tships, tprod, towner, dist):
    produces = (towner != -1).astype(jnp.float32)
    guess = tships + 1.0
    defenders = tships
    for _ in range(2):
        eta = dist / _speed(guess)
        defenders = tships + produces * tprod * eta
        guess = defenders + 1.0
    return jnp.ceil((defenders + 1.0) * MARGIN)


def decode(st, me, action_idx, free, frac=None):
    """action_idx[B,P] (0=no-op, j>=1 -> alvo linha j-1) -> (ang[B,P], ships[B,P])
    para o jogador `me`. Usa intercept preditivo + ships_needed.
    frac[B,P] (opcional): fracao das naves livres a enviar, escolhida pela
    politica. Se None, usa a heuristica antiga (ships_needed / reforco total)."""
    j = jnp.clip(action_idx - 1, 0, P_MAX - 1)                  # [B,P]
    tx = jnp.take_along_axis(st.p_x, j, 1); ty = jnp.take_along_axis(st.p_y, j, 1)
    tr = jnp.take_along_axis(st.p_r, j, 1)
    tsh = jnp.take_along_axis(st.p_ships, j, 1)
    tpr = jnp.take_along_axis(st.p_prod, j, 1)
    tow = jnp.take_along_axis(st.p_owner, j, 1)
    sx, sy, sr = st.p_x, st.p_y, st.p_r
    dist0 = jnp.hypot(tx - sx, ty - sy)
    need = _ships_needed(tsh, tpr, tow, dist0)
    reinforce = tow == me
    if frac is None:
        send = jnp.where(reinforce, free, jnp.minimum(free, jnp.maximum(need, 1.0)))
    else:
        send = frac * free
        # snap-to-need: ao ATACAR um alvo capturavel, nunca enviar MENOS que o
        # necessario (corrige o "erra por 1" do frac discreto + floor). Desconta
        # frotas amigas ja a caminho (friend_inbound) p/ nao overcommit em coalizao.
        friend_in, _ = inbound_ships(st, me)
        fin_t = jnp.take_along_axis(friend_in, j, 1)
        need_eff = jnp.ceil(jnp.maximum(need - fin_t, 0.0))
        capturable = (~reinforce) & (free >= need_eff) & (need_eff > 0.0)
        send = jnp.where(capturable, jnp.maximum(send, need_eff), send)
    send = jnp.floor(jnp.minimum(send, free))
    # intercept preditivo (rotaciona pos atual do alvo por omega*eta)
    sp = _speed(jnp.maximum(send, 1.0))
    omega = st.omega[:, None]
    ax, ay = tx, ty
    for _ in range(6):
        eta = jnp.maximum(0.0, jnp.hypot(ax - sx, ay - sy) - (sr + 0.1)) / sp
        ang = omega * eta
        dx, dy = tx - CENTER, ty - CENTER
        ca, sa = jnp.cos(ang), jnp.sin(ang)
        rx = CENTER + dx * ca - dy * sa
        ry = CENTER + dx * sa + dy * ca
        ax = jnp.where(st.p_orbit[:, :], rx, tx)
        ay = jnp.where(st.p_orbit[:, :], ry, ty)
    angle = jnp.arctan2(ay - sy, ax - sx)
    launching = (action_idx > 0) & (send >= 1.0)
    ships = jnp.where(launching, send, 0.0)
    return angle, ships


def greedy_action(st, me):
    """Oponente fixo: cada planeta ataca o alvo VALIDO mais proximo (ou no-op)."""
    src, tgt, free = masks(st, me)
    xi = st.p_x[:, :, None]; yi = st.p_y[:, :, None]
    xj = st.p_x[:, None, :]; yj = st.p_y[:, None, :]
    dist = jnp.hypot(xi - xj, yi - yj)
    big = jnp.where(tgt, dist, 1e9)
    nearest = jnp.argmin(big, -1)                               # [B,P]
    has = jnp.any(tgt, -1)
    action = jnp.where(has, nearest + 1, 0)
    return decode(st, me, action, free)


# Pesos do oponente economico (espelham A5/A6 do submission.py).
SMART_PROD_WEIGHT = 4.0
SMART_ENEMY_BONUS = 1.8

def smart_action(st, me):
    """Oponente economico: cada planeta escolhe o alvo VALIDO de melhor
    score = valor/custo, igual em espirito ao agent6/7.
      valor = prod*PROD_WEIGHT + 1   (x ENEMY_BONUS se for inimigo)
      custo = ships_needed(chegada) * (1 + dist/50)
    Bem mais forte que o greedy 'mais proximo' -> oponente de treino melhor."""
    src, tgt, free = masks(st, me)
    xi = st.p_x[:, :, None]; yi = st.p_y[:, :, None]
    xj = st.p_x[:, None, :]; yj = st.p_y[:, None, :]
    dist = jnp.hypot(xi - xj, yi - yj)                          # [B,P,P]
    tsh = st.p_ships[:, None, :]                                # [B,1,P]
    tpr = st.p_prod[:, None, :]
    tow = st.p_owner[:, None, :]
    need = _ships_needed(tsh, tpr, tow, dist)                   # [B,P,P]
    enemy = (tow != me) & (tow != -1)                           # [B,1,P]
    value = (tpr * SMART_PROD_WEIGHT + 1.0) * jnp.where(enemy, SMART_ENEMY_BONUS, 1.0)
    score = value / (jnp.maximum(need, 1.0) * (1.0 + dist / 50.0))
    score = jnp.where(tgt, score, -1.0)
    best = jnp.argmax(score, -1)                                # [B,P]
    has = jnp.any(tgt, -1)
    action = jnp.where(has, best + 1, 0)
    return decode(st, me, action, free)


# Pesos do pilkwang_action
_PILK_ENEMY_BONUS  = 2.0
_PILK_THREAT_MULT  = 1.2   # reserva = enemy_inbound * mult

def pilkwang_action(st, me):
    """Oponente inspirado no pilkwang v11 (top Kaggle). Tres melhorias sobre smart:
    1. Reserva dinamica: segura naves proporcional a frotas inimigas em voo.
    2. ships_needed ajustado: desconta frotas amigas ja a caminho do alvo.
    3. Valor proporcional a turnos restantes: prod * (remaining - eta_pessimista).
    Usa f_active/f_owner/f_ships/f_x/f_y/f_ang do State (frotas em voo no motor)."""
    src, tgt, free = masks(st, me)
    xi = st.p_x[:, :, None]; yi = st.p_y[:, :, None]
    xj = st.p_x[:, None, :]; yj = st.p_y[:, None, :]
    dist = jnp.hypot(xi - xj, yi - yj)                          # [B,P,P]

    # --- Rastreia frotas em voo: [B, F, P] ---
    # Para cada (frota f, planeta p): a frota esta indo em direcao a p?
    fx_e = st.f_x[:, :, None];  fy_e = st.f_y[:, :, None]      # [B,F,1]
    fdx  = jnp.cos(st.f_ang)[:, :, None]
    fdy  = jnp.sin(st.f_ang)[:, :, None]
    dpx  = st.p_x[:, None, :] - fx_e                            # [B,F,P]
    dpy  = st.p_y[:, None, :] - fy_e
    dot  = dpx * fdx + dpy * fdy
    perp = jnp.sqrt(jnp.maximum(dpx * dpx + dpy * dpy - dot * dot, 0.0))
    heading = (st.f_active[:, :, None]
               & (dot > 0.0)
               & (perp < st.p_r[:, None, :] + 3.0))             # [B,F,P]

    is_enemy = st.f_active & (st.f_owner != me) & (st.f_owner != -1)  # [B,F]
    is_mine  = st.f_active & (st.f_owner == me)
    fs = st.f_ships[:, :, None]                                  # [B,F,1]
    enemy_inbound  = jnp.sum(jnp.where(heading & is_enemy[:, :, None], fs, 0.0), 1)  # [B,P]
    friend_inbound = jnp.sum(jnp.where(heading & is_mine[:, :, None],  fs, 0.0), 1)  # [B,P]

    # 1. Reserva dinamica
    mine     = (st.p_owner == me) & st.p_valid
    reserve  = jnp.where(mine, enemy_inbound * _PILK_THREAT_MULT, 0.0)
    free_def = jnp.maximum(free - reserve, 0.0)                  # [B,P]

    # 2. ships_needed ajustado por suporte amigo
    tsh = st.p_ships[:, None, :]; tpr = st.p_prod[:, None, :]
    tow = st.p_owner[:, None, :]
    need = jnp.maximum(
        _ships_needed(tsh, tpr, tow, dist) - friend_inbound[:, None, :], 1.0)

    # 3. Valor proporcional a turnos restantes
    remaining   = jnp.maximum(500.0 - st.step[:, None, None].astype(jnp.float32), 1.0)
    turns_profit = jnp.maximum(remaining - dist / MAX_SPEED, 1.0)
    enemy_tgt   = (tow != me) & (tow != -1)
    value = (tpr * turns_profit + 1.0) * jnp.where(enemy_tgt, _PILK_ENEMY_BONUS, 1.0)

    score = value / (need * (1.0 + dist / 50.0))
    # Planetas sob ameaca nao atacam (reserva esgotada)
    not_threatened = ~(mine & (free_def < 1.0))
    score = jnp.where(tgt & not_threatened[:, :, None], score, -1.0)

    best  = jnp.argmax(score, -1)                               # [B,P]
    has   = jnp.any(tgt & not_threatened[:, :, None], -1)
    action = jnp.where(has, best + 1, 0)
    return decode(st, me, action, free_def)


# Pesos do smart_aggressive
_SA_ENEMY_BONUS    = 1.8
_SA_EARLY_BONUS    = 1.35   # neutros valem mais na fase inicial (expansao)
_SA_EARLY_STEPS    = 40
_SA_THREAT_RADIUS  = 30.0   # inimigo a <R ameaca o planeta
_SA_THREAT_FRAC    = 0.35   # reserva = fracao da forca inimiga proxima

def smart_aggressive(st, me):
    """Oponente forte SEM frotas em voo (compila com B grande, ao contrario do
    pilkwang). Melhora o smart com: valor ~ prod*turns_restantes, expansao inicial
    agressiva, e defesa leve por PROXIMIDADE de planetas inimigos (nao frotas).
    Captura por valor/custo (sem vies de tipo) -> pune o ponto cego de estaticos."""
    src, tgt, free = masks(st, me)
    xi = st.p_x[:, :, None]; yi = st.p_y[:, :, None]            # fonte i
    xj = st.p_x[:, None, :]; yj = st.p_y[:, None, :]            # alvo j
    dist = jnp.hypot(xi - xj, yi - yj)                          # [B,P,P]
    tsh = st.p_ships[:, None, :]; tpr = st.p_prod[:, None, :]
    tow = st.p_owner[:, None, :]
    need = _ships_needed(tsh, tpr, tow, dist)                   # [B,P,P]
    enemy_t  = (tow != me) & (tow != -1)
    neutral_t = (tow == -1)

    # valor ~ producao * turnos restantes de lucro (alvo perto rende mais cedo)
    remaining = jnp.maximum(500.0 - st.step[:, None, None].astype(jnp.float32), 1.0)
    turns_profit = jnp.maximum(remaining - dist / MAX_SPEED, 1.0)
    value = (tpr * turns_profit + 1.0) * jnp.where(enemy_t, _SA_ENEMY_BONUS, 1.0)
    # expansao inicial: neutros valem mais nos primeiros turnos
    early = st.step[:, None, None] < _SA_EARLY_STEPS
    value = value * jnp.where(neutral_t & early, _SA_EARLY_BONUS, 1.0)
    score = value / (jnp.maximum(need, 1.0) * (1.0 + dist / 50.0))

    # defesa leve: forca inimiga PROXIMA (planeta-planeta, sem frotas em voo)
    mine   = (st.p_owner == me) & st.p_valid                    # [B,P]
    en_msk = (st.p_owner != me) & (st.p_owner != -1) & st.p_valid
    en_sh  = st.p_ships[:, None, :]                             # [B,1,P]
    near_en = en_msk[:, None, :] & (dist < _SA_THREAT_RADIUS)   # [B,P,P]
    threat = jnp.sum(jnp.where(near_en, en_sh, 0.0), -1)        # [B,P]
    reserve = jnp.minimum(threat * _SA_THREAT_FRAC, free)
    free_def = jnp.maximum(free - reserve, 0.0)

    score = jnp.where(tgt, score, -1.0)
    best = jnp.argmax(score, -1)                                # [B,P]
    has = jnp.any(tgt, -1)
    action = jnp.where(has, best + 1, 0)
    return decode(st, me, action, free_def)


# ---------------------------------------------------------------------------
# Oponentes de COALIZAO (aproximam o LP de transporte do bot Dias): varios
# planetas concentram fogo no mesmo alvo ate cobrir `need`, com anti-overkill.
# Limitado a 1 alvo por fonte (restricao do jax_env) -> e um ASSIGNMENT, nao LP
# fracionario. Ambos sao [B,P,P]/[B,P] (compilam com B grande).
# ---------------------------------------------------------------------------
_CO_ENEMY_BONUS = 1.8
_CO_EARLY_BONUS = 1.35
_CO_EARLY_STEPS = 40
_CO_ETA_W       = 0.10
_CO_ITERS       = 6
_CO_PRICE_STEP  = 0.5

def _coalition_scores(st, me):
    """score[s,t], need_t[t] (capacidade do alvo) e free[s] compartilhados pelos
    dois oponentes de coalizao. score = valor/need - w*eta (espelha o Dias)."""
    src, tgt, free = masks(st, me)
    xi = st.p_x[:, :, None]; yi = st.p_y[:, :, None]
    xj = st.p_x[:, None, :]; yj = st.p_y[:, None, :]
    dist = jnp.hypot(xi - xj, yi - yj)
    tsh = st.p_ships[:, None, :]; tpr = st.p_prod[:, None, :]; tow = st.p_owner[:, None, :]
    need = jnp.maximum(_ships_needed(tsh, tpr, tow, dist), 1.0)  # [B,P,P]
    enemy_t = (tow != me) & (tow != -1)
    neutral_t = (tow == -1)
    remaining = jnp.maximum(500.0 - st.step[:, None, None].astype(jnp.float32), 1.0)
    eta = dist / MAX_SPEED
    turns_profit = jnp.maximum(remaining - eta, 1.0)
    value = (tpr * turns_profit + 1.0) * jnp.where(enemy_t, _CO_ENEMY_BONUS, 1.0)
    early = st.step[:, None, None] < _CO_EARLY_STEPS
    value = value * jnp.where(neutral_t & early, _CO_EARLY_BONUS, 1.0)
    score = value / need - _CO_ETA_W * eta                      # [B,P,P]
    score = jnp.where(tgt, score, -1e9)
    # capacidade do alvo = need minimo entre as fontes que o alcancam
    need_t = jnp.min(jnp.where(tgt, need, 1e9), axis=1)         # [B,P]
    need_t = jnp.where(need_t < 1e8, need_t, 0.0)
    return src, tgt, free, score, need_t

def smart_coalition(st, me):
    """B) Coalizao via LEILAO (auction) vetorizado: precos sobem em alvos
    sobre-demandados -> o excesso de fontes migra p/ outros alvos (anti-overkill)."""
    src, tgt, free, score, need_t = _coalition_scores(st, me)
    B, P, _ = score.shape
    price = jnp.zeros((B, P))
    has = jnp.any(tgt, -1) & (free >= 1.0)                      # [B,P]
    choice = jnp.zeros((B, P), jnp.int32)
    for _ in range(_CO_ITERS):
        adj = score - price[:, None, :]                        # [B,P,P]
        choice = jnp.argmax(adj, -1)                           # [B,P]
        onehot = jax.nn.one_hot(choice, P) * has[:, :, None].astype(jnp.float32)
        demand = jnp.sum(onehot * free[:, :, None], 1)         # [B,P] naves pedindo cada alvo
        over = jnp.maximum(demand - need_t, 0.0)
        price = price + _CO_PRICE_STEP * over / jnp.maximum(need_t, 1.0)
    action = jnp.where(jnp.any(tgt, -1), choice + 1, 0)
    return decode(st, me, action, free)

_SINK_EPS   = 0.5
_SINK_ITERS = 12

def sinkhorn_opp(st, me):
    """A) Coalizao via SINKHORN: transporte otimo regularizado (marginais =
    free por fonte, need por alvo). Aproxima a alocacao suave do LP; cada fonte
    lanca no alvo de maior massa transportada."""
    src, tgt, free, score, need_t = _coalition_scores(st, me)
    K = jnp.exp((score - jnp.max(score, -1, keepdims=True)) / _SINK_EPS)  # [B,P,P] estavel
    K = jnp.where(tgt, K, 0.0)
    a = jnp.maximum(free, 1e-6)                                 # [B,P] capacidade fonte
    b = jnp.maximum(need_t, 1e-6)                               # [B,P] capacidade alvo
    u = jnp.ones_like(a); v = jnp.ones_like(b)
    for _ in range(_SINK_ITERS):
        u = a / jnp.maximum(jnp.sum(K * v[:, None, :], -1), 1e-9)
        v = b / jnp.maximum(jnp.sum(K * u[:, :, None], 1), 1e-9)
    x = u[:, :, None] * K * v[:, None, :]                       # [B,P,P] transporte
    choice = jnp.argmax(jnp.where(tgt, x, -1.0), -1)
    action = jnp.where(jnp.any(tgt, -1), choice + 1, 0)
    return decode(st, me, action, free)


# ---------------------------------------------------------------------------
# Politica (flax linen): DeepSets + pointer
# ---------------------------------------------------------------------------
def _pointer_bias(pair_feat):
    """MLP pequeno [B,P,P,PAIRF] -> [B,P,P]: bias par (dist/ships_needed/sol) que
    soma aos scores fonte->alvo. Da a relacao tatica par-a-par direto ao pointer."""
    b = nn.relu(nn.Dense(8)(pair_feat))
    return nn.Dense(1)(b)[..., 0]


class Policy(nn.Module):
    d: int = 64

    @nn.compact
    def __call__(self, feat, pm, glob, pair_feat=None, frac_sel=None):
        d = self.d
        pe = nn.relu(nn.Dense(d)(feat)); pe = nn.Dense(d)(pe)    # [B,P,d]
        m = pm[..., None].astype(jnp.float32)
        pmean = (pe * m).sum(1) / jnp.clip(m.sum(1), 1.0)        # [B,d]
        g = nn.relu(nn.Dense(d)(jnp.concatenate([pmean, glob], -1)))
        g = nn.Dense(d)(g)                                      # [B,d]
        gexp = jnp.broadcast_to(g[:, None, :], feat.shape[:2] + (d,))
        cat = jnp.concatenate([feat, gexp], -1)
        h = nn.Dense(d)(nn.relu(nn.Dense(d)(cat)))             # fontes [B,P,d]
        k = nn.Dense(d)(nn.relu(nn.Dense(d)(cat)))             # alvos  [B,P,d]
        scores = jnp.einsum("bid,bjd->bij", h, k) / math.sqrt(d)
        if pair_feat is not None:
            scores = scores + _pointer_bias(pair_feat)         # bias par no pointer
        noop = nn.Dense(1)(h)                                   # [B,P,1]
        logits = jnp.concatenate([noop, scores], -1)           # [B,P,1+P]
        value = nn.Dense(1)(g)[:, 0]                           # [B]
        frac_logits = nn.Dense(N_FRAC)(h)                       # [B,P,N_FRAC]
        return logits, frac_logits, value


class PolicyTF(nn.Module):
    """Variante TRANSFORMER: blocos de self-attention deixam os planetas
    'conversarem' par-a-par (alem da media global do DeepSets). Mesma interface
    (logits, frac_logits, value) -> plugavel em policy_act/eval_logp sem mudanca.

    COMPAT de checkpoint: os modulos NOVOS (camadas >=2, ln_f, vh*, fk, fp*) tem
    nome EXPLICITO para nao deslocar a numeracao automatica (Dense_N) dos modulos
    antigos -- senao um ckpt de 2 camadas carregaria pesos nos slots errados.
    Os novos tambem usam kernel zero-init onde da: a rede warm-startada comeca
    computando EXATAMENTE o que computava antes (exceto --final-ln, que perturba)."""
    d: int = 96
    heads: int = 4
    layers: int = 2
    frac_pair: bool = False        # head de fracao condicionada no PAR (fonte,alvo)
    final_ln: bool = False         # LayerNorm de saida (padrao pre-LN)
    vhead: bool = False            # value head MLP 2 camadas + max-pool (alem do linear)
    frac_cont: bool = False        # fracao CONTINUA ~ Beta(a,b): head emite 2 params
    #                                crus (softplus+1 no policy_act) em vez de 4 bins

    @nn.compact
    def __call__(self, feat, pm, glob, pair_feat=None, frac_sel=None):
        # frac_sel [B,P] (opcional, so frac_pair): acao-alvo ja escolhida -> a head
        # de fracao e computada SO no par escolhido (evita o grid no update)
        d = self.d
        x = nn.Dense(d)(feat)                                   # [B,P,d] embed por planeta
        attn_mask = pm[:, None, None, :]                       # [B,1,1,P]: so atende validos
        for li in range(self.layers):
            y = nn.LayerNorm()(x)
            if li < 2:                                         # camadas originais (autonamed)
                a = nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=d)(
                    y, y, mask=attn_mask)
            else:
                # camadas extras: out_proj zero-init -> bloco = identidade no init
                a = nn.MultiHeadDotProductAttention(
                    num_heads=self.heads, qkv_features=d,
                    out_kernel_init=nn.initializers.zeros)(y, y, mask=attn_mask)
            x = x + a                                          # residual
            y = nn.LayerNorm()(x)
            if li < 2:
                ff = nn.Dense(d)(nn.relu(nn.Dense(2 * d)(y)))  # FFN
            else:
                ff = nn.Dense(d, kernel_init=nn.initializers.zeros,
                              name=f"ffn{li}_out")(
                    nn.relu(nn.Dense(2 * d, name=f"ffn{li}_in")(y)))
            x = x + ff
        if self.final_ln:
            x = nn.LayerNorm(name="ln_f")(x)                   # LN final (pre-LN padrao)
        m = pm[..., None].astype(jnp.float32)
        pmean = (x * m).sum(1) / jnp.clip(m.sum(1), 1.0)        # pooling do conjunto
        g = nn.relu(nn.Dense(d)(jnp.concatenate([pmean, glob], -1)))
        g = nn.Dense(d)(g)                                      # contexto global [B,d]
        gexp = jnp.broadcast_to(g[:, None, :], feat.shape[:2] + (d,))
        cat = jnp.concatenate([x, gexp], -1)                   # x ja tem contexto via attention
        h = nn.Dense(d)(nn.relu(nn.Dense(d)(cat)))             # fontes
        k = nn.Dense(d)(nn.relu(nn.Dense(d)(cat)))             # alvos
        scores = jnp.einsum("bid,bjd->bij", h, k) / math.sqrt(d)
        if pair_feat is not None:
            scores = scores + _pointer_bias(pair_feat)         # bias par no pointer
        noop = nn.Dense(1)(h)
        logits = jnp.concatenate([noop, scores], -1)
        value = nn.Dense(1)(g)[:, 0]
        if self.vhead:
            # correcao MLP por cima do head linear: ve tambem o max-pool dos
            # planetas (outliers tipo "frota gigante" somem na media). Zero-init
            # -> no warm-start o critico comeca = ao antigo.
            pmax = jnp.max(jnp.where(pm[..., None], x, -1e9), axis=1)   # [B,d]
            vh = nn.relu(nn.Dense(d, name="vh1")(jnp.concatenate([g, pmax], -1)))
            value = value + nn.Dense(1, kernel_init=nn.initializers.zeros,
                                     name="vh2")(vh)[:, 0]
        # frac_cont: 2 saidas (alpha,beta crus da Beta) em vez de N_FRAC logits.
        # OBS warm-start: a head antiga era [d,4] -> shape diferente -> init fresca
        # (kernel zero + bias zero = Beta(1.69,1.69), prior suave centrado em 0.5).
        n_f = 2 if self.frac_cont else N_FRAC
        if self.frac_pair:
            # fracao depende do PAR (fonte i, alvo j): quanto mandar muda com o alvo
            # (neutro fraco -> minimo; inimigo forte -> mais). Fatoracao aditiva barata:
            # termo da fonte + termo do alvo + termo tatico do par.
            # fh fica autonamed: herda os pesos da head antiga por-fonte (warm-start
            # so no modo bins); fk/fp zero-init -> no init a fracao so depende da fonte.
            fh_kinit = (nn.initializers.zeros if self.frac_cont
                        else nn.initializers.lecun_normal())
            fh_d = nn.Dense(n_f, kernel_init=fh_kinit)
            fk_d = nn.Dense(n_f, kernel_init=nn.initializers.zeros, name="fk")
            fp1_d = nn.Dense(8, name="fp1")
            fp2_d = nn.Dense(n_f, kernel_init=nn.initializers.zeros, name="fp2")
            if frac_sel is not None:
                # acao ja conhecida (caminho do UPDATE): seleciona o alvo ANTES de
                # computar a head -> [B,P,F]. O grid [B,P,P,F] no backward com
                # minibatch 8192 materializava ~9 GB (OOM).
                j = jnp.clip(frac_sel - 1, 0, k.shape[1] - 1)              # [B,P]
                k_t = jnp.take_along_axis(k, j[..., None], 1)              # [B,P,d]
                frac_logits = fh_d(h) + fk_d(k_t)                          # [B,P,F]
                if pair_feat is not None:
                    pf_sel = jnp.take_along_axis(
                        pair_feat, j[:, :, None, None], 2)[:, :, 0]        # [B,P,PAIRF]
                    frac_logits = frac_logits + fp2_d(nn.relu(fp1_d(pf_sel)))
            else:
                # rollout (batch pequeno): grid completo [B,P,P,F] p/ amostrar
                frac_logits = fh_d(h)[:, :, None, :] + fk_d(k)[:, None, :, :]
                if pair_feat is not None:
                    frac_logits = frac_logits + fp2_d(nn.relu(fp1_d(pair_feat)))
        else:
            fh_kinit = (nn.initializers.zeros if self.frac_cont
                        else nn.initializers.lecun_normal())
            frac_logits = nn.Dense(n_f, kernel_init=fh_kinit)(h)  # [B,P,F] (por fonte)
        return logits, frac_logits, value


NEG = -1e9
def masked_logits(logits, tgt):
    full = jnp.concatenate([jnp.ones_like(tgt[..., :1]), tgt], -1)
    return jnp.where(full, logits, NEG)


def _frac_for_action(frac_logits, action):
    """frac por PAR [B,P,P,F]: seleciona o destino escolhido (action-1) -> [B,P,F].
    frac por FONTE [B,P,F]: retorna inalterado. Mantem backcompat (head atual)."""
    if frac_logits.ndim == 4:
        j = jnp.clip(action - 1, 0, frac_logits.shape[2] - 1)          # [B,P]
        sel = jnp.take_along_axis(frac_logits, j[:, :, None, None], 2) # [B,P,1,F]
        return sel[:, :, 0, :]
    return frac_logits


def policy_act(params, model, st, me, key, sample=True, legacy=False, nfeat=None):
    # legacy=True: checkpoint antigo (11 features, sem pointer-bias) como oponente.
    # nfeat: oponente treinado com MENOS colunas que o learner atual (ex: 16 sem
    # eta vs 20 com) -> corta as colunas extras (sao sempre appendadas no fim).
    if legacy:
        feat, pm, glob = features_legacy(st, me); pf = None
    else:
        feat, pm, glob = features(st, me); pf = pair_features(st, me)
        if nfeat is not None:
            feat = feat[..., :nfeat]
    src, tgt, free = masks(st, me)
    logits, frac_logits, value = model.apply(params, feat, pm, glob, pf)
    logits = masked_logits(logits, tgt)
    k_a, k_f = jax.random.split(key)
    if sample:
        action = jax.random.categorical(k_a, logits, axis=-1)        # [B,P]
    else:
        action = jnp.argmax(logits, -1)
    fl = _frac_for_action(frac_logits, action)        # frac condicionada no destino
    if getattr(model, "frac_cont", False):
        # fracao CONTINUA ~ Beta(a,b) em (0,1). softplus+1 garante a,b>=1
        # (densidade unimodal, sem massa explosiva nas bordas). frac_action
        # salvo na traj e a PROPRIA fracao (float); logp = log-DENSIDADE.
        al = jax.nn.softplus(fl[..., 0]) + 1.0
        be = jax.nn.softplus(fl[..., 1]) + 1.0
        if sample:
            frac_action = jnp.clip(jax.random.beta(k_f, al, be), 1e-4, 1.0 - 1e-4)
        else:
            frac_action = al / (al + be)              # media (eval deterministico)
        lp_f = _beta_dist.logpdf(frac_action, al, be)
        frac = frac_action                                       # [B,P] float
    else:
        if sample:
            frac_action = jax.random.categorical(k_f, fl, axis=-1)
        else:
            frac_action = jnp.argmax(fl, -1)
        logp_f = jax.nn.log_softmax(fl, -1)
        lp_f = jnp.take_along_axis(logp_f, frac_action[..., None], -1)[..., 0]
        frac = FRAC_BINS[frac_action]                            # [B,P]
    logp_t = jax.nn.log_softmax(logits, -1)
    lp_t = jnp.take_along_axis(logp_t, action[..., None], -1)[..., 0]      # [B,P]
    srcf = src.astype(jnp.float32)
    launch = (action > 0).astype(jnp.float32)                    # frac so importa ao lancar
    # logp POR PLANETA (nao a soma): o PPO clipa cada planeta como uma acao
    # propria (vantagem compartilhada) -> ratio conjunto de ate P acoes tinha
    # variancia enorme e clipava o turno inteiro em bloco.
    logp = (lp_t + lp_f * launch) * srcf                         # [B,P]
    ang, ships = decode(st, me, action, free, frac)
    return action, frac_action, logp, value, ang, ships, src, tgt


def eval_logp(params, model, st, me, action, frac_action, inbound=None):
    feat, pm, glob = features(st, me, inbound=inbound)
    pf = pair_features(st, me)
    src, tgt, free = masks(st, me)
    # frac_sel=action: com frac_pair a head de fracao sai ja selecionada [B,P,F]
    logits, frac_logits, value = model.apply(params, feat, pm, glob, pf, frac_sel=action)
    logits = masked_logits(logits, tgt)
    fl = _frac_for_action(frac_logits, action)        # frac condicionada na acao tomada
    logp_t = jax.nn.log_softmax(logits, -1)
    lp_t = jnp.take_along_axis(logp_t, action[..., None], -1)[..., 0]
    if getattr(model, "frac_cont", False):
        # frac_action aqui e a FRACAO float amostrada no rollout (ja clipada)
        al = jax.nn.softplus(fl[..., 0]) + 1.0
        be = jax.nn.softplus(fl[..., 1]) + 1.0
        lp_f = _beta_dist.logpdf(jnp.clip(frac_action, 1e-4, 1.0 - 1e-4), al, be)
        # entropia DIFERENCIAL analitica da Beta (pode ser negativa; o bonus
        # +c*H empurra de volta pro uniforme do mesmo jeito)
        ent_f = (_betaln(al, be) - (al - 1.0) * _digamma(al)
                 - (be - 1.0) * _digamma(be) + (al + be - 2.0) * _digamma(al + be))
    else:
        logp_f = jax.nn.log_softmax(fl, -1)
        lp_f = jnp.take_along_axis(logp_f, frac_action[..., None], -1)[..., 0]
        p_f = jnp.exp(logp_f)
        ent_f = -(p_f * jnp.where(jnp.isfinite(logp_f), logp_f, 0.0)).sum(-1)
    srcf = src.astype(jnp.float32)
    launch = (action > 0).astype(jnp.float32)
    p_t = jnp.exp(logp_t)
    ent_t = -(p_t * jnp.where(jnp.isfinite(logp_t), logp_t, 0.0)).sum(-1)
    logp = (lp_t + lp_f * launch) * srcf                # [B,P] por planeta (vide policy_act)
    # entropias por cabeca, medias por amostra (para per-head ent_coef)
    nsrc = jnp.clip(srcf.sum(-1), 1.0)
    ent_t_mean = (ent_t * srcf).sum(-1) / nsrc                       # pointer [B]
    ent_f_mean = (ent_f * srcf * launch).sum(-1) / nsrc             # frac (so quando lanca) [B]
    return logp, srcf, ent_t_mean, ent_f_mean, value


def totals0(st):
    my = jnp.sum(jnp.where(st.p_valid & (st.p_owner == 0), st.p_ships, 0.0), -1) \
        + jnp.sum(jnp.where(st.f_active & (st.f_owner == 0), st.f_ships, 0.0), -1)
    en = jnp.sum(jnp.where(st.p_valid & (st.p_owner == 1), st.p_ships, 0.0), -1) \
        + jnp.sum(jnp.where(st.f_active & (st.f_owner == 1), st.f_ships, 0.0), -1)
    return my, en


def prod_totals0(st):
    """Producao total por jogador (proxy de territorio/expansao)."""
    myp = jnp.sum(jnp.where(st.p_valid & (st.p_owner == 0), st.p_prod, 0.0), -1)
    enp = jnp.sum(jnp.where(st.p_valid & (st.p_owner == 1), st.p_prod, 0.0), -1)
    return myp, enp


def count_totals0(st):
    """Numero de planetas por jogador (sinal de expansao mais direto: +1 ao capturar)."""
    myc = jnp.sum((st.p_valid & (st.p_owner == 0)).astype(jnp.float32), -1)
    enc = jnp.sum((st.p_valid & (st.p_owner == 1)).astype(jnp.float32), -1)
    return myc, enc


def static_count_totals0(st):
    """Numero de planetas ESTATICOS (nao orbitantes) por jogador. A politica do
    self-play tem um ponto cego: home orbitante so ataca orbitantes (e vice-versa),
    porque ambos os homes sao do mesmo tipo (simetria 180) -> ninguem disputa o
    outro tipo. Shaping neste termo torna estaticos mais atraentes e quebra o vies."""
    static = st.p_valid & (~st.p_orbit)
    myc = jnp.sum((static & (st.p_owner == 0)).astype(jnp.float32), -1)
    enc = jnp.sum((static & (st.p_owner == 1)).astype(jnp.float32), -1)
    return myc, enc


# ---------------------------------------------------------------------------
# Rollout (lax.scan) + GAE + PPO update
# ---------------------------------------------------------------------------
_RECAP_DECAY = 0.9   # decaimento da "volatilidade" de dono (~10 turnos de memoria)

def make_rollout(model, horizon, shape_w, shape_prod_w=0.0, shape_plan_w=0.0,
                 shape_static_w=0.0, shape_prod_early_w=0.0, prod_early_until=20,
                 shape_div_w=0.0, shape_defend_w=0.0, lead_ratio=1.5,
                 shape_capture_w=0.0, shape_recap_w=0.0, opp_fn=greedy_action,
                 opp_model=None, opp_legacy=False, shape_waste_w=0.0, waste_margin=2.0,
                 shape_idle_w=0.0, idle_until=30, idle_margin=0.0, both_sides=False,
                 opp_nfeat=None):
    om = opp_model if opp_model is not None else model  # rede do oponente (pode ter
    #                          largura diferente, ex: pool10 d=64 vs modelo 300K d=204)
    # both_sides=True: coleta TAMBEM a trajetoria do player 1 (estado flipado p/
    # a perspectiva canonica) e concatena no eixo do batch -> 2x amostras por
    # rollout de graca. So faz sentido com opp_params = params ATUAIS (self-play
    # puro): logp/value salvos vem da propria politica (on-policy).
    def rollout(params, opp_params, st, key):
        if both_sides and opp_params is None:
            raise ValueError("both_sides exige opp_params (self-play), nao heuristica")
        def stepf(carry, _):
            (st, key, done_acc, prev_diff, prev_pdiff, prev_cdiff,
             prev_sdiff, prev_div, vol) = carry
            key, k0, k1 = jax.random.split(key, 3)
            a0, fa0, logp0, val0, ang0, sh0, _, _ = policy_act(params, model, st, 0, k0)
            if opp_params is None:
                ang1, sh1 = opp_fn(st, 1)
            else:
                # oponente (snapshot/fixo) ve a perspectiva canonica (flip 180); angulo +pi
                fst = flip_state(st)
                a1, fa1, logp1, val1, ang1, sh1, _, _ = policy_act(opp_params, om, fst, 0, k1,
                                                                   legacy=opp_legacy,
                                                                   nfeat=opp_nfeat)
                ang1 = ang1 + jnp.pi
            nst, done, env_r, waste = step_batched(st, ang0, sh0, ang1, sh1)
            my, en = totals0(nst)
            diff = my - en
            myp, enp = prod_totals0(nst)
            pdiff = myp - enp                       # vantagem de PRODUCAO
            myc, enc = count_totals0(nst)
            cdiff = myc - enc                       # vantagem de PLANETAS (expansao direta)
            mys, ens = static_count_totals0(nst)
            sdiff = mys - ens                       # vantagem de planetas ESTATICOS (legado)
            my_orbit = myc - mys
            div = jnp.minimum(my_orbit, mys)        # diversidade (legado)
            newly_mine = (nst.p_owner == 0) & (st.p_owner != 0) & nst.p_valid
            cap_force = jnp.sum(jnp.where(newly_mine, nst.p_ships, 0.0), -1)  # captura concentrada (legado)
            # RETOMADA/OPORTUNISMO: capturar planetas VOLATEIS (trocaram de dono
            # recentemente). `vol[p]` e alto logo apos uma troca e decai. Capturar
            # um planeta volatil = retomada (2p) ou aproveitar guerra entre inimigos
            # (4p, planeta disputado troca de dono -> fica fragil). `vol` aqui e o
            # estado ANTES deste turno (recompensa pegar o que ja estava em disputa).
            recap = jnp.sum(jnp.where(newly_mine, vol, 0.0), -1)             # [B]
            pflip = (nst.p_owner != st.p_owner) & nst.p_valid & st.p_valid   # trocou de dono
            vol_new = jnp.maximum(vol * _RECAP_DECAY, pflip.astype(jnp.float32))
            early = (nst.step < prod_early_until).astype(jnp.float32)
            lead = (myp > enp * lead_ratio).astype(jnp.float32)
            # NAVES MAL USADAS: frota nossa aniquilada num ataque que chegou e
            # deixou o alvo com guarnicao residual <= waste_margin (faltou 1-2 pra
            # tomar). waste[...,0]=naves desperdicadas, waste[...,1]=residuo.
            near_waste = jnp.where(waste[..., 1] <= waste_margin, waste[..., 0], 0.0)
            waste_pen = jnp.sum(near_waste, -1)            # [B] naves desperdicadas/turno
            # WASTE DE PRODUCAO (so cedo): naves PARADAS acumuladas nos nossos
            # planetas alem de uma reserva `idle_margin`. Producao que nao virou
            # expansao/frota = desperdicio. So conta enquanto step < idle_until
            # (estagio inicial) -> empurra deploy/expansao agressiva no comeco.
            early_idle = (nst.step < idle_until).astype(jnp.float32)
            idle_mine = jnp.where(nst.p_valid & (nst.p_owner == 0),
                                  jnp.maximum(nst.p_ships - idle_margin, 0.0), 0.0)
            idle_pen = jnp.sum(idle_mine, -1)              # [B] naves ociosas/turno
            live = (~done_acc).astype(jnp.float32)
            first_done = done & (~done_acc)
            # potential-based shaping (tudo em Δ, subordinado ao terminal ±1).
            # shaped_anti = termos ANTISSIMETRICOS (diffs p0-p1): p/ o lado 1 e so
            # negar. shaped_rest = termos especificos do lado 0 (defesa, waste,
            # recap, idle...) -- no both_sides o lado 1 nao os recebe.
            shaped_anti = (shape_w * (diff - prev_diff) / 10.0
                           + shape_prod_w * (pdiff - prev_pdiff)
                           + shape_prod_early_w * (pdiff - prev_pdiff) * early
                           + shape_plan_w * (cdiff - prev_cdiff)
                           + shape_static_w * (sdiff - prev_sdiff))
            shaped_rest = (shape_div_w * (div - prev_div)
                           + shape_defend_w * jnp.minimum(cdiff - prev_cdiff, 0.0) * lead
                           + shape_capture_w * cap_force / 10.0
                           + shape_recap_w * recap                            # retomada/oportunismo
                           - shape_waste_w * waste_pen / 10.0                 # naves mal usadas
                           - shape_idle_w * idle_pen / 10.0 * early_idle)     # producao parada cedo
            r = jnp.where(first_done, env_r, shaped_anti + shaped_rest) * live
            done_acc = done_acc | done
            out = dict(action=a0, frac_action=fa0, logp=logp0, value=val0, reward=r, live=live,
                       waste=waste_pen,                                # [B] monitor naves mal usadas
                       idle=idle_pen,                                  # [B] monitor producao parada
                       p_owner=st.p_owner, p_x=st.p_x, p_y=st.p_y, p_r=st.p_r,
                       p_ships=st.p_ships, p_prod=st.p_prod, p_valid=st.p_valid,
                       p_orbit=st.p_orbit, omega=st.omega, step=st.step,
                       # inbound do ESTADO REAL (com frotas): salvo aqui pq no update o
                       # estado e reconstruido sem frotas. Sem isso, gradiente das feats
                       # inbound = 0 (bug que matava as 2 features novas do v3).
                       inbound=inbound_feats(st, 0))   # [B,P,K] (K=2 ou 6 c/ eta)
            if both_sides:
                # lado 1 (estado flipado = perspectiva canonica): mesma transicao,
                # recompensa espelhada. done/live identicos (o jogo e um so).
                r1 = jnp.where(first_done, -env_r, -shaped_anti) * live
                out1 = dict(action=a1, frac_action=fa1, logp=logp1, value=val1,
                            reward=r1, live=live,
                            waste=jnp.zeros_like(waste_pen),     # monitores so do lado 0
                            idle=jnp.zeros_like(idle_pen),
                            p_owner=fst.p_owner, p_x=fst.p_x, p_y=fst.p_y, p_r=st.p_r,
                            p_ships=st.p_ships, p_prod=st.p_prod, p_valid=st.p_valid,
                            p_orbit=st.p_orbit, omega=st.omega, step=st.step,
                            inbound=inbound_feats(fst, 0))
                out = {kk: jnp.concatenate([out[kk], out1[kk]], 0) for kk in out}
            return (nst, key, done_acc, diff, pdiff, cdiff, sdiff, div, vol_new), out
        B = st.p_owner.shape[0]
        # baselines do shaping potencial = estado INICIAL (nao zero): com starts de
        # MEIO DE JOGO, partir de zero dava um pulso espurio de recompensa no 1o passo.
        my_i, en_i = totals0(st);              diff0 = my_i - en_i
        myp_i, enp_i = prod_totals0(st);       pdiff0 = myp_i - enp_i
        myc_i, enc_i = count_totals0(st);      cdiff0 = myc_i - enc_i
        mys_i, ens_i = static_count_totals0(st); sdiff0 = mys_i - ens_i
        div0 = jnp.minimum(myc_i - mys_i, mys_i)
        vol0 = jnp.zeros((B, st.p_owner.shape[1]))
        init = (st, key, jnp.zeros(B, bool), diff0, pdiff0, cdiff0, sdiff0, div0, vol0)
        (last_st, *_), traj = jax.lax.scan(stepf, init, None, length=horizon)
        # bootstrap value
        _, _, _, last_val, _, _, _, _ = policy_act(params, model, last_st, 0,
                                                    jax.random.PRNGKey(0), sample=False)
        if both_sides:
            _, _, _, last_val1, _, _, _, _ = policy_act(params, model, flip_state(last_st),
                                                        0, jax.random.PRNGKey(0), sample=False)
            last_val = jnp.concatenate([last_val, last_val1], 0)
        return traj, last_val, last_st
    # JIT: sem isso o lax.scan era re-tracado/recompilado a cada iter e os
    # executaveis se acumulavam no cache do XLA -> vazamento de RAM de host
    # (~20 MB/iter) e OOM lá pelos ~700 iters. Jitado, compila 1x por variante
    # de opp (None=heuristica ou pytree=snapshot/self) e reusa.
    return jax.jit(rollout)


def gae(reward, value, live, last_val, gamma, lam):
    T = reward.shape[0]
    def back(carry, t):
        adv, nextv = carry
        v = value[t]
        nonterm = live[t]                       # se a transicao nao e valida, corta
        delta = reward[t] + gamma * nextv * nonterm - v
        adv = delta + gamma * lam * nonterm * adv
        return (adv, v), adv
    (_, _), advs = jax.lax.scan(back, (jnp.zeros_like(last_val), last_val),
                                jnp.arange(T), reverse=True)
    return advs, advs + value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--width", type=int, default=64,
                    help="largura d da rede (64=~32K params, 204=~300K DeepSets). Mudar "
                         "invalida checkpoints de outra largura (treina do zero).")
    ap.add_argument("--arch", choices=["deepsets", "transformer"], default="deepsets",
                    help="deepsets (padrao) ou transformer (self-attention entre planetas)")
    ap.add_argument("--heads", type=int, default=4, help="cabecas de atencao (transformer)")
    ap.add_argument("--tf-layers", type=int, default=2, help="blocos de transformer")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=4096,
                    help="tamanho do minibatch no update PPO (controla uso de VRAM; "
                         "reduza se OOM, aumente se GPU ociosa)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", action="store_true",
                    help="ativa WARMUP + COSINE no lr (playbook transformer-RL). Default off "
                         "= lr constante (compat com checkpoints antigos de lr fixo).")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="passos de WARMUP do lr (0=auto 3%% dos steps; >0 forca). "
                         "Apos warmup, cosine decay ate lr*0.1. So vale com --lr-schedule.")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01,
                    help="coef de entropia INICIAL (pointer). Decai linear ate --ent-end")
    ap.add_argument("--ent-end", type=float, default=-1.0,
                    help="coef de entropia FINAL (schedule). <0 = constante (=--ent)")
    ap.add_argument("--ent-frac", type=float, default=-1.0,
                    help="coef de entropia separado da cabeca FRAC (per-head). <0 = usa o mesmo do pointer")
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--shape", type=float, default=0.02)
    ap.add_argument("--shape-prod", type=float, default=0.0,
                    help="peso do shaping na vantagem de PRODUCAO (expansao); "
                         "recompensa capturar planetas. 0=desativado, 0.02 razoavel")
    ap.add_argument("--shape-planets", type=float, default=0.0,
                    help="peso do shaping na vantagem de N de PLANETAS (sinal de "
                         "expansao mais direto: +peso ao capturar). 0=desativado")
    ap.add_argument("--shape-static", type=float, default=0.0,
                    help="peso EXTRA do shaping na vantagem de planetas ESTATICOS "
                         "(quebra o vies de self-play onde home orbitante so ataca "
                         "orbitantes). Soma ao --shape-planets. 0.1-0.2 razoavel")
    ap.add_argument("--shape-prod-early", type=float, default=0.0,
                    help="peso EXTRA do shaping de PRODUCAO so nos turnos iniciais "
                         "(< --prod-early-until). Recompensa expandir cedo.")
    ap.add_argument("--prod-early-until", type=int, default=20,
                    help="ate que turno o --shape-prod-early vale (default 20)")
    ap.add_argument("--shape-diversity", type=float, default=0.0,
                    help="peso do shaping de DIVERSIDADE = min(meus orbitantes, meus "
                         "estaticos). Incentiva ter OS DOIS tipos (quebra vies de tipo).")
    ap.add_argument("--shape-defend", type=float, default=0.0,
                    help="peso da PUNICAO por perder planeta quando esta liderando "
                         "(prod > --lead-ratio x do inimigo). Incentiva consolidar a frente.")
    ap.add_argument("--lead-ratio", type=float, default=1.5,
                    help="razao de producao p/ considerar 'ganhando por muito' (--shape-defend)")
    ap.add_argument("--shape-capture", type=float, default=0.0,
                    help="bonus por captura CONCENTRADA (naves sobrando em planeta "
                         "recem-tomado). Privilegia tomar de uma vez c/ frota grande "
                         "(coalizao) vs por atrito. 0.01 razoavel")
    ap.add_argument("--shape-recapture", type=float, default=0.0,
                    help="bonus por capturar planetas VOLATEIS (trocaram de dono "
                         "recentemente): melhora retomada e, no 4p, aproveita guerra "
                         "entre inimigos (planeta disputado fica fragil). 0.05 razoavel")
    ap.add_argument("--shape-waste", type=float, default=0.0,
                    help="PUNICAO por naves mal usadas: frota que chega num planeta "
                         "alheio, e aniquilada e deixa o alvo com guarnicao residual "
                         "<= --waste-margin (faltou 1-2 pra tomar). Penaliza o "
                         "subcomprometimento/overcommit. 0.05-0.1 razoavel")
    ap.add_argument("--waste-margin", type=float, default=2.0,
                    help="residuo maximo (naves que faltaram) p/ contar como ataque "
                         "mal usado em --shape-waste. Default 2 = 'faltou 1 ou 2'.")
    ap.add_argument("--shape-idle", type=float, default=0.0,
                    help="PUNICAO por WASTE DE PRODUCAO: naves paradas acumuladas nos "
                         "nossos planetas (alem de --idle-margin) enquanto step < "
                         "--idle-until. Producao que nao virou expansao/frota cedo. "
                         "Empurra deploy agressivo no comeco. 0.02-0.05 razoavel")
    ap.add_argument("--idle-until", type=int, default=30,
                    help="ate que turno o --shape-idle vale (estagio inicial). Default 30")
    ap.add_argument("--idle-margin", type=float, default=0.0,
                    help="reserva por planeta isenta de punicao em --shape-idle (naves "
                         "que pode segurar sem contar como ocioso). Default 0")
    ap.add_argument("--fixed-opp", type=str, default="",
                    help="checkpoint extra (ex: ckpt_pool5.pkl) incluido como oponente "
                         "FIXO no pool (nunca descartado). Vazio=desativado")
    ap.add_argument("--fixed-opp-frac", type=float, default=0.2,
                    help="fracao das iters de pool (nao-heuristica) que usam o --fixed-opp")
    ap.add_argument("--selfplay", action="store_true")
    ap.add_argument("--opp", choices=["nearest", "smart", "pilkwang", "smart_aggressive",
                                      "smart_coalition", "sinkhorn"],
                    default="nearest",
                    help="oponente heuristico no rollout quando NAO e selfplay: "
                         "'nearest', 'smart', 'pilkwang' (OOM c/ B grande), "
                         "'smart_aggressive' (forte e leve), "
                         "'smart_coalition' (coalizao via leilao, ~LP do Dias), "
                         "'sinkhorn' (coalizao via transporte otimo regularizado)")
    ap.add_argument("--pool-size", type=int, default=0,
                    help="self-play com POOL de snapshots (>0 ativa, estilo liga do "
                         "AlphaStar): oponente = checkpoint aleatorio dos ultimos N. "
                         "Quebra o otimo local do self-play espelhado.")
    ap.add_argument("--pool-every", type=int, default=10,
                    help="adiciona os pesos atuais ao pool a cada N iters")
    ap.add_argument("--pool-heur-frac", type=float, default=0.25,
                    help="fracao das iters em que o oponente e a heuristica (--opp), "
                         "como ancora forte anti-esquecimento; resto = snapshot do pool")
    ap.add_argument("--pool-heur-sched", type=str, default="",
                    help="agenda o --pool-heur-frac por iteracao: breakpoints "
                         "'it:frac' separados por virgula, ex '200:0.1,300:0.0' "
                         "(antes do 1o bp usa --pool-heur-frac; a partir de cada it "
                         "aplica o frac dado). Util p/ desligar a ancora heuristica "
                         "apos o bootstrap no treino do zero. NB: o it reinicia a "
                         "cada processo (restart re-aplica o ramp inicial).")
    ap.add_argument("--pool-refresh", type=int, default=0,
                    help="re-gera o pool de mapas com seeds NOVAS a cada N iters "
                         "(0=nunca; elimina memorizacao das condicoes iniciais)")
    ap.add_argument("--eval-games", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-heur-every", type=int, default=0,
                    help="avalia contra heuristica real a cada N iters (0=desativado). "
                         "Ex: 50 -> avalia vs agent7 a cada 50 iters (~16 partidas, ~3min)")
    ap.add_argument("--eval-heur-games", type=int, default=16,
                    help="partidas por avaliacao vs heuristica (padrao 16)")
    ap.add_argument("--eval-heur-mode", choices=["agent7","starter","random","dias"],
                    default="agent7", help="oponente heuristico para avaliacao (motor real)")
    ap.add_argument("--save", default="", help="salva checkpoint (pesos+adam) aqui (a cada eval e no fim)")
    ap.add_argument("--load", default="", help="retoma de um checkpoint (pesos+adam)")
    ap.add_argument("--feat-inbound", action="store_true",
                    help="v3: anexa 2 features de frotas em voo (friend/enemy inbound) "
                         "-> 16 feats. Resolve overcommit. Exige ckpt 16-feat (warm-start).")
    ap.add_argument("--frac-pair", action="store_true",
                    help="v4: head de fracao condicionada no PAR (fonte,alvo) em vez de "
                         "so na fonte. Politica autoregressiva frac|destino. So transformer.")
    ap.add_argument("--feat-eta", action="store_true",
                    help="v5: bucketiza o inbound por ETA (chegando em <=6/<=18 turnos, "
                         "+4 colunas -> 20 feats). Exige --feat-inbound. Warm-start de "
                         "ckpt 16-feat expande o embed com linhas zero (sem perturbar).")
    ap.add_argument("--frac-cont", action="store_true",
                    help="fracao CONTINUA ~ Beta(a,b) em (0,1) em vez dos 4 bins. So "
                         "transformer. A head antiga [d,4] nao e aproveitavel -> init "
                         "fresca zero (prior Beta(1.7,1.7) ~ centrado em 0.5).")
    ap.add_argument("--final-ln", action="store_true",
                    help="LayerNorm de saida apos os blocos de attention (padrao pre-LN). "
                         "PERTURBA warm-start de ckpt sem ln_f (precisa de iters p/ readaptar).")
    ap.add_argument("--value-head", action="store_true",
                    help="value head MLP (2 camadas, ve tambem max-pool dos planetas) somada "
                         "ao head linear. Zero-init: warm-start do critico intacto.")
    ap.add_argument("--midgame-frac", type=float, default=0.0,
                    help="fracao dos jogos do rollout que COMECAM de estados de MEIO DE "
                         "JOGO (last_st de rollouts anteriores) em vez de step=0. Com "
                         "horizon<500 e a unica forma de treinar o meio/fim de jogo e de "
                         "ver o reward terminal com frequencia. 0.5 razoavel")
    ap.add_argument("--both-frac", type=float, default=0.0,
                    help="fracao das iters (pool/selfplay) que rodam self-play PURO contra "
                         "os pesos ATUAIS coletando os DOIS lados (2x amostras de graca). "
                         "0.25 razoavel")
    ap.add_argument("--pfsp", action="store_true",
                    help="amostra o oponente do pool por DIFICULDADE (peso (1-wr)^2+0.05, "
                         "wr = EMA do win-rate contra aquele snapshot) em vez de uniforme "
                         "(PFSP, estilo AlphaStar)")
    ap.add_argument("--archive-every", type=int, default=0,
                    help="a cada N iters adiciona um snapshot PERMANENTE ao archive de "
                         "oponentes (nunca descartado; anti-esquecimento sem heuristica). "
                         "0=desativado")
    ap.add_argument("--reset-opt", action="store_true",
                    help="ignora o opt_state do checkpoint carregado (re-inicia Adam + "
                         "lr-schedule). Use ao retreinar com --lr-schedule: o count do "
                         "Adam carregado avanca o cosine p/ o fim (lr fica no end_value)")
    args = ap.parse_args()
    if args.feat_inbound:
        global _FEAT_INBOUND
        _FEAT_INBOUND = True
        print("FEAT v3: inbound ligado -> 16 features por planeta (friend/enemy em voo)", flush=True)
    if args.feat_eta:
        if not args.feat_inbound:
            ap.error("--feat-eta exige --feat-inbound")
        global _FEAT_ETA
        _FEAT_ETA = True
        print(f"FEAT v5: eta buckets {ETA_BUCKETS} -> 20 features por planeta", flush=True)

    # Device: o codigo e agnostico; o JAX usa a GPU automaticamente se o
    # jax[cuda12] estiver instalado (ver requirements-gpu.txt / WSL no README).
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}", flush=True)

    if args.arch == "transformer":
        model = PolicyTF(d=args.width, heads=args.heads, layers=args.tf_layers,
                         frac_pair=args.frac_pair, final_ln=args.final_ln,
                         vhead=args.value_head, frac_cont=args.frac_cont)
    else:
        model = Policy(d=args.width)
    key = jax.random.PRNGKey(0)
    st0 = reset_batch(list(range(args.games)))
    feat, pm, glob = features(st0, 0)
    params = model.init(key, feat, pm, glob, pair_features(st0, 0))
    # lr: constante (default) ou WARMUP+COSINE (--lr-schedule, playbook transformer-RL).
    if args.lr_schedule:
        _nmb = max(1, (args.games * args.horizon) // args.minibatch) * args.epochs
        _total = max(1, args.iters * _nmb)
        _warm = args.warmup_steps if args.warmup_steps > 0 else max(1, int(0.03 * _total))
        lr_for_opt = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr, warmup_steps=_warm,
            decay_steps=_total, end_value=args.lr * 0.1)
        print(f"lr schedule: warmup {_warm} steps -> peak {args.lr:.1e} -> cosine ate "
              f"{args.lr*0.1:.1e} ({_total} steps totais)", flush=True)
    else:
        lr_for_opt = args.lr
    opt = optax.chain(optax.clip_by_global_norm(0.5),
                      optax.adamw(lr_for_opt, weight_decay=args.weight_decay))
    opt_state = opt.init(params)
    if args.load:
        loaded_params, loaded_opt = load_checkpoint(args.load)
        params, n_loaded, n_total = merge_params(params, loaded_params)
        if args.reset_opt or n_loaded != n_total or loaded_opt is None:
            opt_state = opt.init(params)
            why = "--reset-opt" if args.reset_opt else (
                "opt ausente" if loaded_opt is None else
                f"parcial ({n_loaded}/{n_total} tensores, {n_total - n_loaded} novos c/ init fresca)")
            print(f"checkpoint {args.load}: {n_loaded}/{n_total} tensores carregados; "
                  f"optimizer reiniciado ({why})", flush=True)
        else:
            opt_state = loaded_opt
            print(f"checkpoint completo (pesos+opt) carregado de {args.load}", flush=True)
            if args.lr_schedule:
                # o count do Adam vive no opt_state: warm-start AVANCA o cosine. Se o
                # count carregado ja passou de decay_steps, o lr fica preso no end_value.
                cnts = [int(x) for x in jax.tree_util.tree_leaves(loaded_opt)
                        if hasattr(x, "ndim") and x.ndim == 0 and
                        jnp.issubdtype(x.dtype, jnp.integer)]
                if cnts and max(cnts) >= _total:
                    print(f"AVISO: count do Adam carregado ({max(cnts)}) >= decay_steps "
                          f"({_total}): lr ficara em {args.lr*0.1:.1e} o treino todo. "
                          f"Use --reset-opt p/ recomecar o schedule.", flush=True)
    opp_fn = {"smart": smart_action, "pilkwang": pilkwang_action,
              "smart_aggressive": smart_aggressive, "smart_coalition": smart_coalition,
              "sinkhorn": sinkhorn_opp}.get(args.opp, greedy_action)
    rollout = make_rollout(model, args.horizon, args.shape, args.shape_prod,
                           args.shape_planets, args.shape_static,
                           args.shape_prod_early, args.prod_early_until,
                           args.shape_diversity, args.shape_defend, args.lead_ratio,
                           args.shape_capture, args.shape_recapture, opp_fn,
                           shape_waste_w=args.shape_waste, waste_margin=args.waste_margin,
                           shape_idle_w=args.shape_idle, idle_until=args.idle_until,
                           idle_margin=args.idle_margin)
    # variante que coleta os DOIS lados (self-play puro c/ pesos atuais)
    rollout_both = None
    if args.both_frac > 0.0:
        rollout_both = make_rollout(model, args.horizon, args.shape, args.shape_prod,
                                    args.shape_planets, args.shape_static,
                                    args.shape_prod_early, args.prod_early_until,
                                    args.shape_diversity, args.shape_defend, args.lead_ratio,
                                    args.shape_capture, args.shape_recapture, opp_fn,
                                    shape_waste_w=args.shape_waste, waste_margin=args.waste_margin,
                                    shape_idle_w=args.shape_idle, idle_until=args.idle_until,
                                    idle_margin=args.idle_margin, both_sides=True)

    # -- GAE em JAX (sem gradiente, escala bem) ---------------------------------
    @jax.jit
    def compute_gae(traj, last_val):
        adv, ret = gae(traj["reward"], traj["value"], traj["live"], last_val,
                       args.gamma, args.lam)
        live = traj["live"]
        # Explained variance do critico (so passos validos): 1 - Var(ret-v)/Var(ret).
        n = jnp.clip(live.sum(), 1.0)
        rmean = (ret * live).sum() / n
        ret_var = (((ret - rmean) ** 2) * live).sum() / n
        resid = ret - traj["value"]
        rsmean = (resid * live).sum() / n
        resid_var = (((resid - rsmean) ** 2) * live).sum() / n
        ev = 1.0 - resid_var / jnp.clip(ret_var, 1e-8)
        # normaliza a vantagem SO sobre passos validos (os zeros de passos mortos
        # encolhiam a media/std e inflavam a escala dos passos vivos)
        amean = (adv * live).sum() / n
        astd = jnp.sqrt((((adv - amean) ** 2) * live).sum() / n)
        adv = (adv - amean) / (astd + 1e-6) * live
        return adv, ret, ev

    # -- Passo de gradiente em UM minibatch (JIT compila uma vez para MB fixo) --
    # Recebe fatias ja indexadas — VRAM proporcional ao MB, nao a T*B inteiro.
    @jax.jit
    def mb_step(params, opt_state, stf, act, frac_act, oldlp, A, R, LV, ent_c, ent_c_f, inb):
        def loss_fn(p):
            lp, srcf, ent_t, ent_f, val = eval_logp(p, model, stf, 0, act, frac_act, inbound=inb)
            # PPO POR PLANETA: lp/oldlp sao [MB,P]; cada planeta-fonte e uma acao
            # com a vantagem do turno -> ratio/clip individuais (menos variancia
            # que o ratio conjunto, que era o produto de ate P razoes).
            w = srcf * LV[:, None]                       # [MB,P] so fontes de passos validos
            wsum = jnp.clip(w.sum(), 1.0)
            ratio = jnp.exp(lp - oldlp)
            Ab = A[:, None]
            s1 = ratio * Ab
            s2 = jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * Ab
            pol = -jnp.sum(jnp.minimum(s1, s2) * w) / wsum
            vloss = jnp.sum(((val - R) ** 2) * LV) / jnp.clip(LV.sum(), 1.0)
            denom = jnp.clip(LV.sum(), 1.0)
            ent_t_m = jnp.sum(ent_t * LV) / denom        # entropia media do pointer
            ent_f_m = jnp.sum(ent_f * LV) / denom        # entropia media do frac
            # per-head: coef proprio p/ cada cabeca (escalas de entropia diferentes)
            ent_bonus = ent_c * ent_t_m + ent_c_f * ent_f_m
            clipf = jnp.sum((jnp.abs(ratio - 1.0) > args.clip).astype(jnp.float32) * w) / wsum
            return pol + args.vf * vloss - ent_bonus, (pol, vloss, ent_t_m + ent_f_m, clipf)
        (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state2 = opt.update(g, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, aux

    # -- Update PPO com minibatches (loop Python) --------------------------------
    # CORRECAO DE OOM: antes era um lax.scan dentro de @jax.jit que materializava
    # gradientes sobre T*B inteiro (ex: 2048*128=262k amostras, ~6-9 GB de VRAM).
    # Agora o loop de epochs/minibatches roda em Python; cada mb_step processa so
    # `--minibatch` amostras (padrao 4096), mantendo uso de VRAM constante e baixo.
    def update(params, opt_state, traj, last_val, key, ent_c, ent_c_f):
        adv, ret, ev = compute_gae(traj, last_val)
        T, B = traj["reward"].shape
        N = T * B
        flat = lambda a: a.reshape((N,) + a.shape[2:])

        # Flattena SO os campos de planeta para [N, ...] na GPU uma unica vez.
        # Os campos de frota ficam zerados (a politica nao os le); antes eram
        # materializados como [N, F_MAX] (~2.8 GB de zeros mortos), o que
        # fragmentava a VRAM e causava OOM apos ~200 iters. Agora criamos os
        # zeros no tamanho do minibatch ([MB, F_MAX], 16x menor) e reusamos.
        pv_f=flat(traj["p_valid"]); po_f=flat(traj["p_owner"])
        px_f=flat(traj["p_x"]);     py_f=flat(traj["p_y"]);   pr_f=flat(traj["p_r"])
        psh_f=flat(traj["p_ships"]); ppr_f=flat(traj["p_prod"]); por_f=flat(traj["p_orbit"])
        om_f=flat(traj["omega"]);   stp_f=flat(traj["step"])
        act_f   = flat(traj["action"])
        fa_f    = flat(traj["frac_action"])
        oldlp_f = flat(traj["logp"])
        inb_f   = flat(traj["inbound"])     # [N,P,2] inbound salvo no rollout (estado real)
        A_f     = flat(adv)
        R_f     = flat(ret)
        LV_f    = flat(traj["live"])

        MB = args.minibatch
        zb = jnp.zeros((MB, J.F_MAX), bool)
        zi = jnp.zeros((MB, J.F_MAX), jnp.int32)
        zf = jnp.zeros((MB, J.F_MAX), jnp.float32)
        stats = (0.0, 0.0, 0.0, 0.0)
        # Permutacao e fatias no DEVICE (antes era np.random.permutation no host +
        # jnp.asarray por minibatch -> ~25 MB/iter de churn de RAM que vazava e
        # causava OOM de host lá pelos ~700 iters). Agora zero alocacao de host aqui.
        for _ in range(args.epochs):
            key, ke = jax.random.split(key)
            perm = jax.random.permutation(ke, N)
            for start in range(0, N - MB + 1, MB):
                idx = perm[start : start + MB]
                mb_st = J.State(
                    p_valid=pv_f[idx], p_owner=po_f[idx], p_x=px_f[idx], p_y=py_f[idx],
                    p_r=pr_f[idx], p_ships=psh_f[idx], p_prod=ppr_f[idx], p_orbit=por_f[idx],
                    f_active=zb, f_owner=zi, f_x=zf, f_y=zf, f_ang=zf, f_ships=zf,
                    omega=om_f[idx], step=stp_f[idx])
                params, opt_state, stats = mb_step(
                    params, opt_state,
                    mb_st, act_f[idx], fa_f[idx], oldlp_f[idx], A_f[idx], R_f[idx], LV_f[idx],
                    ent_c, ent_c_f, inb_f[idx])
        return params, opt_state, (stats[0], stats[1], stats[2], stats[3], ev)

    @partial(jax.jit, static_argnames=("opp_fn",))
    def eval_winrate(params, st, opp_fn=greedy_action):
        def stepf(carry, _):
            st, done_acc = carry
            _, _, _, _, ang0, sh0, _, _ = policy_act(params, model, st, 0,
                                                     jax.random.PRNGKey(0), sample=False)
            ang1, sh1 = opp_fn(st, 1)
            nst, done, _, _ = step_batched(st, ang0, sh0, ang1, sh1)
            return (nst, done_acc | done), None
        (last, _), _ = jax.lax.scan(stepf, (st, jnp.zeros(st.p_owner.shape[0], bool)),
                                    None, length=J.EPISODE_STEPS)
        my, en = totals0(last)
        return jnp.mean((my > en).astype(jnp.float32))

    # Pool de mapas gerado UMA vez no host; cada iteracao amostra no device.
    # Reduzido de games*8 para games*4 para economizar RAM e VRAM.
    pool_n = max(256, args.games * 4)
    pool_seed0 = 20000
    pool_box = [reset_batch(list(range(pool_seed0, pool_seed0 + pool_n)))]
    ev = reset_batch(list(range(5000, 5000 + args.eval_games)))

    # Re-geracao do pool de mapas em BACKGROUND (reset_batch(4096)~50s): uma thread
    # gera o proximo pool enquanto a GPU treina; o main troca quando pronto (zero stall).
    _bg = {"thread": None, "result": None}
    def _bg_gen(seed0):
        pl = reset_batch(list(range(seed0, seed0 + pool_n)))
        jax.block_until_ready(pl.p_x)        # forca o device transfer na thread
        _bg["result"] = pl

    # RANDOM-START de meio de jogo: o last_st de cada rollout vira candidato a
    # estado inicial do proximo (jogos terminados sao trocados pelo start). Sem
    # isso, com horizon<500 a politica NUNCA treinava nos turnos [horizon,500)
    # nem via o reward terminal (so eliminacao precoce dava sinal).
    mid_box = [None]

    def store_midgame(st_start, last_st):
        p0a = (jnp.any(last_st.p_valid & (last_st.p_owner == 0), -1)
               | jnp.any(last_st.f_active & (last_st.f_owner == 0), -1))
        p1a = (jnp.any(last_st.p_valid & (last_st.p_owner == 1), -1)
               | jnp.any(last_st.f_active & (last_st.f_owner == 1), -1))
        fin = (last_st.step >= J.EPISODE_STEPS - 1) | (~p0a) | (~p1a)
        sel = lambda a, b: jnp.where(fin.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        mid_box[0] = jax.tree_util.tree_map(sel, st_start, last_st)

    def sample_states(k):
        k1, k2, k3 = jax.random.split(k, 3)
        idx = jax.random.randint(k1, (args.games,), 0, pool_n)
        base = jax.tree_util.tree_map(lambda a: a[idx], pool_box[0])
        if args.midgame_frac <= 0.0 or mid_box[0] is None:
            return base
        mid = mid_box[0]
        j = jax.random.randint(k2, (args.games,), 0, mid.p_x.shape[0])
        midsel = jax.tree_util.tree_map(lambda a: a[j], mid)
        use = jax.random.uniform(k3, (args.games,)) < args.midgame_frac
        pick = lambda b, m: jnp.where(use.reshape((-1,) + (1,) * (b.ndim - 1)), m, b)
        return jax.tree_util.tree_map(pick, base, midsel)

    # Pool de snapshots p/ self-play estilo liga (opcional, --pool-size>0).
    # Cada entrada = {p: params, wr: EMA do win-rate do LEARNER contra ela}
    # (wr alimenta o PFSP). O archive guarda snapshots permanentes espacados.
    use_pool = args.pool_size > 0
    snap_pool = [{"p": params, "wr": 0.5}] if use_pool else []
    snap_archive = []

    # Oponentes FIXOS (ex: pool10,pool7): 1+ checkpoints congelados, nunca descartados.
    # Aceita lista separada por virgula; todos devem ter a MESMA arquitetura.
    fixed_opps = []
    fixed_names = []
    rollout_fixed = rollout            # por padrao usa o mesmo rollout (mesma largura)
    opp_model = model                  # rede do oponente fixo (default = mesma)
    opp_legacy = False                 # oponente usa as 11 features antigas?
    opp_nfeat = None                   # oponente com MENOS colunas (corta as extras)
    cur_pf = PF + (2 if args.feat_inbound else 0) + (4 if args.feat_eta else 0)
    if args.fixed_opp:
        for _p in args.fixed_opp.split(","):
            _p = _p.strip()
            if not _p:
                continue
            _fo, _ = load_checkpoint(_p)
            fixed_opps.append(_fo); fixed_names.append(_p)
        fo_flat = dict(flax.traverse_util.flatten_dict(fixed_opps[0], sep="/"))
        d_opp = int(fo_flat["params/Dense_0/kernel"].shape[1])
        pf_opp = int(fo_flat["params/Dense_0/kernel"].shape[0])  # 11=legacy, 14=novo
        opp_legacy = (pf_opp == 11)
        # detecta a ARQUITETURA do ckpt fixo pelos nomes dos tensores: um ckpt de
        # 2 camadas sem extras precisa de um PolicyTF proprio (apply exige a
        # arvore de params exata; usar o model novo quebraria/embaralharia).
        fo_keys = set(fo_flat.keys())
        fo_is_tf = any("MultiHeadDotProductAttention_0" in k for k in fo_keys)
        if not opp_legacy and pf_opp != cur_pf:
            opp_nfeat = pf_opp         # ckpt fixo de antes do --feat-eta: corta colunas
        if fo_is_tf and not opp_legacy:
            fo_layers = 1 + max(int(k.split("MultiHeadDotProductAttention_")[1].split("/")[0])
                                for k in fo_keys if "MultiHeadDotProductAttention_" in k)
            _fo_fh = fo_flat.get("params/Dense_15/kernel")
            fo_cfg = dict(d=d_opp, heads=args.heads, layers=fo_layers,
                          frac_pair=any("/fk/" in k for k in fo_keys),
                          final_ln=any("/ln_f/" in k for k in fo_keys),
                          vhead=any("/vh1/" in k for k in fo_keys),
                          frac_cont=(_fo_fh is not None and _fo_fh.shape[-1] == 2))
            my_cfg = dict(d=args.width, heads=args.heads, layers=args.tf_layers,
                          frac_pair=args.frac_pair, final_ln=args.final_ln,
                          vhead=args.value_head, frac_cont=args.frac_cont)
            if fo_cfg != my_cfg:
                opp_model = PolicyTF(**fo_cfg)
        elif d_opp != args.width or opp_legacy:
            opp_model = Policy(d=d_opp)
        if opp_model is not model or opp_nfeat is not None:
            rollout_fixed = make_rollout(model, args.horizon, args.shape, args.shape_prod,
                                         args.shape_planets, args.shape_static,
                                         args.shape_prod_early, args.prod_early_until,
                                         args.shape_diversity, args.shape_defend, args.lead_ratio,
                                         args.shape_capture, args.shape_recapture, opp_fn,
                                         opp_model=opp_model, opp_legacy=opp_legacy,
                                         shape_waste_w=args.shape_waste, waste_margin=args.waste_margin,
                                         shape_idle_w=args.shape_idle, idle_until=args.idle_until,
                                         idle_margin=args.idle_margin, opp_nfeat=opp_nfeat)
        print(f"{len(fixed_opps)} oponente(s) fixo(s) {fixed_names}: d={d_opp} feats={pf_opp} "
              f"(learner={cur_pf}) legacy={opp_legacy} arch_propria={opp_model is not model} "
              f"(total {args.fixed_opp_frac:.0%} das iters de pool)", flush=True)

    # eval da politica vs o oponente fixo (ex: pool10), no jax_env (perspectiva flip 180)
    @jax.jit
    def eval_vs_fixed(params, fopp, st):
        def stepf(carry, _):
            st, done = carry
            _, _, _, _, a0, s0, _, _ = policy_act(params, model, st, 0,
                                                  jax.random.PRNGKey(0), sample=False)
            fst = flip_state(st)
            _, _, _, _, a1, s1, _, _ = policy_act(fopp, opp_model, fst, 0,
                                                  jax.random.PRNGKey(0), sample=False,
                                                  legacy=opp_legacy, nfeat=opp_nfeat)
            a1 = a1 + jnp.pi
            nst, d, _, _ = step_batched(st, a0, s0, a1, s1)
            return (nst, done | d), None
        (last, _), _ = jax.lax.scan(stepf, (st, jnp.zeros(st.p_owner.shape[0], bool)),
                                    None, length=J.EPISODE_STEPS)
        my, en = totals0(last)
        return jnp.mean((my > en).astype(jnp.float32))

    if use_pool:
        opp_label = f"pool[{args.pool_size}] +{args.pool_heur_frac:.0%}{args.opp}"
        if args.pfsp:
            opp_label += " pfsp"
        if args.archive_every > 0:
            opp_label += f" arch/{args.archive_every}"
    elif args.selfplay:
        opp_label = "self"
    else:
        opp_label = args.opp
    if args.both_frac > 0:
        opp_label += f" both={args.both_frac:.0%}"
    if args.midgame_frac > 0:
        opp_label += f" mid={args.midgame_frac:.0%}"
    _refresh_lbl = f", refresh c/seeds novas a cada {args.pool_refresh} it" if args.pool_refresh > 0 else " fixos"
    print(f"PPO-JAX | games={args.games} horizon={args.horizon} iters={args.iters} "
          f"opp={opp_label} mb={args.minibatch} (pool={pool_n} mapas{_refresh_lbl})", flush=True)
    # agenda do pool-heur-frac por iteracao (breakpoints 'it:frac')
    _heur_sched = []
    if args.pool_heur_sched:
        for tok in args.pool_heur_sched.split(","):
            a, b = tok.split(":")
            _heur_sched.append((int(a), float(b)))
        _heur_sched.sort()
        print(f"pool-heur-frac sched: base {args.pool_heur_frac:.2f} -> "
              + " -> ".join(f"it{t}:{v:.2f}" for t, v in _heur_sched), flush=True)
    def heur_frac_at(it):
        f = args.pool_heur_frac
        for thr, val in _heur_sched:
            if it >= thr:
                f = val
        return f

    t0 = None
    for it in range(1, args.iters + 1):
        key, kr, ks, ku = jax.random.split(key, 4)
        if args.pool_refresh > 0:
            # dispara a geracao do proximo pool (seeds ineditas) no intervalo
            if it % args.pool_refresh == 0 and _bg["thread"] is None:
                seed0 = pool_seed0 + (it // args.pool_refresh + 1) * pool_n
                _bg["thread"] = threading.Thread(target=_bg_gen, args=(seed0,), daemon=True)
                _bg["thread"].start()
            # quando a thread terminar (alguns iters depois), troca o pool
            if _bg["thread"] is not None and not _bg["thread"].is_alive():
                pool_box[0] = _bg["result"]
                _bg["thread"] = None; _bg["result"] = None
        st = sample_states(ks)
        rl = rollout
        pool_entry = None      # entrada do pool/archive p/ atualizar o wr (PFSP)
        if use_pool:
            # ordem: ancora heuristica > self-play both-sides > oponente fixo >
            # snapshot do pool/archive (PFSP ou uniforme)
            r_draw = np.random.rand()
            if r_draw < heur_frac_at(it):
                opp = None
            elif rollout_both is not None and np.random.rand() < args.both_frac:
                opp = params; rl = rollout_both
            elif fixed_opps and np.random.rand() < args.fixed_opp_frac:
                opp = fixed_opps[np.random.randint(len(fixed_opps))]; rl = rollout_fixed
            else:
                entries = snap_pool + snap_archive
                if args.pfsp:
                    w = np.array([(1.0 - e["wr"]) ** 2 + 0.05 for e in entries])
                    i_e = int(np.random.choice(len(entries), p=w / w.sum()))
                else:
                    i_e = np.random.randint(len(entries))
                pool_entry = entries[i_e]
                opp = pool_entry["p"]
        elif args.selfplay:
            opp = params
            if rollout_both is not None:
                rl = rollout_both
        else:
            opp = None
        traj, last_val, last_st = rl(params, opp, st, kr)
        if args.midgame_frac > 0.0:
            store_midgame(st, last_st)
        if pool_entry is not None:
            # win-rate medido no fim do rollout (proxy: placar em last_st)
            my_s, en_s = totals0(last_st)
            score = float(jnp.mean((my_s > en_s) + 0.5 * (my_s == en_s)))
            pool_entry["wr"] = 0.9 * pool_entry["wr"] + 0.1 * score
        # entropy schedule: decai linear de --ent ate --ent-end ao longo do treino.
        # per-head: cabeca frac usa --ent-frac (se dado), senao o mesmo do pointer.
        frac_t = (it - 1) / max(1, args.iters - 1)
        ent_c = args.ent if args.ent_end < 0 else args.ent + (args.ent_end - args.ent) * frac_t
        ent_c_f = ent_c if args.ent_frac < 0 else args.ent_frac
        params, opt_state, stats = update(params, opt_state, traj, last_val, ku, ent_c, ent_c_f)
        if use_pool and it % args.pool_every == 0:
            snap_pool.append({"p": params, "wr": 0.5})  # arrays jax sao imutaveis: snapshot fiel
            if len(snap_pool) > args.pool_size:
                snap_pool.pop(0)                # descarta o mais antigo
        if use_pool and args.archive_every > 0 and it % args.archive_every == 0:
            snap_archive.append({"p": params, "wr": 0.5})   # permanente (anti-esquecimento)
        if it == 1:                       # 1a iter compila o JIT; cronometra dali
            jax.block_until_ready(params); t0 = time.time(); done_steps = 0
        else:
            done_steps += args.games * args.horizon
        rmean = float((traj["reward"] * traj["live"]).sum() / jnp.clip(traj["live"].sum(), 1.0))
        wmean = float((traj["waste"] * traj["live"]).sum() / jnp.clip(traj["live"].sum(), 1.0))
        imean = float((traj["idle"] * traj["live"]).sum() / jnp.clip(traj["live"].sum(), 1.0))
        pol, vloss, ent, clipf, ev_v = [float(x) for x in stats]
        msg = (f"it {it:3d} | r/step={rmean:+.4f} pol={pol:+.3f} v={vloss:.3f} "
               f"ev={ev_v:+.3f} ent={ent:.3f} clipf={clipf:.3f} waste={wmean:.2f} idle={imean:.1f}")
        if it % args.eval_every == 0:
            wr_g = float(eval_winrate(params, ev, greedy_action))
            wr_s = float(eval_winrate(params, ev, smart_action))
            wr_p = float(eval_winrate(params, ev, pilkwang_action))
            sps = done_steps / (time.time() - t0) if t0 and done_steps else 0.0
            msg += f" | WR g={wr_g:.3f} s={wr_s:.3f} p={wr_p:.3f}"
            for _fo, _nm in zip(fixed_opps, fixed_names):
                wr_fx = float(eval_vs_fixed(params, _fo, ev))
                tag = _nm.replace("ckpt_", "").replace(".pkl", "")
                msg += f" vs_{tag}={wr_fx:.3f}"
            if use_pool and (snap_pool or snap_archive):
                _wrs = [e["wr"] for e in snap_pool + snap_archive]
                msg += f" poolwr={min(_wrs):.2f}/{float(np.median(_wrs)):.2f}"
            msg += f"  ({sps:,.0f} SPS)"
            if args.save:
                save_checkpoint(args.save, params, opt_state)
        if args.eval_heur_every and it % args.eval_heur_every == 0:
            t_heur = time.time()
            wr_h = eval_vs_heuristic(params, model, args.eval_heur_games,
                                     mode=args.eval_heur_mode)
            msg += f" | vs {args.eval_heur_mode}={wr_h:.3f} ({time.time()-t_heur:.0f}s)"
        print(msg, flush=True)
    if args.save:
        save_checkpoint(args.save, params, opt_state)
        print(f"checkpoint final salvo em {args.save}", flush=True)
    print("fim.")


if __name__ == "__main__":
    main()
