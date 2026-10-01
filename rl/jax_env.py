"""Motor Orbit Wars em JAX -- rapido (jit + vmap, batched), para RL.

Versao 2 JOGADORES, SEM COMETAS (cometas exigem rejection-sampling de orbitas,
inviavel dentro do JIT; o motor real continua usado para avaliacao/replay e o
deploy). Espelha a fisica do interpreter de kaggle_environments: launch ->
producao -> rotacao -> movimento com colisao swept (continua) -> combate.

Estado = pytree de arrays de shape FIXO (pools de planetas/frotas + mascaras),
o que permite jit/vmap. `reset_*` roda no host (numpy) reaproveitando o gerador
de planetas do motor real para PARIDADE de mapa. So o `step` precisa ser rapido.

Acao por jogador: arrays [P_MAX] -> ang[i], ships[i]. ships[i]==0 => planeta i
nao lanca (no maximo um lancamento por planeta/turno, casando com a politica
pointer). Lancamentos invalidos (planeta nao e do jogador / sem naves) sao
ignorados.

CLI:
    python3 jax_env.py parity      # compara com o motor real (pre-cometa)
    python3 jax_env.py bench       # mede SPS (steps/seg) batched
"""
import math
import os
import sys
import time
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

CENTER = 50.0
SUN_R = 10.0
ROT_LIMIT = 50.0
BOARD = 100.0
MAX_SPEED = 6.0
P_MAX = 48          # capacidade de planetas (mapas tem 20-40)
F_MAX = 256         # capacidade do pool de frotas
EPISODE_STEPS = 500
LOG1000 = math.log(1000.0)


class State(NamedTuple):
    p_valid: jnp.ndarray   # [P] bool
    p_owner: jnp.ndarray   # [P] int32  (-1 neutro, 0/1 jogadores)
    p_x: jnp.ndarray       # [P] f32
    p_y: jnp.ndarray
    p_r: jnp.ndarray
    p_ships: jnp.ndarray
    p_prod: jnp.ndarray
    p_orbit: jnp.ndarray   # [P] bool (gira em torno do sol)
    f_active: jnp.ndarray  # [F] bool
    f_owner: jnp.ndarray   # [F] int32
    f_x: jnp.ndarray
    f_y: jnp.ndarray
    f_ang: jnp.ndarray
    f_ships: jnp.ndarray
    omega: jnp.ndarray     # escalar f32
    step: jnp.ndarray      # escalar int32


# ----------------------------------------------------------------------------
# Host reset (numpy) -- reaproveita o gerador do motor real p/ mesmo mapa.
# ----------------------------------------------------------------------------
def _engine_module():
    real = os.listdir
    os.listdir = lambda p: (["orbit_wars"] if "orbit_wars" in real(p) else real(p))
    try:
        from kaggle_environments.envs.orbit_wars import orbit_wars as ow
    finally:
        os.listdir = real
    return ow


def reset_single_np(seed):
    """Replica a ordem de RNG do interpreter real p/ mapa/omega/homes identicos."""
    import random
    ow = _engine_module()
    rng = random.Random(int(seed))
    omega = rng.uniform(0.025, 0.05)            # 1o draw (igual ao motor)
    planets = ow.generate_planets(rng)          # 2o
    ngroups = len(planets) // 4
    if ngroups > 0:                             # 3o: home group
        base = rng.randint(0, ngroups - 1) * 4
        planets[base][1] = 0; planets[base][5] = 10
        planets[base + 3][1] = 1; planets[base + 3][5] = 10

    n = len(planets)
    pv = np.zeros(P_MAX, bool); po = np.full(P_MAX, -1, np.int32)
    px = np.zeros(P_MAX, np.float32); py = np.zeros(P_MAX, np.float32)
    pr = np.zeros(P_MAX, np.float32); ps = np.zeros(P_MAX, np.float32)
    pp = np.zeros(P_MAX, np.float32); porb = np.zeros(P_MAX, bool)
    for i, pl in enumerate(planets[:P_MAX]):
        _id, owner, x, y, r, ships, prod = pl
        pv[i] = True; po[i] = owner; px[i] = x; py[i] = y
        pr[i] = r; ps[i] = ships; pp[i] = prod
        porb[i] = (math.hypot(x - CENTER, y - CENTER) + r) < ROT_LIMIT
    return dict(p_valid=pv, p_owner=po, p_x=px, p_y=py, p_r=pr, p_ships=ps,
                p_prod=pp, p_orbit=porb, omega=np.float32(omega))


def reset_batch(seeds):
    """Empilha varios resets num State batched (leading dim = batch)."""
    singles = [reset_single_np(s) for s in seeds]
    B = len(seeds)
    def st(key, dt):
        return jnp.asarray(np.stack([s[key] for s in singles]).astype(dt))
    f0b = lambda shp, dt: jnp.zeros((B,) + shp, dt)
    return State(
        p_valid=st("p_valid", bool), p_owner=st("p_owner", np.int32),
        p_x=st("p_x", np.float32), p_y=st("p_y", np.float32),
        p_r=st("p_r", np.float32), p_ships=st("p_ships", np.float32),
        p_prod=st("p_prod", np.float32), p_orbit=st("p_orbit", bool),
        f_active=f0b((F_MAX,), bool), f_owner=f0b((F_MAX,), np.int32),
        f_x=f0b((F_MAX,), np.float32), f_y=f0b((F_MAX,), np.float32),
        f_ang=f0b((F_MAX,), np.float32), f_ships=f0b((F_MAX,), np.float32),
        omega=st("omega", np.float32), step=jnp.zeros((B,), np.int32),
    )


# ----------------------------------------------------------------------------
# Fisica (single-game; vmap/jit aplicados depois)
# ----------------------------------------------------------------------------
def _fleet_speed(ships):
    s = 1.0 + (MAX_SPEED - 1.0) * (jnp.log(jnp.maximum(ships, 1.0)) / LOG1000) ** 1.5
    return jnp.minimum(s, MAX_SPEED)


def _seg_dist_to_center(ax, ay, bx, by):
    """Distancia do sol (CENTER,CENTER) ao segmento (ax,ay)-(bx,by). [F]"""
    vx, vy = ax - bx, ay - by
    l2 = vx * vx + vy * vy
    t = ((CENTER - ax) * (bx - ax) + (CENTER - ay) * (by - ay)) / jnp.where(l2 == 0, 1.0, l2)
    t = jnp.clip(t, 0.0, 1.0)
    t = jnp.where(l2 == 0, 0.0, t)
    projx = ax + t * (bx - ax); projy = ay + t * (by - ay)
    return jnp.hypot(CENTER - projx, CENTER - projy)


def step_single(state: State, a0_ang, a0_ships, a1_ang, a1_ships):
    P = state.p_owner
    px, py, pr, ps = state.p_x, state.p_y, state.p_r, state.p_ships
    pv, porb = state.p_valid, state.p_orbit
    owner = P

    # --- 1) LAUNCH: no maximo 1 por planeta (o dono). ---
    req_ships = jnp.where(owner == 0, a0_ships, jnp.where(owner == 1, a1_ships, 0.0))
    req_ang = jnp.where(owner == 0, a0_ang, a1_ang)
    launch_ships = jnp.clip(jnp.floor(req_ships), 0.0, ps)
    launch = pv & (owner >= 0) & (launch_ships >= 1.0)
    launch_ships = jnp.where(launch, launch_ships, 0.0)
    ps = ps - launch_ships
    spawn_x = px + jnp.cos(req_ang) * (pr + 0.1)
    spawn_y = py + jnp.sin(req_ang) * (pr + 0.1)

    # alocar lancamentos nos slots inativos do pool de frotas
    fa = state.f_active
    inactive = ~fa
    inactive_rank = jnp.cumsum(inactive.astype(jnp.int32)) - 1          # [F]
    # slot_of_rank[r] = indice do r-esimo slot inativo
    slot_of_rank = jnp.zeros(F_MAX, jnp.int32).at[
        jnp.where(inactive, inactive_rank, F_MAX - 1)].set(
        jnp.where(inactive, jnp.arange(F_MAX), 0))
    launch_rank = jnp.cumsum(launch.astype(jnp.int32)) - 1              # [P]
    n_free = jnp.sum(inactive)
    dest = slot_of_rank[jnp.clip(launch_rank, 0, F_MAX - 1)]            # [P] slot p/ cada planeta
    can = launch & (launch_rank < n_free)                              # cabe no pool?

    def scatter(arr, vals, default_for_overflow=None):
        return arr.at[jnp.where(can, dest, F_MAX - 1)].set(
            jnp.where(can, vals, arr[jnp.where(can, dest, F_MAX - 1)]))
    # escreve nos slots destino (apenas onde can)
    sel = jnp.where(can, dest, F_MAX - 1)
    fa = fa.at[sel].set(jnp.where(can, True, fa[sel]))
    fo = state.f_owner.at[sel].set(jnp.where(can, owner, state.f_owner[sel]))
    fx = state.f_x.at[sel].set(jnp.where(can, spawn_x, state.f_x[sel]))
    fy = state.f_y.at[sel].set(jnp.where(can, spawn_y, state.f_y[sel]))
    fang = state.f_ang.at[sel].set(jnp.where(can, req_ang, state.f_ang[sel]))
    fsh = state.f_ships.at[sel].set(jnp.where(can, launch_ships, state.f_ships[sel]))

    # --- 2) PRODUCAO ---
    ps = ps + jnp.where(pv & (owner >= 0), state.p_prod, 0.0)

    # --- 3) ROTACAO dos planetas (old -> new) ---
    # O motor real NAO rotaciona no 1o passo (usa angulo init+omega*step, step=0).
    # Gateamos em step>=1 p/ casar exatamente.
    ca, sa = jnp.cos(state.omega), jnp.sin(state.omega)
    dx, dy = px - CENTER, py - CENTER
    rx = CENTER + dx * ca - dy * sa
    ry = CENTER + dx * sa + dy * ca
    rot = porb & (state.step >= 1)
    p_old_x, p_old_y = px, py
    p_new_x = jnp.where(rot, rx, px)
    p_new_y = jnp.where(rot, ry, py)

    # --- 4) MOVIMENTO das frotas + colisao swept ---
    sp = _fleet_speed(fsh)
    f_old_x, f_old_y = fx, fy
    f_new_x = fx + jnp.cos(fang) * sp
    f_new_y = fy + jnp.sin(fang) * sp

    # swept-pair hit [F,P]
    A = jnp.stack([f_old_x, f_old_y], -1)[:, None, :]        # [F,1,2]
    Bp = jnp.stack([f_new_x, f_new_y], -1)[:, None, :]
    P0 = jnp.stack([p_old_x, p_old_y], -1)[None, :, :]       # [1,P,2]
    P1 = jnp.stack([p_new_x, p_new_y], -1)[None, :, :]
    d0 = A - P0                                              # [F,P,2]
    dv = (Bp - A) - (P1 - P0)
    aa = jnp.sum(dv * dv, -1)
    bb = 2.0 * jnp.sum(d0 * dv, -1)
    cc = jnp.sum(d0 * d0, -1) - (pr ** 2)[None, :]
    disc = bb * bb - 4.0 * aa * cc
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-bb - sq) / (2.0 * jnp.maximum(aa, 1e-12))
    t2 = (-bb + sq) / (2.0 * jnp.maximum(aa, 1e-12))
    hit_quad = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    hit_lin = cc <= 0.0
    hit = jnp.where(aa < 1e-12, hit_lin, hit_quad)
    hit = hit & pv[None, :] & fa[:, None]                   # so frotas ativas x planetas validos
    hit_any = jnp.any(hit, -1)                              # [F]
    hit_idx = jnp.argmax(hit.astype(jnp.int32), -1)         # 1o planeta na ordem (= motor)

    oob = (f_new_x < 0) | (f_new_x > BOARD) | (f_new_y < 0) | (f_new_y > BOARD)
    sun = _seg_dist_to_center(f_old_x, f_old_y, f_new_x, f_new_y) < SUN_R
    removed = fa & (hit_any | oob | sun)                    # somem do pool
    survives = fa & ~removed
    fx = jnp.where(survives, f_new_x, fx)
    fy = jnp.where(survives, f_new_y, fy)

    # --- 5) aplica movimento dos planetas ---
    px, py = p_new_x, p_new_y

    # --- 6) COMBATE (2 jogadores) ---
    contrib = fa & hit_any
    s0 = jnp.zeros(P_MAX).at[hit_idx].add(jnp.where(contrib & (fo == 0), fsh, 0.0))
    s1 = jnp.zeros(P_MAX).at[hit_idx].add(jnp.where(contrib & (fo == 1), fsh, 0.0))
    top = jnp.maximum(s0, s1)
    second = jnp.minimum(s0, s1)
    surv = top - second
    surv_owner = jnp.where(s0 > s1, 0, jnp.where(s1 > s0, 1, -1))
    has = (surv > 0.0) & pv
    same = owner == surv_owner
    diff = ps - surv
    flip = diff < 0.0
    new_owner = jnp.where(has & (~same) & flip, surv_owner, owner)
    new_ships = jnp.where(
        has,
        jnp.where(same, ps + surv, jnp.where(flip, -diff, diff)),
        ps)
    # --- NAVES MAL USADAS (player 0): ataque que chega num planeta alheio,
    # vence o duelo de frotas (surv_owner==0) mas NAO toma o planeta (flip falso),
    # deixando uma guarnicao residual `shortfall = ps - surv`. Quando esse residuo
    # e pequeno (1-2), gastamos a frota inteira por pouco -> frota desperdicada.
    # Exportamos `waste0[p] = surv` (naves nossas aniquiladas) e o residuo p/ que o
    # treino possa punir o subcomprometimento. Calculado aqui pq s0/surv/ps so
    # existem dentro do combate.
    we_dom = (surv_owner == 0) & has & (s0 > 0.0)      # nossa frota dominou o duelo
    foreign = pv & (owner != 0)                         # planeta nao era nosso
    failed = ~flip                                      # nao virou (surv <= ps)
    shortfall = jnp.where(we_dom & foreign & failed, ps - surv, jnp.inf)  # >=0
    waste0 = jnp.where(we_dom & foreign & failed, surv, 0.0)
    owner = jnp.where(pv, new_owner, owner)
    ps = jnp.where(pv, new_ships, ps)

    # frotas removidas saem do pool
    fa = fa & ~removed

    new = State(p_valid=pv, p_owner=owner, p_x=px, p_y=py, p_r=pr, p_ships=ps,
                p_prod=state.p_prod, p_orbit=porb,
                f_active=fa, f_owner=fo, f_x=fx, f_y=fy, f_ang=fang, f_ships=fsh,
                omega=state.omega, step=state.step + 1)

    # done + reward (placar = naves em planetas + frotas)
    my = jnp.sum(jnp.where(pv & (owner == 0), ps, 0.0)) + jnp.sum(jnp.where(fa & (fo == 0), fsh, 0.0))
    en = jnp.sum(jnp.where(pv & (owner == 1), ps, 0.0)) + jnp.sum(jnp.where(fa & (fo == 1), fsh, 0.0))
    p0_alive = jnp.any(pv & (owner == 0)) | jnp.any(fa & (fo == 0))
    p1_alive = jnp.any(pv & (owner == 1)) | jnp.any(fa & (fo == 1))
    done = (new.step >= EPISODE_STEPS - 1) | (~p0_alive) | (~p1_alive)
    reward = jnp.where(done, jnp.where(my > en, 1.0, jnp.where(en > my, -1.0, 0.0)), 0.0)
    # diag: [P] naves nossas desperdicadas em ataques falhos + residuo do alvo
    waste = jnp.stack([waste0, shortfall], -1)          # [P,2]
    return new, done, reward, waste


step_batched = jax.jit(jax.vmap(step_single))


# ----------------------------------------------------------------------------
# Validacao de paridade + benchmark
# ----------------------------------------------------------------------------
def _ref_env(seed):
    real = os.listdir
    os.listdir = lambda p: (["orbit_wars"] if "orbit_wars" in real(p) else real(p))
    try:
        from kaggle_environments import make
    finally:
        os.listdir = real
    e = make("orbit_wars", configuration={"seed": seed}, debug=False)
    e.reset(num_agents=2)
    return e


def parity(seed=42, steps=45):
    """Compara JAX vs motor real com acoes IDENTICAS (pre-cometa, step<50)."""
    rng = np.random.default_rng(0)
    e = _ref_env(seed)
    st = reset_batch([seed])

    def jax_planets(st):
        owner = np.array(st.p_owner[0]); ships = np.array(st.p_ships[0])
        valid = np.array(st.p_valid[0])
        return owner, ships, valid

    max_ship_diff = 0.0
    owner_mismatch = 0
    for t in range(steps):
        obs = e.state[0].observation
        planets = obs["planets"]
        # acao aleatoria: cada planeta do dono lanca metade das naves p/ um angulo
        a0 = np.zeros(P_MAX); a0s = np.zeros(P_MAX)
        a1 = np.zeros(P_MAX); a1s = np.zeros(P_MAX)
        ref_moves = [[], []]
        for i, pl in enumerate(planets[:P_MAX]):
            _id, own, x, y, r, sh, pr = pl
            if own in (0, 1) and sh >= 4 and rng.random() < 0.5:
                ang = float(rng.uniform(0, 2 * math.pi))
                send = int(sh // 2)
                if own == 0:
                    a0[i] = ang; a0s[i] = send
                else:
                    a1[i] = ang; a1s[i] = send
                ref_moves[own].append([_id, ang, send])
        # passo no motor real
        e.step([ref_moves[0], ref_moves[1]])
        # passo no JAX
        st, _, _, _ = step_batched(st,
            jnp.asarray(a0[None], jnp.float32), jnp.asarray(a0s[None], jnp.float32),
            jnp.asarray(a1[None], jnp.float32), jnp.asarray(a1s[None], jnp.float32))
        # comparar (motor real indexa planetas por id; aqui por linha -- mesma ordem ate cometas)
        ro = e.state[0].observation["planets"]
        jo, jsh, jv = jax_planets(st)
        for i, pl in enumerate(ro[:P_MAX]):
            if i < P_MAX and jv[i]:
                if int(pl[1]) != int(jo[i]):
                    owner_mismatch += 1
                max_ship_diff = max(max_ship_diff, abs(float(pl[5]) - float(jsh[i])))
    print(f"[parity seed={seed}, {steps} steps] max |Δships|={max_ship_diff:.3f}  "
          f"owner_mismatches={owner_mismatch}")
    return max_ship_diff, owner_mismatch


def bench(batch=256, steps=200):
    seeds = list(range(batch))
    st = reset_batch(seeds)
    z = jnp.zeros((batch, P_MAX), jnp.float32)
    # acao aleatoria fixa p/ estressar o step
    key = jax.random.PRNGKey(0)
    ang = jax.random.uniform(key, (batch, P_MAX), maxval=2 * math.pi)
    sh = (jax.random.uniform(key, (batch, P_MAX)) < 0.3).astype(jnp.float32) * 5.0
    # warmup (compila)
    st, _, _, _ = step_batched(st, ang, sh, ang, sh)
    jax.block_until_ready(st.p_ships)
    t0 = time.time()
    for _ in range(steps):
        st, _, _, _ = step_batched(st, ang, sh, ang, sh)
    jax.block_until_ready(st.p_ships)
    dt = time.time() - t0
    sps = batch * steps / dt
    print(f"[bench] batch={batch} steps={steps}  {dt:.2f}s  -> {sps:,.0f} SPS "
          f"(steps/seg agregando o batch)")
    return sps


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    if mode == "parity":
        parity()
    elif mode == "bench":
        bench(batch=int(sys.argv[2]) if len(sys.argv) > 2 else 256)
    else:
        print("uso: python3 jax_env.py [parity|bench]")
