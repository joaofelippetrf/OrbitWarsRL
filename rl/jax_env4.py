"""Motor Orbit Wars em JAX -- 4 JOGADORES (estende jax_env 2p).

Mesma fisica do jax_env (launch -> producao -> rotacao -> movimento com colisao
swept -> combate), mudando so o que difere no 4-player:
  - LAUNCH: 4 acoes (a0..a3), uma por jogador, selecionadas por owner.
  - COMBATE: resolucao de N vias via sort (survivor = top - second, owner =
    argmax). A mesma formula vale p/ 2 e 4 jogadores (motor real: orbit_wars.py
    linha ~650). Empate no topo -> ninguem sobrevive (survivor=-1).
  - REWARD: rank do player 0 (treinado) mapeado p/ [-1,+1]: #1=+1, #4=-1.

reset_*_4p atribui os 4 homes (base+j[1]=j, igual ao motor com num_agents=4).
SEM cometas (como o jax_env). Reusa State/fisica/helpers do jax_env.

CLI:
    python3 jax_env4.py parity   # compara com o motor real 4-player
    python3 jax_env4.py bench
"""
import math
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from jax_env import (State, CENTER, SUN_R, ROT_LIMIT, BOARD, MAX_SPEED, P_MAX,
                     F_MAX, EPISODE_STEPS, _fleet_speed, _seg_dist_to_center,
                     _engine_module)


# ----------------------------------------------------------------------------
# Host reset 4-player (numpy) -- reaproveita o gerador do motor real.
# ----------------------------------------------------------------------------
def reset_single_np_4p(seed):
    import random
    ow = _engine_module()
    rng = random.Random(int(seed))
    omega = rng.uniform(0.025, 0.05)
    planets = ow.generate_planets(rng)
    ngroups = len(planets) // 4
    if ngroups > 0:
        base = rng.randint(0, ngroups - 1) * 4
        for j in range(4):                       # 4 homes (igual ao motor 4p)
            planets[base + j][1] = j
            planets[base + j][5] = 10
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


def reset_batch_4p(seeds):
    singles = [reset_single_np_4p(s) for s in seeds]
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
# Step 4-player (single-game; vmap/jit aplicados depois)
# ----------------------------------------------------------------------------
def step_single_4p(state, a0_ang, a0_sh, a1_ang, a1_sh,
                   a2_ang, a2_sh, a3_ang, a3_sh):
    owner = state.p_owner
    px, py, pr, ps = state.p_x, state.p_y, state.p_r, state.p_ships
    pv, porb = state.p_valid, state.p_orbit

    # --- 1) LAUNCH: 1 por planeta (o dono escolhe entre os 4 jogadores) ---
    req_sh = jnp.where(owner == 0, a0_sh,
             jnp.where(owner == 1, a1_sh,
             jnp.where(owner == 2, a2_sh,
             jnp.where(owner == 3, a3_sh, 0.0))))
    req_ang = jnp.where(owner == 0, a0_ang,
              jnp.where(owner == 1, a1_ang,
              jnp.where(owner == 2, a2_ang, a3_ang)))
    launch_sh = jnp.clip(jnp.floor(req_sh), 0.0, ps)
    launch = pv & (owner >= 0) & (launch_sh >= 1.0)
    launch_sh = jnp.where(launch, launch_sh, 0.0)
    ps = ps - launch_sh
    spawn_x = px + jnp.cos(req_ang) * (pr + 0.1)
    spawn_y = py + jnp.sin(req_ang) * (pr + 0.1)

    fa = state.f_active
    inactive = ~fa
    inactive_rank = jnp.cumsum(inactive.astype(jnp.int32)) - 1
    slot_of_rank = jnp.zeros(F_MAX, jnp.int32).at[
        jnp.where(inactive, inactive_rank, F_MAX - 1)].set(
        jnp.where(inactive, jnp.arange(F_MAX), 0))
    launch_rank = jnp.cumsum(launch.astype(jnp.int32)) - 1
    n_free = jnp.sum(inactive)
    dest = slot_of_rank[jnp.clip(launch_rank, 0, F_MAX - 1)]
    can = launch & (launch_rank < n_free)
    sel = jnp.where(can, dest, F_MAX - 1)
    fa = fa.at[sel].set(jnp.where(can, True, fa[sel]))
    fo = state.f_owner.at[sel].set(jnp.where(can, owner, state.f_owner[sel]))
    fx = state.f_x.at[sel].set(jnp.where(can, spawn_x, state.f_x[sel]))
    fy = state.f_y.at[sel].set(jnp.where(can, spawn_y, state.f_y[sel]))
    fang = state.f_ang.at[sel].set(jnp.where(can, req_ang, state.f_ang[sel]))
    fsh = state.f_ships.at[sel].set(jnp.where(can, launch_sh, state.f_ships[sel]))

    # --- 2) PRODUCAO ---
    ps = ps + jnp.where(pv & (owner >= 0), state.p_prod, 0.0)

    # --- 3) ROTACAO dos planetas ---
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
    A = jnp.stack([f_old_x, f_old_y], -1)[:, None, :]
    Bp = jnp.stack([f_new_x, f_new_y], -1)[:, None, :]
    P0 = jnp.stack([p_old_x, p_old_y], -1)[None, :, :]
    P1 = jnp.stack([p_new_x, p_new_y], -1)[None, :, :]
    d0 = A - P0
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
    hit = hit & pv[None, :] & fa[:, None]
    hit_any = jnp.any(hit, -1)
    hit_idx = jnp.argmax(hit.astype(jnp.int32), -1)

    oob = (f_new_x < 0) | (f_new_x > BOARD) | (f_new_y < 0) | (f_new_y > BOARD)
    sun = _seg_dist_to_center(f_old_x, f_old_y, f_new_x, f_new_y) < SUN_R
    removed = fa & (hit_any | oob | sun)
    survives = fa & ~removed
    fx = jnp.where(survives, f_new_x, fx)
    fy = jnp.where(survives, f_new_y, fy)

    # --- 5) aplica movimento dos planetas ---
    px, py = p_new_x, p_new_y

    # --- 6) COMBATE de N vias (sort: survivor = top - second, owner = argmax) ---
    contrib = fa & hit_any
    def arrivals(pl):
        return jnp.zeros(P_MAX).at[hit_idx].add(jnp.where(contrib & (fo == pl), fsh, 0.0))
    s_all = jnp.stack([arrivals(0), arrivals(1), arrivals(2), arrivals(3)], -1)  # [P,4]
    s_sorted = -jnp.sort(-s_all, -1)                       # desc
    top = s_sorted[:, 0]; second = s_sorted[:, 1]
    surv = top - second                                    # empate no topo -> 0
    surv_owner = jnp.where(surv > 0.0, jnp.argmax(s_all, -1).astype(jnp.int32), -1)
    has = (surv > 0.0) & pv
    same = owner == surv_owner
    diff = ps - surv
    flip = diff < 0.0
    new_owner = jnp.where(has & (~same) & flip, surv_owner, owner)
    new_ships = jnp.where(has,
                          jnp.where(same, ps + surv, jnp.where(flip, -diff, diff)),
                          ps)
    owner = jnp.where(pv, new_owner, owner)
    ps = jnp.where(pv, new_ships, ps)
    fa = fa & ~removed

    new = State(p_valid=pv, p_owner=owner, p_x=px, p_y=py, p_r=pr, p_ships=ps,
                p_prod=state.p_prod, p_orbit=porb,
                f_active=fa, f_owner=fo, f_x=fx, f_y=fy, f_ang=fang, f_ships=fsh,
                omega=state.omega, step=state.step + 1)

    # --- done + reward (rank-based para o player 0) ---
    def total(pl):
        return jnp.sum(jnp.where(pv & (owner == pl), ps, 0.0)) \
             + jnp.sum(jnp.where(fa & (fo == pl), fsh, 0.0))
    t0, t1p, t2p, t3p = total(0), total(1), total(2), total(3)
    rank = ((t1p > t0).astype(jnp.float32) + (t2p > t0).astype(jnp.float32)
            + (t3p > t0).astype(jnp.float32))          # 0 (lider) .. 3 (ultimo)
    alive = lambda pl: jnp.any(pv & (owner == pl)) | jnp.any(fa & (fo == pl))
    a0v, a1v, a2v, a3v = alive(0), alive(1), alive(2), alive(3)
    n_alive = a0v.astype(jnp.int32) + a1v.astype(jnp.int32) \
            + a2v.astype(jnp.int32) + a3v.astype(jnp.int32)
    done = (new.step >= EPISODE_STEPS - 1) | (~a0v) | (n_alive <= 1)
    # reward BINARIO: +1 so se vivo e #1 (lider), senao -1. Sem shaping.
    rew = jnp.where(a0v & (rank == 0.0), 1.0, -1.0)
    reward = jnp.where(done, rew, 0.0)
    return new, done, reward


step_batched_4p = jax.jit(jax.vmap(step_single_4p))


# ----------------------------------------------------------------------------
# Paridade vs motor real (4-player) + bench
# ----------------------------------------------------------------------------
def _ref_env(seed):
    import os
    real = os.listdir
    os.listdir = lambda p: (["orbit_wars"] if "orbit_wars" in real(p) else real(p))
    try:
        from kaggle_environments import make
    finally:
        os.listdir = real
    e = make("orbit_wars", configuration={"seed": seed}, debug=False)
    e.reset(num_agents=4)
    return e


def parity(seed=42, steps=45):
    rng = np.random.default_rng(0)
    e = _ref_env(seed)
    st = reset_batch_4p([seed])
    max_ship_diff = 0.0; owner_mismatch = 0
    for t in range(steps):
        obs = e.state[0].observation
        planets = obs["planets"]
        acts = [np.zeros(P_MAX) for _ in range(4)]      # angulos
        ashp = [np.zeros(P_MAX) for _ in range(4)]      # naves
        ref_moves = [[], [], [], []]
        for i, pl in enumerate(planets[:P_MAX]):
            _id, own, x, y, r, sh, pr = pl
            if own in (0, 1, 2, 3) and sh >= 4 and rng.random() < 0.5:
                ang = float(rng.uniform(0, 2 * math.pi)); send = int(sh // 2)
                acts[own][i] = ang; ashp[own][i] = send
                ref_moves[own].append([_id, ang, send])
        e.step([ref_moves[0], ref_moves[1], ref_moves[2], ref_moves[3]])
        b = lambda a: jnp.asarray(a[None], jnp.float32)
        st, _, _ = step_batched_4p(st, b(acts[0]), b(ashp[0]), b(acts[1]), b(ashp[1]),
                                   b(acts[2]), b(ashp[2]), b(acts[3]), b(ashp[3]))
        ro = e.state[0].observation["planets"]
        jo = np.array(st.p_owner[0]); jsh = np.array(st.p_ships[0]); jv = np.array(st.p_valid[0])
        for i, pl in enumerate(ro[:P_MAX]):
            if i < P_MAX and jv[i]:
                if int(pl[1]) != int(jo[i]):
                    owner_mismatch += 1
                max_ship_diff = max(max_ship_diff, abs(float(pl[5]) - float(jsh[i])))
    print(f"[parity-4p seed={seed}, {steps} steps] max |Δships|={max_ship_diff:.3f} "
          f"owner_mismatches={owner_mismatch}")
    return max_ship_diff, owner_mismatch


def bench(batch=256, steps=200):
    st = reset_batch_4p(list(range(batch)))
    key = jax.random.PRNGKey(0)
    ang = jax.random.uniform(key, (batch, P_MAX), maxval=2 * math.pi)
    sh = (jax.random.uniform(key, (batch, P_MAX)) < 0.3).astype(jnp.float32) * 5.0
    st, _, _ = step_batched_4p(st, ang, sh, ang, sh, ang, sh, ang, sh)
    jax.block_until_ready(st.p_ships)
    t0 = time.time()
    for _ in range(steps):
        st, _, _ = step_batched_4p(st, ang, sh, ang, sh, ang, sh, ang, sh)
    jax.block_until_ready(st.p_ships)
    dt = time.time() - t0
    print(f"[bench-4p] batch={batch} steps={steps} {dt:.2f}s -> {batch*steps/dt:,.0f} SPS")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    if mode == "parity":
        for s in (42, 7, 100):
            parity(s)
    elif mode == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 256)
    else:
        print("uso: python3 jax_env4.py [parity|bench]")
