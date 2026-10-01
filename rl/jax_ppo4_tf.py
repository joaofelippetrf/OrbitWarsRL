"""PPO 4-JOGADORES com a politica TRANSFORMER v3 (feat-inbound, 16 feats).

Diferenca-chave p/ jax_ppo4.py (DeepSets): os 3 assentos de oponente sao sorteados
de forma INDEPENDENTE a cada partida, de um pool de snapshots de self-play + um
oponente FIXO (ex: ckpt_tf3_init.pkl) com probabilidade `--fixed-opp-frac` (10%).
Assim uma partida pode ter qualquer combinacao: 1 v3_init + 2 self-play, 3 self-play,
2 v3_init + 1 self-play, etc. O player 0 e o unico treinado.

Correcoes vs o 4p antigo (necessarias p/ o transformer v3):
  - rot_state_4p agora transforma TAMBEM as frotas (posicao/angulo/owner), senao as
    features de inbound do oponente ficariam erradas (frotas na perspectiva errada).
  - o rollout salva o `inbound` do estado REAL no traj; o update reconstroi o estado
    SEM frotas, entao sem isso as 2 colunas de inbound teriam gradiente 0 (mesmo
    bug ja corrigido no 2p).

Uso:
    python3 jax_ppo4_tf.py --iters 2000 --games 1024 --horizon 128 \
        --arch transformer --width 112 --tf-layers 2 --heads 4 \
        --load ckpt_tf3.pkl --fixed-opp ckpt_tf3_init.pkl --fixed-opp-frac 0.10 \
        --save ckpt_tf3_4p.pkl
"""
import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jax_ppo as JP
import jax_env4 as J4
from jax_env4 import (reset_batch_4p, step_batched_4p, P_MAX, F_MAX,
                      EPISODE_STEPS, CENTER)

# este trainer e exclusivo da politica v3 -> features com inbound (16 colunas).
JP._FEAT_INBOUND = True

# rotacao do home de cada slot rel. ao player 0 (medido: P1=+90, P2=+270, P3=+180)
_THETA4 = [0.0, math.pi / 2, 3 * math.pi / 2, math.pi]


def rot_state_4p(st, k):
    """Estado visto pelo jogador k na perspectiva canonica (como player 0).
    Rotaciona planetas E FROTAS por -theta_k e remapeia owner (k->0, demais
    nao-neutros->1). Transformar as frotas e essencial p/ as features de inbound."""
    theta = _THETA4[k]
    ca, sa = math.cos(-theta), math.sin(-theta)

    def _rot(x, y):
        dx = x - CENTER
        dy = y - CENTER
        return CENTER + dx * ca - dy * sa, CENTER + dx * sa + dy * ca

    nx, ny = _rot(st.p_x, st.p_y)
    fx, fy = _rot(st.f_x, st.f_y)
    no = jnp.where(st.p_owner == k, 0, jnp.where(st.p_owner == -1, -1, 1))
    fo = jnp.where(st.f_owner == k, 0, jnp.where(st.f_owner == -1, -1, 1))
    return st._replace(p_x=nx, p_y=ny, p_owner=no,
                       f_x=fx, f_y=fy, f_ang=st.f_ang - theta, f_owner=fo)


def _opp_move(opp_params, model, st, k, key):
    """Jogada do oponente (snapshot/fixo) no assento k, vendo a perspectiva
    canonica; o angulo de saida e des-rotacionado por +theta_k."""
    fst = rot_state_4p(st, k)
    _, _, _, _, ang, sh, _, _ = JP.policy_act(opp_params, model, fst, 0, key, sample=True)
    return ang + _THETA4[k], sh


def make_rollout_4p(model, horizon):
    def rollout(params, opp1, opp2, opp3, st, key):
        def stepf(carry, _):
            st, key, done_acc = carry
            key, k0, k1, k2, k3 = jax.random.split(key, 5)
            a0, fa0, logp0, val0, ang0, sh0, _, _ = JP.policy_act(params, model, st, 0, k0)
            ang1, sh1 = _opp_move(opp1, model, st, 1, k1)
            ang2, sh2 = _opp_move(opp2, model, st, 2, k2)
            ang3, sh3 = _opp_move(opp3, model, st, 3, k3)
            nst, done, env_r = step_batched_4p(st, ang0, sh0, ang1, sh1,
                                               ang2, sh2, ang3, sh3)
            live = (~done_acc).astype(jnp.float32)
            first_done = done & (~done_acc)
            r = jnp.where(first_done, env_r, 0.0) * live      # so terminal (#1=+1)
            done_acc = done_acc | done
            out = dict(action=a0, frac_action=fa0, logp=logp0, value=val0, reward=r, live=live,
                       p_owner=st.p_owner, p_x=st.p_x, p_y=st.p_y, p_r=st.p_r,
                       p_ships=st.p_ships, p_prod=st.p_prod, p_valid=st.p_valid,
                       p_orbit=st.p_orbit, omega=st.omega, step=st.step,
                       # inbound do estado REAL (com frotas) p/ o player 0: salvo aqui
                       # pois no update o estado e reconstruido sem frotas.
                       inbound=jnp.stack(JP.inbound_ships(st, 0), -1))   # [B,P,2]
            return (nst, key, done_acc), out
        B = st.p_owner.shape[0]
        init = (st, key, jnp.zeros(B, bool))
        (last_st, _, _), traj = jax.lax.scan(stepf, init, None, length=horizon)
        _, _, _, last_val, _, _, _, _ = JP.policy_act(params, model, last_st, 0,
                                                      jax.random.PRNGKey(0), sample=False)
        return traj, last_val
    return jax.jit(rollout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", action="store_true",
                    help="warmup+cosine (playbook transformer-RL)")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--arch", choices=["transformer", "deepsets"], default="transformer")
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--tf-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=512)
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--pool-every", type=int, default=10)
    ap.add_argument("--fixed-opp", type=str, default="",
                    help="checkpoint usado como oponente FIXO por assento (ex: ckpt_tf3_init.pkl)")
    ap.add_argument("--fixed-opp-frac", type=float, default=0.10,
                    help="prob. de CADA assento de oponente ser o --fixed-opp (resto = self-play pool)")
    ap.add_argument("--load", default="")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    if args.arch == "transformer":
        model = JP.PolicyTF(d=args.width, heads=args.heads, layers=args.tf_layers)
    else:
        model = JP.Policy(d=args.width)
    key = jax.random.PRNGKey(0)
    st0 = reset_batch_4p(list(range(args.games)))
    feat, pm, glob = JP.features(st0, 0)
    params = model.init(key, feat, pm, glob, JP.pair_features(st0, 0))

    if args.lr_schedule:
        _nmb = max(1, (args.games * args.horizon) // args.minibatch) * args.epochs
        _total = max(1, args.iters * _nmb)
        _warm = args.warmup_steps if args.warmup_steps > 0 else max(1, int(0.03 * _total))
        lr_for_opt = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr, warmup_steps=_warm,
            decay_steps=_total, end_value=args.lr * 0.1)
        print(f"lr schedule: warmup {_warm} -> peak {args.lr:.1e} -> cosine ate "
              f"{args.lr*0.1:.1e} ({_total} steps)", flush=True)
    else:
        lr_for_opt = args.lr
    opt = optax.chain(optax.clip_by_global_norm(0.5),
                      optax.adamw(lr_for_opt, weight_decay=args.weight_decay))
    opt_state = opt.init(params)

    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}", flush=True)
    if args.load:
        lp, lo = JP.load_checkpoint(args.load)
        params, n_ok, n_tot = JP.merge_params(params, lp)
        try:
            opt_state = lo if (n_ok == n_tot and lo is not None) else opt.init(params)
        except Exception:
            opt_state = opt.init(params)
        print(f"warm-start de {args.load}: {n_ok}/{n_tot} tensores", flush=True)

    # oponente fixo (ex: v3_init): carregado e alinhado a arch atual via merge_params
    fixed_opp = None
    if args.fixed_opp:
        fp, _ = JP.load_checkpoint(args.fixed_opp)
        fixed_opp, n_ok, n_tot = JP.merge_params(params, fp)
        print(f"oponente FIXO {args.fixed_opp}: {n_ok}/{n_tot} tensores, "
              f"{args.fixed_opp_frac:.0%}/assento", flush=True)

    rollout = make_rollout_4p(model, args.horizon)

    @jax.jit
    def mb_step(params, opt_state, stf, inb, act, fa, oldlp, A, R, LV):
        def loss_fn(p):
            lp, ent_t, ent_f, val = JP.eval_logp(p, model, stf, 0, act, fa, inbound=inb)
            ent = ent_t + ent_f
            ratio = jnp.exp(lp - oldlp)
            s1 = ratio * A
            s2 = jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * A
            pol = -jnp.sum(jnp.minimum(s1, s2) * LV) / jnp.clip(LV.sum(), 1.0)
            vloss = jnp.sum(((val - R) ** 2) * LV) / jnp.clip(LV.sum(), 1.0)
            entl = -jnp.sum(ent * LV) / jnp.clip(LV.sum(), 1.0)
            clipf = jnp.sum((jnp.abs(ratio - 1.0) > args.clip).astype(jnp.float32) * LV) \
                    / jnp.clip(LV.sum(), 1.0)
            return pol + args.vf * vloss + args.ent * entl, (pol, vloss, -entl, clipf)
        (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state2 = opt.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state2, aux

    def update(params, opt_state, traj, last_val, key):
        adv, ret = JP.gae(traj["reward"], traj["value"], traj["live"], last_val,
                          args.gamma, args.lam)
        live = traj["live"]
        n = jnp.clip(live.sum(), 1.0)
        rmean = (ret * live).sum() / n
        resid = ret - traj["value"]
        ev = 1.0 - ((resid - (resid * live).sum() / n) ** 2 * live).sum() \
                 / jnp.clip(((ret - rmean) ** 2 * live).sum(), 1e-8)
        T, B = traj["reward"].shape
        N = T * B
        flat = lambda a: a.reshape((N,) + a.shape[2:])
        pv_f=flat(traj["p_valid"]); po_f=flat(traj["p_owner"])
        px_f=flat(traj["p_x"]); py_f=flat(traj["p_y"]); pr_f=flat(traj["p_r"])
        psh_f=flat(traj["p_ships"]); ppr_f=flat(traj["p_prod"]); por_f=flat(traj["p_orbit"])
        om_f=flat(traj["omega"]); stp_f=flat(traj["step"]); inb_f=flat(traj["inbound"])
        act_f=flat(traj["action"]); fa_f=flat(traj["frac_action"])
        oldlp_f=flat(traj["logp"]); A_f=flat(adv); R_f=flat(ret); LV_f=flat(traj["live"])
        MB = args.minibatch
        zb=jnp.zeros((MB,F_MAX),bool); zi=jnp.zeros((MB,F_MAX),jnp.int32); zf=jnp.zeros((MB,F_MAX),jnp.float32)
        stats=(0.,0.,0.,0.)
        for _ in range(args.epochs):
            key, ke = jax.random.split(key)
            perm = jax.random.permutation(ke, N)
            for s in range(0, N - MB + 1, MB):
                idx = perm[s:s+MB]
                mb = J4.State(p_valid=pv_f[idx], p_owner=po_f[idx], p_x=px_f[idx], p_y=py_f[idx],
                              p_r=pr_f[idx], p_ships=psh_f[idx], p_prod=ppr_f[idx], p_orbit=por_f[idx],
                              f_active=zb, f_owner=zi, f_x=zf, f_y=zf, f_ang=zf, f_ships=zf,
                              omega=om_f[idx], step=stp_f[idx])
                params, opt_state, stats = mb_step(params, opt_state, mb, inb_f[idx],
                    act_f[idx], fa_f[idx], oldlp_f[idx], A_f[idx], R_f[idx], LV_f[idx])
        return params, opt_state, (stats[0], stats[1], stats[2], stats[3], ev), float(rmean)

    # ---- evals: #1-rate vs 3x heuristica e vs 3x oponente-fixo --------------
    @jax.jit
    def eval_vs_heur(params, st):
        def stepf(carry, _):
            st, done = carry
            _, _, _, _, a0, s0, _, _ = JP.policy_act(params, model, st, 0,
                                                     jax.random.PRNGKey(0), sample=False)
            a1, s1 = JP.smart_aggressive(st, 1)
            a2, s2 = JP.smart_aggressive(st, 2)
            a3, s3 = JP.smart_aggressive(st, 3)
            nst, d, _ = step_batched_4p(st, a0, s0, a1, s1, a2, s2, a3, s3)
            return (nst, done | d), None
        (last, _), _ = jax.lax.scan(stepf, (st, jnp.zeros(st.p_owner.shape[0], bool)),
                                    None, length=EPISODE_STEPS)
        return _first_rate(last)

    @jax.jit
    def eval_vs_opp(params, opp, st):
        def stepf(carry, _):
            st, done = carry
            _, _, _, _, a0, s0, _, _ = JP.policy_act(params, model, st, 0,
                                                     jax.random.PRNGKey(0), sample=False)
            a1, s1 = _opp_move(opp, model, st, 1, jax.random.PRNGKey(1))
            a2, s2 = _opp_move(opp, model, st, 2, jax.random.PRNGKey(2))
            a3, s3 = _opp_move(opp, model, st, 3, jax.random.PRNGKey(3))
            nst, d, _ = step_batched_4p(st, a0, s0, a1, s1, a2, s2, a3, s3)
            return (nst, done | d), None
        (last, _), _ = jax.lax.scan(stepf, (st, jnp.zeros(st.p_owner.shape[0], bool)),
                                    None, length=EPISODE_STEPS)
        return _first_rate(last)

    def _first_rate(last):
        def total(pl):
            return jnp.sum(jnp.where(last.p_valid & (last.p_owner == pl), last.p_ships, 0.0), -1) \
                 + jnp.sum(jnp.where(last.f_active & (last.f_owner == pl), last.f_ships, 0.0), -1)
        t0 = total(0)
        rank = (total(1) > t0).astype(jnp.float32) + (total(2) > t0).astype(jnp.float32) \
             + (total(3) > t0).astype(jnp.float32)
        return jnp.mean((rank == 0).astype(jnp.float32))

    pool_n = max(256, args.games * 4)
    pool = reset_batch_4p(list(range(20000, 20000 + pool_n)))
    ev = reset_batch_4p(list(range(5000, 5000 + args.eval_games)))
    rng = np.random.default_rng(0)

    def sample_states(k):
        idx = jax.random.randint(k, (args.games,), 0, pool_n)
        return jax.tree_util.tree_map(lambda a: a[idx], pool)

    def sample_opp():
        """1 oponente p/ um assento: fixo (v3_init) c/ prob fixed_opp_frac, senao
        um snapshot aleatorio do pool de self-play."""
        if fixed_opp is not None and rng.random() < args.fixed_opp_frac:
            return fixed_opp
        return snap_pool[rng.integers(len(snap_pool))]

    snap_pool = [params]
    print(f"PPO-4P-TF | games={args.games} horizon={args.horizon} iters={args.iters} "
          f"pool[{args.pool_size}] fixed={args.fixed_opp_frac:.0%} mb={args.minibatch}", flush=True)
    t0 = None; done_steps = 0
    for it in range(1, args.iters + 1):
        key, kr, ks, ku = jax.random.split(key, 4)
        st = sample_states(ks)
        # cada assento sorteado INDEPENDENTE -> qualquer combinacao por partida
        opp1, opp2, opp3 = sample_opp(), sample_opp(), sample_opp()
        traj, last_val = rollout(params, opp1, opp2, opp3, st, kr)
        params, opt_state, stats, rmean = update(params, opt_state, traj, last_val, ku)
        if it % args.pool_every == 0:
            snap_pool.append(params)
            if len(snap_pool) > args.pool_size:
                snap_pool.pop(0)
        if it == 1:
            jax.block_until_ready(params); t0 = time.time(); done_steps = 0
        else:
            done_steps += args.games * args.horizon
        pol, vloss, ent, clipf, ev_v = [float(x) for x in stats]
        msg = (f"it {it:3d} | r/step={rmean:+.4f} pol={pol:+.3f} v={vloss:.3f} "
               f"ev={ev_v:+.3f} ent={ent:.3f} clipf={clipf:.3f}")
        if it % args.eval_every == 0:
            fr_h = float(eval_vs_heur(params, ev))
            sps = done_steps / (time.time() - t0) if t0 and done_steps else 0.0
            msg += f" | #1 vs 3xheur={fr_h:.3f}"
            if fixed_opp is not None:
                fr_i = float(eval_vs_opp(params, fixed_opp, ev))
                msg += f" vs 3xinit={fr_i:.3f}"
            msg += f"  ({sps:,.0f} SPS)"
            if args.save:
                JP.save_checkpoint(args.save, params, opt_state)
        print(msg, flush=True)
    if args.save:
        JP.save_checkpoint(args.save, params, opt_state)
        print(f"checkpoint final salvo em {args.save}", flush=True)
    print("fim.")


if __name__ == "__main__":
    main()
