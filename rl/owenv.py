"""Wrapper do motor Orbit Wars para RL (controle turno-a-turno) + decode da
acao pointer em jogadas do motor, reaproveitando a geometria do submission
(_intercept_n, _ships_needed, _seg_hits_sun).

Acao da politica: por linha de planeta i, indice em [0..N]: 0 = no-op, j>=1 =
mandar naves do planeta i para o planeta da linha j-1 (atacar se inimigo/neutro,
reforcar se for meu outro planeta). Quantidade e angulo saem da heuristica.
"""
import math
import os
import sys
import importlib.util
import contextlib
import numpy as np

import features

MAX_DIST = 60.0
MARGIN = 1.1
GARRISON = 1


@contextlib.contextmanager
def _suppress():
    sys.stdout.flush(); sys.stderr.flush()
    dn = os.open(os.devnull, os.O_WRONLY)
    so, se = os.dup(1), os.dup(2)
    os.dup2(dn, 1); os.dup2(dn, 2)
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(so, 1); os.dup2(se, 2)
        os.close(so); os.close(se); os.close(dn)


# Import rapido do motor + submission (para os helpers de geometria).
_real_listdir = os.listdir
def _only_ow(p):
    items = _real_listdir(p)
    return ["orbit_wars"] if "orbit_wars" in items else items

with _suppress():
    os.listdir = _only_ow
    try:
        from kaggle_environments import make as _make
    finally:
        os.listdir = _real_listdir
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _spec = importlib.util.spec_from_file_location("submission_rl",
                                                   os.path.join(_ROOT, "submission.py"))
    SUB = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(SUB)

Planet = SUB.Planet


def _planets(obs):
    return [Planet(*p) for p in obs["planets"]]


def build_masks(obs, player, enc):
    """src_mask[N] (planetas meus com naves livres), tgt_mask[N,N] (alvo j
    valido p/ fonte i: dist<=MAX_DIST, rota nao cruza o sol, j!=i)."""
    N = features.N_MAX
    pl = _planets(obs)
    n = len(pl)
    src = np.zeros(N, bool)
    tgt = np.zeros((N, N), bool)
    free = np.zeros(N, np.float32)
    for i in range(min(n, N)):
        if pl[i].owner == player and (pl[i].ships - GARRISON) >= 1:
            src[i] = True
            free[i] = pl[i].ships - GARRISON
    for i in range(min(n, N)):
        if not src[i]:
            continue
        si = pl[i]
        for j in range(min(n, N)):
            if j == i:
                continue
            tj = pl[j]
            if math.hypot(si.x - tj.x, si.y - tj.y) > MAX_DIST:
                continue
            if SUB._seg_hits_sun(si, (tj.x, tj.y)):
                continue
            tgt[i, j] = True
        if not tgt[i].any():
            src[i] = False                  # sem alvo valido -> fonte inativa
    return src, tgt, free


def decode_action(obs, player, enc, actions, free):
    """actions[N] (0=no-op, j>=1 -> alvo linha j-1) -> jogadas do motor."""
    omega = obs.get("angular_velocity", 0.0)
    pl = _planets(obs)
    n = len(pl)
    moves = []
    for i in range(min(n, features.N_MAX)):
        a = int(actions[i])
        if a == 0 or free[i] < 1:
            continue
        j = a - 1
        if j >= n or j == i:
            continue
        src, tgt = pl[i], pl[j]
        if tgt.owner == player:
            ships = int(free[i])                         # reforco: manda tudo
        else:
            need = SUB._ships_needed(tgt, math.hypot(src.x - tgt.x, src.y - tgt.y), MARGIN)
            ships = int(min(free[i], max(need, 1)))
        if ships < 1:
            continue
        ax, ay, d, sp = SUB._intercept_n(src, tgt, omega, ships)
        moves.append([src.id, math.atan2(ay - src.y, ax - src.x), ships])
        free[i] -= ships
    return moves


class OWGame:
    """Uma partida controlavel turno-a-turno. Treinamos como player 0; o
    oponente (player 1) e fornecido pelo trainer a cada step."""

    def __init__(self, seed=0):
        self.seed = seed
        self.env = None
        self.done = True

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        with _suppress():
            self.env = _make("orbit_wars", configuration={"seed": self.seed}, debug=False)
            self.state = self.env.reset(num_agents=2)
        self.done = False
        self._prev = features.totals(self.obs(0), 0)
        return self.obs(0)

    def obs(self, player):
        return self.state[player].observation

    def step(self, moves0, moves1, shape_w=0.01, term_w=1.0):
        with _suppress():
            self.state = self.env.step([moves0, moves1])
        o0 = self.obs(0)
        statuses = [s.status for s in self.state]
        step_n = o0.get("step", 0)
        self.done = (statuses[0] != "ACTIVE") or (step_n >= 499)
        my, en = features.totals(o0, 0)
        dmy, den = my - self._prev[0], en - self._prev[1]
        self._prev = (my, en)
        r = shape_w * (dmy - den) / 10.0
        if self.done:
            r += term_w * (1.0 if my > en else (-1.0 if en > my else 0.0))
        return o0, r, self.done
