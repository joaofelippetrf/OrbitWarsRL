"""Codificacao da observacao do Orbit Wars em tensores de tamanho fixo.

Tudo relativo ao jogador `player`. Saidas em numpy (servem ao treino com torch
e a inferencia em numpy no deploy). Planetas e frotas viram matrizes [N_MAX, F]
e [M_MAX, F] com mascaras booleanas; quem nao existe e zero/mascara=False.

Geometria do tabuleiro (espelha o motor): sol em (50,50), raio 10, board 100,
limite de rotacao 50.
"""
import math
import numpy as np

N_MAX = 40            # max de planetas (incl. cometas)
M_MAX = 80            # max de frotas
PF = 12               # features por planeta
FF = 7                # features por frota
GF = 8                # features globais
SUN = 50.0
ROT_LIMIT = 50.0


def _is_rotating(x, y, r):
    return math.hypot(x - SUN, y - SUN) + r < ROT_LIMIT


def encode(obs, player):
    """obs (dict) -> dict de arrays:
      planets[N_MAX,PF], pmask[N_MAX]
      fleets[M_MAX,FF], fmask[M_MAX]
      glob[GF]
      pid[N_MAX] int (id do planeta na linha; -1 se vazio)
      is_mine[N_MAX] bool, p_xy[N_MAX,2] (coords cruas p/ decode de acao)
    """
    planets = obs["planets"]
    fleets = obs["fleets"]
    comet_ids = set(obs.get("comet_planet_ids", []))
    step = obs.get("step", 0)
    omega = obs.get("angular_velocity", 0.0)

    P = np.zeros((N_MAX, PF), np.float32)
    pmask = np.zeros(N_MAX, bool)
    pid = np.full(N_MAX, -1, np.int64)
    is_mine = np.zeros(N_MAX, bool)
    p_xy = np.zeros((N_MAX, 2), np.float32)

    my_ships = en_ships = my_prod = en_prod = 0.0
    my_cnt = en_cnt = 0
    for i, p in enumerate(planets[:N_MAX]):
        _id, owner, x, y, r, ships, prod = p
        mine = owner == player
        enemy = owner != player and owner != -1
        neutral = owner == -1
        rot = _is_rotating(x, y, r)
        P[i] = [1.0 if mine else 0.0,
                1.0 if enemy else 0.0,
                1.0 if neutral else 0.0,
                x / 100.0, y / 100.0,
                r / 3.0,
                math.log1p(max(ships, 0)) / 7.0,
                prod / 5.0,
                1.0 if rot else 0.0,
                math.hypot(x - SUN, y - SUN) / 70.0,
                1.0 if _id in comet_ids else 0.0,
                math.atan2(y - SUN, x - SUN) / math.pi]
        pmask[i] = True
        pid[i] = _id
        is_mine[i] = mine
        p_xy[i] = [x, y]
        if mine:
            my_ships += ships; my_prod += prod; my_cnt += 1
        elif enemy:
            en_ships += ships; en_prod += prod; en_cnt += 1

    F = np.zeros((M_MAX, FF), np.float32)
    fmask = np.zeros(M_MAX, bool)
    for j, f in enumerate(fleets[:M_MAX]):
        _id, owner, x, y, angle, _from, ships = f
        mine = owner == player
        F[j] = [1.0 if mine else 0.0,
                0.0 if mine else 1.0,
                x / 100.0, y / 100.0,
                math.cos(angle), math.sin(angle),
                math.log1p(max(ships, 0)) / 7.0]
        fmask[j] = True
        if mine:
            my_ships += ships
        else:
            en_ships += ships

    glob = np.array([
        step / 500.0,
        omega * 20.0,
        math.log1p(my_ships) / 9.0,
        math.log1p(en_ships) / 9.0,
        my_prod / 30.0,
        en_prod / 30.0,
        my_cnt / 10.0,
        en_cnt / 10.0,
    ], np.float32)

    return {"planets": P, "pmask": pmask, "fleets": F, "fmask": fmask,
            "glob": glob, "pid": pid, "is_mine": is_mine, "p_xy": p_xy}


def totals(obs, player):
    """(naves minhas, naves inimigas) somando planetas + frotas. P/ reward shaping."""
    mine = enemy = 0
    for p in obs["planets"]:
        if p[1] == player:
            mine += p[5]
        elif p[1] != -1:
            enemy += p[5]
    for f in obs["fleets"]:
        if f[1] == player:
            mine += f[6]
        else:
            enemy += f[6]
    return float(mine), float(enemy)
