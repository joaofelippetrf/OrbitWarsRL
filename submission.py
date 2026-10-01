# submission.py -- agent6 (orbit_wars) autocontido
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet


# ============================================================
#  Helpers compartilhados (agent3 e agent4)
# ============================================================
SUN_CENTER = (50.0, 50.0)        # centro do tabuleiro / sol
SUN_RADIUS = 10.0                # frota destruida se a rota passar dentro disso
SUN_BUFFER = 2.0                 # margem extra de seguranca
MAX_SPEED = 6.0                  # shipSpeed padrao
ROTATION_RADIUS_LIMIT = 50.0     # planeta gira se raio_orbital + raio < isso


def _obs_get(obs, key, default):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


def _point_seg_dist(p, a, b):
    """Distancia minima do ponto p ao segmento a-b."""
    l2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    if l2 == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * (b[0] - a[0]) +
                           (p[1] - a[1]) * (b[1] - a[1])) / l2))
    px, py = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
    return math.hypot(p[0] - px, p[1] - py)


def _fleet_speed(ships):
    """Velocidade da frota: escala com o tamanho ate MAX_SPEED."""
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)


def _orbit_radius(p):
    return math.hypot(p.x - SUN_CENTER[0], p.y - SUN_CENTER[1])


def _is_rotating(p):
    return _orbit_radius(p) + p.radius < ROTATION_RADIUS_LIMIT


def _predict(target, eta, omega):
    """Onde o planeta estara daqui a 'eta' ticks (orbita em torno do sol)."""
    if omega == 0.0 or not _is_rotating(target):
        return (target.x, target.y)
    r = _orbit_radius(target)
    ang = math.atan2(target.y - SUN_CENTER[1], target.x - SUN_CENTER[0]) + omega * eta
    return (SUN_CENTER[0] + r * math.cos(ang), SUN_CENTER[1] + r * math.sin(ang))


def _intercept(mine, target, omega):
    """Mira preditiva: resolve o ponto onde lancar a frota pra interceptar
    o planeta em movimento. Retorna (aim_x, aim_y, dist_ate_o_aim)."""
    ax, ay = target.x, target.y
    speed = _fleet_speed(target.ships + 1)        # tamanho aproximado da frota
    for _ in range(5):                            # converge o ponto de encontro
        eta = math.hypot(ax - mine.x, ay - mine.y) / speed
        ax, ay = _predict(target, eta, omega)
    return ax, ay, math.hypot(ax - mine.x, ay - mine.y)


def _seg_hits_sun(mine, point):
    """True se a reta do planeta ate 'point' cruza (perto demais) o sol."""
    return _point_seg_dist(SUN_CENTER, (mine.x, mine.y), point) < SUN_RADIUS + SUN_BUFFER


def _ships_needed(target, dist, margin):
    """Naves do alvo na CHEGADA (planetas com dono produzem durante a viagem)
    + folga 'margin'."""
    produces = target.owner != -1
    defenders = target.ships
    guess = target.ships + 1
    for _ in range(2):
        eta = dist / _fleet_speed(guess)
        defenders = target.ships + (target.production * eta if produces else 0.0)
        guess = defenders + 1
    return math.ceil((defenders + 1) * margin)


# ============================================================
#  agent3  -- heuristica focada, agora com mira preditiva
# ============================================================
A3_MAX_DIST = 45.0       # so ataca alvos "perto"
A3_SUPERIORITY = 1.4     # precisa de bastante superioridade
A3_MARGIN = 1.2          # folga sobre o necessario
A3_GARRISON = 15         # nunca esvazia o planeta de origem


def agent3(obs):
    """Ataca quando: (1) confianca de capturar (conta producao na viagem),
    (2) alvo perto, (3) bastante superioridade, (4) rota livre do sol.
    Usa mira preditiva pra interceptar planetas em orbita. Prioriza inimigos."""
    moves = []
    player = _obs_get(obs, "player", 0)
    omega = _obs_get(obs, "angular_velocity", 0.0)
    planets = [Planet(*p) for p in _obs_get(obs, "planets", [])]

    my_planets = [p for p in planets if p.owner == player]
    enemies = [p for p in planets if p.owner != player and p.owner != -1]
    neutrals = [p for p in planets if p.owner == -1]

    def best_move(mine, candidates):
        ranked = sorted(candidates, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        for target in ranked:
            ax, ay, d = _intercept(mine, target, omega)   # mira preditiva
            if d > A3_MAX_DIST:                            # perto?
                continue
            if _seg_hits_sun(mine, (ax, ay)):              # rota livre do sol?
                continue
            needed = _ships_needed(target, d, A3_MARGIN)
            if mine.ships - A3_GARRISON >= needed and mine.ships >= needed * A3_SUPERIORITY:
                angle = math.atan2(ay - mine.y, ax - mine.x)
                return [mine.id, angle, needed]
        return None

    for mine in my_planets:
        move = best_move(mine, enemies) or best_move(mine, neutrals)
        if move is not None:
            moves.append(move)

    return moves



# ============================================================
#  agent4  -- heuristica "completa" (usa os helpers do agent3)
#
#  Estrategia, em ordem de importancia:
#   1. DEFESA  : detecta frotas inimigas vindo pros meus planetas e
#                segura naves (garrison dinamico) pra nao perder a base.
#   2. ECONOMIA: escolhe alvos por SCORE = valor / custo, onde valor
#                cresce com a PRODUCAO do planeta (planetas que fabricam
#                naves valem mais) e tem bonus se for inimigo (negar
#                producao ao oponente vale dobrado).
#   3. PRECISAO: mira preditiva (intercepta orbita) + custo ajustado pela
#                producao do alvo durante a viagem.
#   4. SEGURANCA: descarta rotas que passam pelo sol.
#   5. NAO DESPERDICAR: cada planeta-alvo so e atacado uma vez por turno
#                (uma frota que ja captura); cada frota leva, sozinha, o
#                suficiente pra vencer (frotas de bases diferentes chegam
#                em ticks diferentes e NAO somam no combate).
# ============================================================

A4_MAX_DIST = 60.0        # alcance de ataque (mais agressivo que o agent3)
A4_SUPERIORITY = 1.25     # superioridade minima exigida
A4_MARGIN = 1.2           # folga de naves sobre o necessario
A4_GARRISON = 3           # reserva minima de paz
A4_THREAT_HORIZON = 30    # ticks: so conta ameacas que chegam ate aqui
A4_PROD_WEIGHT = 6.0      # peso da producao no valor do alvo
A4_ENEMY_BONUS = 2.0      # negar producao ao inimigo vale o dobro


def _incoming_threat(myplanet, enemy_fleets, omega):
    """Soma das naves inimigas cuja trajetoria reta passa pelo meu planeta
    e que chegam dentro do horizonte de ameaca."""
    threat = 0
    for f in enemy_fleets:
        far = (f.x + math.cos(f.angle) * 200.0, f.y + math.sin(f.angle) * 200.0)
        if _point_seg_dist((myplanet.x, myplanet.y), (f.x, f.y), far) < myplanet.radius + 3.0:
            d = math.hypot(myplanet.x - f.x, myplanet.y - f.y)
            if d / _fleet_speed(f.ships) <= A4_THREAT_HORIZON:
                threat += f.ships
    return threat


def agent4(obs):
    moves = []
    player = _obs_get(obs, "player", 0)
    omega = _obs_get(obs, "angular_velocity", 0.0)
    planets = [Planet(*p) for p in _obs_get(obs, "planets", [])]
    fleets = [Fleet(*f) for f in _obs_get(obs, "fleets", [])]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]      # inimigos + neutros
    enemy_fleets = [f for f in fleets if f.owner != player]

    committed = {}   # target.id -> True quando ja tem frota suficiente indo

    # Planetas com mais naves decidem primeiro (tem mais opcoes de alvo).
    for mine in sorted(my_planets, key=lambda p: -p.ships):
        # --- 1) defesa: segura naves se houver ameaca chegando ---
        threat = _incoming_threat(mine, enemy_fleets, omega)
        garrison = max(A4_GARRISON, threat + 1)
        available = mine.ships - garrison
        if available <= 0:
            continue

        # --- 2) escolhe o alvo de melhor score viavel ---
        best = None
        best_score = -1.0
        best_needed = 0
        best_aim = None
        for t in targets:
            if committed.get(t.id):                      # alguem ja captura
                continue
            ax, ay, d = _intercept(mine, t, omega)       # mira preditiva
            if d > A4_MAX_DIST:
                continue
            if _seg_hits_sun(mine, (ax, ay)):            # rota livre do sol
                continue
            needed = _ships_needed(t, d, A4_MARGIN)
            if available < needed or mine.ships < needed * A4_SUPERIORITY:
                continue
            value = (t.production * A4_PROD_WEIGHT + 1.0)
            if t.owner != -1:
                value *= A4_ENEMY_BONUS                   # negar inimigo vale +
            score = value / (needed * (1.0 + d / 50.0))   # por nave, penaliza dist
            if score > best_score:
                best_score, best, best_needed, best_aim = score, t, needed, (ax, ay)

        if best is not None:
            ax, ay = best_aim
            angle = math.atan2(ay - mine.y, ax - mine.x)
            moves.append([mine.id, angle, best_needed])
            committed[best.id] = True

    return moves



# ============================================================
#  agent5  -- custo combinado + coalizao de varios planetas
#
#  Reaproveita os helpers do agent3 (_intercept, _ships_needed,
#  _seg_hits_sun, _fleet_speed) e _incoming_threat do agent4.
#
#  IDEIA CENTRAL -- funcao de CUSTO/VALOR global por alvo, combinando:
#    (a) distancia aos MEUS planetas  -> mais perto = mais barato de tomar
#    (b) distancia ao INIMIGO         -> perto do inimigo = mais arriscado
#                                        de manter (ele recaptura) => custa +
#    (c) neutro vs inimigo            -> inimigo regenera e fica contestado,
#                                        mas capturar nega producao a ele
#
#  VISAO COMBINADA -- em vez de 1 planeta por alvo, monta uma COALIZAO:
#    soma a forca dos meus planetas ao alcance (do mais proximo ao mais
#    distante) ate cobrir o necessario na CHEGADA. Ondas escalonadas
#    "lascam" o alvo (confirmado no motor); para neutros acumulam direto,
#    para inimigos o custo ja inclui a producao gerada no intervalo.
#    CAVEAT: as ondas chegam em ticks proximos, nao identicos -- modelamos
#    o pior caso (producao ate a ultima onda), entao tende a ser seguro.
# ============================================================
A5_MAX_DIST = 60.0        # alcance maximo de uma frota
A5_MARGIN = 1.05           # folga de naves sobre o necessario
A5_GARRISON = 3           # reserva minima
A5_PROD_WEIGHT = 6.0      # peso da producao no valor do alvo
A5_ENEMY_BONUS = 1.8      # negar producao ao inimigo
A5_W_MINE = 1.0           # peso da distancia aos meus planetas (custo)
A5_W_ENEMY_RISK = 1     # peso do risco de estar perto do inimigo (custo)
A5_MAX_COALITION = 4      # no maximo N planetas concentram no mesmo alvo


def _nearest_dist(p, others):
    """Menor distancia de p a um conjunto de planetas (inf se vazio)."""
    return min((math.hypot(p.x - o.x, p.y - o.y) for o in others), default=float("inf"))


def _target_score(t, my_planets, enemy_planets):
    """Valor / custo do alvo t, combinando (a) dist. aos meus, (b) dist. ao
    inimigo e (c) neutro vs inimigo. Maior score = melhor alvo."""
    d_mine = _nearest_dist(t, my_planets)            # (a) perto de mim = barato
    d_enemy = _nearest_dist(t, enemy_planets)        # (b) perto do inimigo = risco
    is_enemy = t.owner != -1                         # (c)

    value = (t.production * A5_PROD_WEIGHT + 1.0)
    if is_enemy:
        value *= A5_ENEMY_BONUS                      # negar producao vale mais

    # custo sobe com a distancia ate mim e com a PROXIMIDADE ao inimigo
    cost_mine = 1.0 + (d_mine / 50.0) * A5_W_MINE
    cost_risk = 1.0 + (1.0 / (1.0 + d_enemy)) * A5_W_ENEMY_RISK * 50.0
    return value / (cost_mine * cost_risk)


def agent5(obs):
    moves = []
    player = _obs_get(obs, "player", 0)
    omega = _obs_get(obs, "angular_velocity", 0.0)
    planets = [Planet(*p) for p in _obs_get(obs, "planets", [])]
    fleets = [Fleet(*f) for f in _obs_get(obs, "fleets", [])]

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    targets = [p for p in planets if p.owner != player]
    enemy_fleets = [f for f in fleets if f.owner != player]

    # quanto cada planeta pode ceder (mantendo defesa contra ameacas)
    free = {}
    for mine in my_planets:
        threat = _incoming_threat(mine, enemy_fleets, omega)
        free[mine.id] = max(0, mine.ships - max(A5_GARRISON, threat + 1))

    # alvos do melhor para o pior segundo o custo combinado
    ranked_targets = sorted(
        targets, key=lambda t: _target_score(t, my_planets, enemy_planets), reverse=True
    )

    for t in ranked_targets:
        # candidatos: meus planetas que alcancam t (rota livre do sol)
        candidates = []
        for mine in my_planets:
            if free[mine.id] < 1:
                continue
            ax, ay, d = _intercept(mine, t, omega)
            if d > A5_MAX_DIST or _seg_hits_sun(mine, (ax, ay)):
                continue
            candidates.append((mine, ax, ay, d))
        if not candidates:
            continue

        # coalizao: do mais proximo ao mais distante ate cobrir o necessario
        candidates.sort(key=lambda c: c[3])
        candidates = candidates[:A5_MAX_COALITION]

        coalition = []
        total = 0
        needed = None
        for (mine, ax, ay, d) in candidates:
            coalition.append((mine, ax, ay, d))
            total += free[mine.id]
            # custo avaliado na CHEGADA da onda mais distante (pior caso)
            needed = _ships_needed(t, d, A5_MARGIN)
            if total >= needed:
                break

        if needed is None or total < needed:
            continue   # nem a coalizao inteira captura -> ignora o alvo

        # distribui o 'needed' entre a coalizao, dos mais proximos primeiro
        remaining = needed
        for (mine, ax, ay, d) in coalition:
            if remaining <= 0:
                break
            send = min(free[mine.id], remaining)
            if send < 1:
                continue
            angle = math.atan2(ay - mine.y, ax - mine.x)
            moves.append([mine.id, angle, int(send)])
            free[mine.id] -= send
            remaining -= send

    return moves



# ============================================================
#  agent6  -- evolucao do agent5 (params afinados por sweep)
#  Params de distancia vencedores do sweep: W_MINE=1.5 (custo QUADRATICO),
#  CLOSE_BONUS=0.0 (o bonus de proximidade extra atrapalhava).
#
#  NOVO: A6_ROTATING_BONUS -> prioriza planetas em ROTACAO e COMETAS sobre
#  os estaticos. A6_TARGET_COMETS liga/desliga capturar cometas (mira pela
#  trajetoria conhecida do cometa, nao pela orbita).
# ============================================================
COMET_RADIUS = 1.0        # raio do cometa (igual ao motor)

# NOTA: estes valores foram afinados por sweep de self-play do agent7 (ver
# orbit-wars/sweep_agent7.py). A config abaixo venceu a config anterior em 86%
# de 80 partidas held-out. Os comentarios "(sweep: ...)" do agent6 refletem o
# tuning ANTIGO; o agent6 continua usando este mesmo dict como baseline.
A6 = dict(MAX_DIST=50.0, MARGIN=1.2, GARRISON=1, PROD_WEIGHT=4.0,
          ENEMY_BONUS=1, W_MINE=2.5, W_ENEMY_RISK=1.0, MAX_COALITION=3)
A6_THREAT_HORIZON = 30    # ticks de antecedencia pra contar ameacas
A6_SUN_BUFFER = 2.5       # folga em volta do sol
A6_OVERSHOOT = 16.0       # quanto a frota pode passar do alvo se errar
A6_COMET_CLEAR = 2.5      # folga em volta de um cometa (rota)
A6_EARLY_STEPS = 70       # duracao da fase de expansao inicial
A6_EARLY_PROD_BOOST = 4 # peso extra na producao durante o inicio
A6_CLOSE_BONUS = 1.0      # bonus de proximidade extra (sweep agent7: 1.0)
A6_ROTATING_BONUS = 1   # >1 prioriza planetas que GIRAM e COMETAS (teste: 1.0 e melhor)
A6_TARGET_COMETS = False  # incluir cometas como alvo (teste: False e melhor)

A6_CHIP_NEUTRALS = False  # ataque PARCIAL em neutros (teste: False e melhor, concentrar vence)
A6_CHIP_MIN_FRAC = 0.34   # so faz chip se cobrir essa fracao do necessario

def _intercept_n(mine, target, omega, fleet_ships):
    """Mira preditiva orbital usando o tamanho REAL da frota. Corrige o
    'head start': a frota NASCE na borda do planeta (raio+0.1 na direcao do
    tiro), entao chega ~offset/speed ticks mais cedo. Sem isso ela erra
    alvos em MOVIMENTO (chega antes do planeta) e voa pra fora da tela.
    Retorna (aim_x, aim_y, dist, speed)."""
    offset = mine.radius + 0.1
    speed = _fleet_speed(fleet_ships)
    ax, ay = target.x, target.y
    for _ in range(6):
        eta = max(0.0, math.hypot(ax - mine.x, ay - mine.y) - offset) / speed
        ax, ay = _predict(target, eta, omega)
    return ax, ay, math.hypot(ax - mine.x, ay - mine.y), speed


def _comet_tracks(obs):
    """Lista (comet_id, path_index_atual, path) de cada cometa visivel."""
    tracks = []
    for group in _obs_get(obs, "comets", []):
        idx = group.get("path_index", 0)
        for i, pid in enumerate(group.get("planet_ids", [])):
            tracks.append((pid, idx, group["paths"][i]))
    return tracks


def _intercept_comet(mine, comet, comet_tracks, fleet_ships):
    """Mira preditiva pela TRAJETORIA do cometa (ele nao orbita o sol).
    Retorna (aim, dist, speed) ou None se o cometa expira antes da chegada."""
    info = next(((idx, path) for cid, idx, path in comet_tracks if cid == comet.id), None)
    if info is None:
        return None
    idx, path = info
    speed = _fleet_speed(fleet_ships)
    ax, ay = comet.x, comet.y
    for _ in range(6):
        eta = math.hypot(ax - mine.x, ay - mine.y) / speed
        j = idx + int(round(eta))
        if not (0 <= j < len(path)):
            return None                      # some antes de chegar
        ax, ay = path[j][0], path[j][1]
    return ax, ay, math.hypot(ax - mine.x, ay - mine.y), speed


def _swept_hit(A, B, P0, P1, r):
    """Espelha o motor: frota A->B e cometa P0->P1 chegam a < r em algum
    instante do tick? (deteccao CONTINUA, nao por amostra)."""
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    return (-b + sq) / (2.0 * a) >= 0.0 and (-b - sq) / (2.0 * a) <= 1.0


def _route_clear(mine, ax, ay, speed, comet_tracks, ignore_id=None):
    """True se a rota reta ate (ax,ay) -- com overshoot -- nao bate no sol
    nem cruza a trajetoria de um cometa (exceto o cometa-alvo 'ignore_id').
    Fuga de cometa usa o MESMO teste continuo (swept) do motor + margem."""
    d = math.hypot(ax - mine.x, ay - mine.y)
    if d < 1e-9:
        return True
    ux, uy = (ax - mine.x) / d, (ay - mine.y) / d
    end = (ax + ux * A6_OVERSHOOT, ay + uy * A6_OVERSHOOT)
    if _point_seg_dist(SUN_CENTER, (mine.x, mine.y), end) < SUN_RADIUS + A6_SUN_BUFFER:
        return False
    rr = COMET_RADIUS + A6_COMET_CLEAR
    # cobre o voo + o OVERSHOOT (frota que erra o alvo segue reto)
    nticks = int(math.ceil((d + A6_OVERSHOOT) / speed)) + 1
    for k in range(1, nticks):
        f0 = (mine.x + ux * speed * (k - 1), mine.y + uy * speed * (k - 1))
        f1 = (mine.x + ux * speed * k, mine.y + uy * speed * k)
        for cid, idx, path in comet_tracks:
            if cid == ignore_id:
                continue
            j0, j1 = idx + k - 1, idx + k
            if j0 >= 0 and j1 < len(path):
                c0 = (path[j0][0], path[j0][1])
                c1 = (path[j1][0], path[j1][1])
                if _swept_hit(f0, f1, c0, c1, rr):
                    return False
    return True


def _will_hit(mine, ax, ay, speed, target, omega):
    """Simula a frota reta (nascendo na BORDA) contra a ORBITA do alvo, com
    o mesmo teste swept do motor (r = raio do alvo). True se realmente
    intercepta. Evita lancar frotas que vao errar e sair da tela."""
    d = math.hypot(ax - mine.x, ay - mine.y)
    if d < 1e-9:
        return True
    ux, uy = (ax - mine.x) / d, (ay - mine.y) / d
    offset = mine.radius + 0.1
    nticks = int(math.ceil((d + offset + A6_OVERSHOOT) / speed)) + 1
    for k in range(1, nticks):
        f0 = (mine.x + ux * (offset + speed * (k - 1)),
              mine.y + uy * (offset + speed * (k - 1)))
        f1 = (mine.x + ux * (offset + speed * k),
              mine.y + uy * (offset + speed * k))
        t0 = _predict(target, k - 1, omega)
        t1 = _predict(target, k, omega)
        if _swept_hit(f0, f1, t0, t1, target.radius):
            return True
    return False


def _threat_fear(myplanet, enemy_fleets, omega):
    """Naves inimigas que convergem pro meu planeta, prevendo a posicao
    FUTURA do planeta (gira) no ETA da frota. Retorna (total, menor_eta)."""
    total, soonest = 0, float("inf")
    for f in enemy_fleets:
        d = math.hypot(myplanet.x - f.x, myplanet.y - f.y)
        sp = _fleet_speed(f.ships)
        eta = d / sp
        if eta > A6_THREAT_HORIZON:
            continue
        mp = _predict(myplanet, eta, omega)
        fx = f.x + math.cos(f.angle) * sp * eta
        fy = f.y + math.sin(f.angle) * sp * eta
        if math.hypot(fx - mp[0], fy - mp[1]) < myplanet.radius + 4.0:
            total += f.ships
            soonest = min(soonest, eta)
    return total, soonest


def agent6(obs):
    moves = []
    player = _obs_get(obs, "player", 0)
    omega = _obs_get(obs, "angular_velocity", 0.0)
    step = _obs_get(obs, "step", 0)
    planets = [Planet(*p) for p in _obs_get(obs, "planets", [])]
    fleets = [Fleet(*f) for f in _obs_get(obs, "fleets", [])]
    comet_ids = set(_obs_get(obs, "comet_planet_ids", []))
    comet_tracks = _comet_tracks(obs)

    my = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1
                     and p.id not in comet_ids]
    targets = [p for p in planets if p.owner != player
               and (A6_TARGET_COMETS or p.id not in comet_ids)]
    enemy_fleets = [f for f in fleets if f.owner != player]

    # --- MEDO ---
    threat, threat_eta = {}, {}
    for mine in my:
        threat[mine.id], threat_eta[mine.id] = _threat_fear(mine, enemy_fleets, omega)

    free = {}
    for mine in my:
        hold = max(A6["GARRISON"], min(mine.ships, threat[mine.id]))
        free[mine.id] = max(0, mine.ships - hold)

    # --- REFORCO entre planetas ---
    for mine in my:
        shortfall = threat[mine.id] + 1 - mine.ships
        if shortfall <= 0:
            continue
        donors = sorted(
            (dn for dn in my if dn.id != mine.id and free[dn.id] > 0),
            key=lambda dn: math.hypot(dn.x - mine.x, dn.y - mine.y),
        )
        for donor in donors:
            if shortfall <= 0:
                break
            send = min(free[donor.id], shortfall)
            ax, ay, dd, sp = _intercept_n(donor, mine, omega, send)
            if dd / sp >= threat_eta[mine.id]:
                continue
            if not _will_hit(donor, ax, ay, sp, mine, omega):
                continue                               # reforco vai errar -> nao gasta
            if not _route_clear(donor, ax, ay, sp, comet_tracks):
                continue
            moves.append([donor.id, math.atan2(ay - donor.y, ax - donor.x), int(send)])
            free[donor.id] -= send
            shortfall -= send

    # --- ATAQUE/EXPANSAO ---
    early = step < A6_EARLY_STEPS
    prod_w = A6["PROD_WEIGHT"] * (A6_EARLY_PROD_BOOST if early else 1.0)
    enemy_bonus = 1.0 if early else A6["ENEMY_BONUS"]

    def score(t):
        d_mine = _nearest_dist(t, my)
        d_enemy = _nearest_dist(t, enemy_planets)
        val = (t.production * prod_w + 1.0)
        if t.owner != -1:
            val *= enemy_bonus
        if _is_rotating(t) or t.id in comet_ids:    # prioriza moveis
            val *= A6_ROTATING_BONUS
        prox = max(0.0, 1.0 - d_mine / A6["MAX_DIST"])
        val *= (1.0 + prox ** 2 * A6_CLOSE_BONUS)
        cost_mine = 1.0 + (d_mine / 50.0) ** 2 * A6["W_MINE"]
        cost_risk = 1.0 + (1.0 / (1.0 + d_enemy)) * A6["W_ENEMY_RISK"] * 50.0
        return val / (cost_mine * cost_risk)

    for t in sorted(targets, key=score, reverse=True):
        is_comet = t.id in comet_ids
        cand = []
        for mine in my:
            if free[mine.id] < 1:
                continue
            guess = min(free[mine.id],
                        _ships_needed(t, _nearest_dist(t, [mine]), A6["MARGIN"]))
            if is_comet:
                res = _intercept_comet(mine, t, comet_tracks, max(guess, 1))
                if res is None:
                    continue
                ax, ay, d, sp = res
            else:
                ax, ay, d, sp = _intercept_n(mine, t, omega, max(guess, 1))
            if d > A6["MAX_DIST"]:
                continue
            if not is_comet and not _will_hit(mine, ax, ay, sp, t, omega):
                continue                               # vai errar -> nao gasta naves
            if not _route_clear(mine, ax, ay, sp, comet_tracks,
                                ignore_id=(t.id if is_comet else None)):
                continue
            cand.append((mine, ax, ay, d))
        if not cand:
            continue
        cand.sort(key=lambda c: c[3])
        cand = cand[:A6["MAX_COALITION"]]

        coalition, total, needed = [], 0, None
        for (mine, ax, ay, d) in cand:
            coalition.append((mine, ax, ay, d))
            total += free[mine.id]
            needed = _ships_needed(t, d, A6["MARGIN"])
            if total >= needed:
                break
        if needed is None:
            continue
        if total < needed:
            # neutros nao produzem: ondas parciais acumulam ("chipping"),
            # entao vale comprometer naves se cobrimos uma fracao relevante.
            if not (A6_CHIP_NEUTRALS and t.owner == -1
                    and total >= needed * A6_CHIP_MIN_FRAC):
                continue
            remaining = total
        else:
            remaining = needed
        for (mine, ax, ay, d) in coalition:
            if remaining <= 0:
                break
            send = min(free[mine.id], remaining)
            if send < 1:
                continue
            moves.append([mine.id, math.atan2(ay - mine.y, ax - mine.x), int(send)])
            free[mine.id] -= send
            remaining -= send

    return moves

# ============================================================
#  agent7 -- estrategia de BLOCOS + STAGING + ataque SINCRONIZADO
#
#  Reaproveita TODA a logica comprovada do agent6 (defesa + ataque/expansao,
#  com os helpers _intercept_n, _will_hit, _route_clear, _threat_fear,
#  _ships_needed, _nearest_dist) e adiciona dois deltas:
#
#    (A) CLUSTERS: agrupa meus planetas em blocos por proximidade. O reforco
#        defensivo passa a preferir doadores do MESMO bloco (ajuda chega mais
#        perto/rapido).
#    (B) STAGING (pos-ataque): planetas de RETAGUARDA com naves OCIOSAS -- os
#        que nao tinham alvo alcancavel e por isso sobraram com excedente alto
#        -- funilam essas naves, em hop, pro vizinho de bloco mais a frente.
#        Concentra forca na borda SEM roubar nenhuma captura (so age depois que
#        o ataque ja consumiu o que precisava).
#
#  Por que so isso: experimentos mostraram que adicionar ataque "sincronizado"
#  memoryless e staging agressivo PIORAVAM vs agent6 (atrasavam a expansao, que
#  e o que mais importa). Staging CONSERVADOR (so backwater profundo) + reforco
#  por cluster sobem o win-rate vs agent6 pra ~60-67%.
#
#  Foco em 2 jogadores. Tudo dono-agnostico, entao roda em 4p sem quebrar.
# ============================================================
A7_CLUSTER_RADIUS = 32.0     # planetas a <= isso (unidades) ficam no mesmo bloco
A7_STAGE_ENABLE = True
A7_REAR_DIST = 25.0          # so funila planeta a >= isso do alvo mais proximo
A7_STAGE_MIN_SURPLUS = 30    # so funila excedente ocioso acima disso


def _cluster_planets(planets, radius):
    """Single-linkage (union-find): planetas a <= radius caem no mesmo bloco.
    Retorna lista de listas de planetas."""
    n = len(planets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(planets[i].x - planets[j].x,
                          planets[i].y - planets[j].y) <= radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(planets[i])
    return list(groups.values())


def agent7(obs):
    # Base: a logica COMPROVADA do agent6 (defesa + ataque/expansao). Deltas:
    #   (A) reforco prefere doadores do MESMO cluster;
    #   (B) STAGING pos-ataque: planetas com naves OCIOSAS (sem alvo alcancavel)
    #       funilam o excedente pro vizinho de cluster mais a frente (hop),
    #       montando concentracao na borda sem roubar capturas.
    moves = []
    player = _obs_get(obs, "player", 0)
    omega = _obs_get(obs, "angular_velocity", 0.0)
    step = _obs_get(obs, "step", 0)
    planets = [Planet(*p) for p in _obs_get(obs, "planets", [])]
    fleets = [Fleet(*f) for f in _obs_get(obs, "fleets", [])]
    comet_ids = set(_obs_get(obs, "comet_planet_ids", []))
    comet_tracks = _comet_tracks(obs)

    my = [p for p in planets if p.owner == player]
    if not my:
        return moves
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1
                     and p.id not in comet_ids]
    targets = [p for p in planets if p.owner != player
               and (A6_TARGET_COMETS or p.id not in comet_ids)]
    enemy_fleets = [f for f in fleets if f.owner != player]

    # --- MEDO (igual agent6) ---
    threat, threat_eta = {}, {}
    for mine in my:
        threat[mine.id], threat_eta[mine.id] = _threat_fear(mine, enemy_fleets, omega)

    free = {}
    for mine in my:
        hold = max(A6["GARRISON"], min(mine.ships, threat[mine.id]))
        free[mine.id] = max(0, mine.ships - hold)

    # --- CLUSTERS + "frente" (distancia ao alvo mais proximo) ---
    clusters = _cluster_planets(my, A7_CLUSTER_RADIUS)
    cluster_of = {}
    for cl in clusters:
        for p in cl:
            cluster_of[p.id] = id(cl)
    front = {mine.id: (_nearest_dist(mine, targets) if targets else float("inf"))
             for mine in my}

    # --- REFORCO entre planetas (agent6; doadores do mesmo cluster primeiro) ---
    for mine in my:
        shortfall = threat[mine.id] + 1 - mine.ships
        if shortfall <= 0:
            continue
        donors = sorted(
            (dn for dn in my if dn.id != mine.id and free[dn.id] > 0),
            key=lambda dn: (cluster_of[dn.id] != cluster_of[mine.id],
                            math.hypot(dn.x - mine.x, dn.y - mine.y)),
        )
        for donor in donors:
            if shortfall <= 0:
                break
            send = min(free[donor.id], shortfall)
            ax, ay, dd, sp = _intercept_n(donor, mine, omega, send)
            if dd / sp >= threat_eta[mine.id]:
                continue
            if not _will_hit(donor, ax, ay, sp, mine, omega):
                continue
            if not _route_clear(donor, ax, ay, sp, comet_tracks):
                continue
            moves.append([donor.id, math.atan2(ay - donor.y, ax - donor.x), int(send)])
            free[donor.id] -= send
            shortfall -= send

    # --- ATAQUE/EXPANSAO (identico ao agent6) ---
    early = step < A6_EARLY_STEPS
    prod_w = A6["PROD_WEIGHT"] * (A6_EARLY_PROD_BOOST if early else 1.0)
    enemy_bonus = 1.0 if early else A6["ENEMY_BONUS"]

    def score(t):
        d_mine = _nearest_dist(t, my)
        d_enemy = _nearest_dist(t, enemy_planets)
        val = (t.production * prod_w + 1.0)
        if t.owner != -1:
            val *= enemy_bonus
        if _is_rotating(t) or t.id in comet_ids:
            val *= A6_ROTATING_BONUS
        prox = max(0.0, 1.0 - d_mine / A6["MAX_DIST"])
        val *= (1.0 + prox ** 2 * A6_CLOSE_BONUS)
        cost_mine = 1.0 + (d_mine / 50.0) ** 2 * A6["W_MINE"]
        cost_risk = 1.0 + (1.0 / (1.0 + d_enemy)) * A6["W_ENEMY_RISK"] * 50.0
        return val / (cost_mine * cost_risk)

    for t in sorted(targets, key=score, reverse=True):
        is_comet = t.id in comet_ids
        cand = []
        for mine in my:
            if free[mine.id] < 1:
                continue
            guess = min(free[mine.id],
                        _ships_needed(t, _nearest_dist(t, [mine]), A6["MARGIN"]))
            if is_comet:
                res = _intercept_comet(mine, t, comet_tracks, max(guess, 1))
                if res is None:
                    continue
                ax, ay, d, sp = res
            else:
                ax, ay, d, sp = _intercept_n(mine, t, omega, max(guess, 1))
            if d > A6["MAX_DIST"]:
                continue
            if not is_comet and not _will_hit(mine, ax, ay, sp, t, omega):
                continue
            if not _route_clear(mine, ax, ay, sp, comet_tracks,
                                ignore_id=(t.id if is_comet else None)):
                continue
            cand.append((mine, ax, ay, d))
        if not cand:
            continue
        cand.sort(key=lambda c: c[3])
        cand = cand[:A6["MAX_COALITION"]]

        coalition, total, needed = [], 0, None
        for (mine, ax, ay, d) in cand:
            coalition.append((mine, ax, ay, d))
            total += free[mine.id]
            needed = _ships_needed(t, d, A6["MARGIN"])
            if total >= needed:
                break
        if needed is None:
            continue
        if total < needed:
            if not (A6_CHIP_NEUTRALS and t.owner == -1
                    and total >= needed * A6_CHIP_MIN_FRAC):
                continue
            remaining = total
        else:
            remaining = needed
        for (mine, ax, ay, d) in coalition:
            if remaining <= 0:
                break
            send = min(free[mine.id], remaining)
            if send < 1:
                continue
            moves.append([mine.id, math.atan2(ay - mine.y, ax - mine.x), int(send)])
            free[mine.id] -= send
            remaining -= send

    # --- STAGING (delta B): naves OCIOSAS que sobraram funilam pra frente ---
    # So dispara para planetas SEM alvo alcancavel (free ainda alto apos o
    # ataque), entao nunca rouba uma captura. Manda em hop pro vizinho de
    # cluster mais proximo da frente.
    if A7_STAGE_ENABLE:
        for cl in clusters:
            if len(cl) < 2:
                continue
            for donor in cl:
                if threat[donor.id] > 0 or free[donor.id] < A7_STAGE_MIN_SURPLUS:
                    continue
                if front[donor.id] < A7_REAR_DIST:        # ja esta na frente
                    continue
                fwd = [p for p in cl if front[p.id] < front[donor.id] - 1.0]
                if not fwd:
                    continue
                hop = min(fwd, key=lambda p: math.hypot(p.x - donor.x, p.y - donor.y))
                send = free[donor.id]
                ax, ay, dd, sp = _intercept_n(donor, hop, omega, send)
                if not _will_hit(donor, ax, ay, sp, hop, omega):
                    continue
                if not _route_clear(donor, ax, ay, sp, comet_tracks):
                    continue
                moves.append([donor.id, math.atan2(ay - donor.y, ax - donor.x), int(send)])
                free[donor.id] -= send

    return moves


def agent(obs):
    return agent7(obs)
