#!/bin/bash
# Transformer v3: WARM-START do v1 (ckpt_tf3_init = v1 + 2 colunas zeradas) com a
# nova visao de FROTAS EM VOO (--feat-inbound -> 16 feats). Resolve o overcommit
# (mandar 2a leva antes da 1a chegar) e da defesa antecipada.
# v3 comeca IDENTICO ao v1 e aprende a usar inbound por cima.
# Mesma receita: playbook + self-play 100% (pool 20 snaps x 10 = janela 200), refresh 150.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

LOAD_INIT=ckpt_tf3_init.pkl   # v1 com Dense_0 expandido (14->16, novas zeradas)
SAVE=ckpt_tf3.pkl             # destino: nao toca v1 nem v2
ITERS=2000
LOG=/tmp/ppo_train.log
MAX_RESTARTS=40
: > "$LOG"

run_once () {
  local extra="$1"
  python3 jax_ppo.py --iters "$ITERS" --games 1024 --horizon 128 --epochs 2 \
    --arch transformer --width 112 --tf-layers 2 --heads 4 \
    --feat-inbound \
    --opp nearest --pool-size 20 --pool-every 10 --pool-heur-frac 0.0 \
    --pool-refresh 150 \
    --fixed-opp "ckpt_tf3_init.pkl" --fixed-opp-frac 0.10 \
    --eval-heur-every 1000 --eval-heur-mode agent7 --eval-heur-games 12 \
    --minibatch 8192 --eval-every 25 --gamma 0.997 \
    --lr-schedule --lr 2e-4 \
    --ent 0.03 --ent-end 0.005 --ent-frac 0.01 \
    --shape 0.0 --shape-prod 0.0 --shape-planets 0.0 \
    --shape-waste 0.05 --waste-margin 2 \
    $extra --save "$SAVE"
}

for r in $(seq 1 $MAX_RESTARTS); do
  if [ -f "$SAVE" ]; then LOAD="$SAVE"; else LOAD="$LOAD_INIT"; fi
  echo "===== RUN $r START (load=$LOAD save=$SAVE) $(date -Is) =====" | tee -a "$LOG"
  start_it=$(grep -cE '^it ' "$LOG" 2>/dev/null || echo 0)
  run_once "--load $LOAD" >> "$LOG" 2>&1
  end_it=$(grep -cE '^it ' "$LOG" 2>/dev/null || echo 0)
  did=$(( end_it - start_it ))
  echo "===== RUN $r END iters~=$did $(date -Is) =====" | tee -a "$LOG"
  if grep -q "checkpoint final salvo" "$LOG"; then
    echo "TREINO TF3 COMPLETO." | tee -a "$LOG"; break; fi
  if [ "$did" -lt 5 ]; then
    echo "ABORTANDO: run $r avancou so $did iters." | tee -a "$LOG"; break; fi
  echo "aguardando VRAM..." | tee -a "$LOG"
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1500 ] && break; sleep 3
  done
done
echo "===== WRAPPER TF3 FIM $(date -Is) =====" | tee -a "$LOG"
