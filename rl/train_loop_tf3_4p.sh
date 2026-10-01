#!/bin/bash
# 4-PLAYER da politica transformer v3. Warm-start do v3 2p (ckpt_tf3.pkl).
# Oponentes por-assento sorteados INDEPENDENTE de: pool de self-play 4p + 10% do
# ckpt_tf3_init.pkl (v3_init). Cada partida pode ter qualquer combinacao nos 3
# assentos (1 v3_init + 2 self-play, 3 self-play, 2 v3_init + 1 self-play, etc).
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

LOAD_INIT=ckpt_tf3.pkl        # warm-start: v3 2p ja treinado
SAVE=ckpt_tf3_4p.pkl          # destino 4p (nao toca o 2p)
ITERS=2000
LOG=/tmp/ppo_train_4p.log
MAX_RESTARTS=40
: > "$LOG"

run_once () {
  local extra="$1"
  python3 jax_ppo4_tf.py --iters "$ITERS" --games 1024 --horizon 128 --epochs 2 \
    --arch transformer --width 112 --tf-layers 2 --heads 4 \
    --pool-size 20 --pool-every 10 \
    --fixed-opp "ckpt_tf3_init.pkl" --fixed-opp-frac 0.10 \
    --minibatch 8192 --eval-every 25 --eval-games 512 --gamma 0.997 \
    --lr-schedule --lr 2e-4 --ent 0.01 \
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
    echo "TREINO TF3-4P COMPLETO." | tee -a "$LOG"; break; fi
  if [ "$did" -lt 5 ]; then
    echo "ABORTANDO: run $r avancou so $did iters." | tee -a "$LOG"; break; fi
  echo "aguardando VRAM..." | tee -a "$LOG"
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1500 ] && break; sleep 3
  done
done
echo "===== WRAPPER TF3-4P FIM $(date -Is) =====" | tee -a "$LOG"
