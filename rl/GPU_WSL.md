# Treinar na GPU (RTX 3060) via WSL2

Guia passo a passo para rodar `jax_ppo.py` na sua GPU usando **WSL2 + Ubuntu**.
O código JAX é agnóstico de device — não muda nada no código; o JAX usa a GPU
sozinho assim que o `jax[cuda12]` enxerga a placa.

## Por que WSL

JAX-GPU **não** tem suporte nativo bom no Windows. O caminho confiável é WSL2
(um Ubuntu real dentro do Windows) com passagem da GPU NVIDIA. PyTorch roda nativo
no Windows, mas nosso pipeline rápido é JAX — então: WSL.

## 1. Driver (no Windows, não no WSL)

- Instale/atualize o **driver NVIDIA do Windows** (GeForce/Studio mais recente).
- **NÃO** instale driver NVIDIA dentro do WSL. O CUDA do WSL usa o driver do host.

## 2. Instalar o WSL2 + Ubuntu (PowerShell como admin)

```powershell
wsl --install -d Ubuntu
wsl --update
```
Reinicie se pedir. Abra o "Ubuntu" no menu iniciar.

## 3. Dentro do Ubuntu (WSL): ambiente Python

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip python3.10-venv

# vá para o projeto
cd /caminho/para/OrbitWarsRL/rl

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt

# orbit_wars nao esta no PyPI (versao recente requer Python 3.11+)
# instala manualmente copiando do GitHub:
git clone --depth=1 --filter=blob:none --sparse https://github.com/Kaggle/kaggle-environments.git /tmp/kenv_git
cd /tmp/kenv_git && git sparse-checkout set kaggle_environments/envs/orbit_wars
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r /tmp/kenv_git/kaggle_environments/envs/orbit_wars "$SITE/kaggle_environments/envs/"
```

> **Dica WSL:** rodar de `/mnt/c/...` funciona, mas I/O é mais lento. Para treinos
> longos, copie o projeto para `~/` (ext4, muito mais rápido).

## 4. Confirmar que a GPU foi detectada

```bash
python3 -c "import jax; print(jax.default_backend(), jax.devices())"
# esperado: gpu [CudaDevice(id=0)]
```
O `jax_ppo.py` também imprime isso no começo:
`JAX backend: gpu | devices: [CudaDevice(id=0)]`

Se aparecer `cpu`, o jax[cuda12] não enxergou a placa — cheque o driver do Windows
e `nvidia-smi` dentro do WSL (deve listar a placa).

## 5. Treinar

### Sanidade rápida (vs greedy, ~10 min na 3060)

```bash
python3 jax_ppo.py --iters 60 --games 2048 --horizon 128 \
    --minibatch 32768 --eval-every 5 --save ckpt_greedy.pkl
```

Deve chegar a **~95%+ winrate vs greedy** em ~60 iterações.

### Self-play sério (overnight), retomável, com avaliação vs agent7

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python3 jax_ppo.py --iters 4000 --games 4096 --horizon 128 --selfplay \
    --minibatch 32768 --eval-every 25 \
    --eval-heur-every 100 --eval-heur-games 16 --eval-heur-mode agent7 \
    --save ckpt_selfplay.pkl
```

### Retomar de onde parou (checkpoint completo: pesos + Adam)

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python3 jax_ppo.py --iters 4000 --games 4096 --horizon 128 --selfplay \
    --minibatch 32768 --eval-every 25 \
    --eval-heur-every 100 --eval-heur-games 16 \
    --load ckpt_selfplay.pkl --save ckpt_selfplay.pkl
```

O `--save` grava pesos **e** estado do Adam (momentum/variância) a cada eval e no
fim. O `--load` restaura tudo — retomada exata sem re-aquecimento do otimizador.

## 6. Desempenho medido (RTX 3060 12 GB)





> O ganho de games=4096 vs 2048 vem de melhor utilização da GPU (batch maior).
> A VRAM com games=4096 + horizon=128 + sem arrays de frota fica em ~5–6 GB.

## 7. Controle de memória

| Problema | Sintoma | Solução |
|---|---|---|
| OOM no rollout | crash no 1º iter, mensagem XLA | reduza `--games` ou `--horizon` |
| OOM no update | crash depois do rollout | reduza `--minibatch` |
| OOM por fragmentação | crash após centenas de iters, "memory fragmentation" | use `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (ver abaixo) |
| RAM do WSL alta | sistema lento | veja seção 8 |

### OOM por fragmentação (treinos longos)

O alocador padrão do XLA (BFC) fragmenta a VRAM ao longo de horas de treino,
eventualmente falhando em alocar blocos contíguos grandes. **Solução:** usar o
alocador `platform` (CUDA malloc direto), que não fragmenta:

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform python3 jax_ppo.py ...
```

Ou exportar permanentemente no shell:
```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

> Este é o flag recomendado para **todos os treinos longos** (>200 iters).

### O update não é mais o gargalo

Em versões anteriores, o `update` usava `lax.scan(epochs)` dentro de um `@jax.jit`,
materializando gradientes sobre `T×B` amostras inteiras — causava OOM com games≥2048.

**Fix aplicado:** o loop de epochs/minibatches roda em Python; cada `mb_step` JIT'd
processa só `--minibatch` amostras. VRAM do update = constante e pequena,
independente de `--games`.

## 8. RAM do WSL2

WSL2 pode consumir RAM agressivamente. Para limitar (arquivo no Windows):

`C:\Users\<seu_usuario>\.wslconfig`:
```ini
[wsl2]
memory=12GB    # deixa 4 GB pro Windows (ajuste conforme sua RAM total)
swap=4GB
```

Reinicie o WSL após (`wsl --shutdown` no PowerShell).

## 9. Limitações que a GPU NÃO resolve

- Motor JAX é **2 jogadores, sem cometas** (v1). Avaliação final/deploy usam o
  motor real (com cometas).
- O oponente no rollout é **greedy** (baseline) ou **self-play** (snapshot). O
  `agent7` é avaliado periodicamente via `--eval-heur-every` no motor real (lento,
  ~10s/game), mas não é usado no rollout de treino.
