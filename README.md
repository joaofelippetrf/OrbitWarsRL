# OrbitWarsRL

Agente de **Reinforcement Learning** para o jogo **Orbit Wars** (competição Kaggle), um
RTS em tempo real em espaço 2D contínuo onde bots conquistam planetas que orbitam um sol.
A cada turno o agente observa o estado e devolve jogadas no formato
`[planeta_origem_id, ângulo, n_naves]`.

Este repositório contém todo o caminho percorrido. O motor de física reimplementado em JAX
para treinar em GPU, o pipeline de aprendizado por reforço com self-play, as arquiteturas de
rede testadas e o modelo final exportado para submissão no Kaggle, escrito em numpy puro para
não depender de bibliotecas pesadas.

O modelo em produção é o **`rl_submission_v3.py`**, o transformer que chamamos de "v3", que
foi o melhor no motor real do Kaggle entre tudo que treinamos. As seções abaixo explicam o
pipeline, as arquiteturas testadas e por que a v3 venceu variantes posteriores e maiores.

---

## 1. O jogo e o desafio de RL

Orbit Wars é economicamente denso e adversarial.

- Planetas orbitam o sol em movimento circular determinístico, então mirar exige prever onde
  o alvo estará no momento em que a frota chegar.
- Planetas com dono produzem naves durante a viagem, então o custo de captura precisa ser
  calculado para o instante de chegada, e não para o presente.
- Frotas somem se a rota cruza o sol ou sai do tabuleiro, gerando naves desperdiçadas.
- Uma captura só acontece se a frota que chega for maior que a guarnição que defende o alvo.
- O jogo tem simetria de rotação. Em duas equipes um jogador é o outro girado em meia volta
  em torno do centro, e em quatro equipes os pontos de partida são giros de um quarto de
  volta entre si.

As regras completas do jogo estão na documentação da competição Orbit Wars no Kaggle.

---

## 2. Ponto de partida heurístico

O projeto começou por agentes heurísticos, antes de qualquer aprendizado por reforço. O
objetivo foi construir intuição sobre o jogo, como prever a mira, calcular o custo de captura
na chegada, concentrar naves de vários planetas num mesmo alvo e segurar tropas para defesa.
Essa heurística virou depois o baseline e o oponente de treino que a rede precisava superar.

---

## 3. O motor em JAX para treino em GPU

Treinar aprendizado por reforço exige milhões de passos, e rodar o motor original do jogo,
escrito em Python, é lento demais. Por isso reimplementamos a física inteira em JAX, de forma
vetorizada e compilada, rodando milhares de partidas em paralelo na GPU. Há uma versão para
duas equipes e outra para quatro.

Antes de treinar, validamos que esse motor reproduz o jogo oficial de forma exata. Em vários
mapas e por dezenas de passos, o número de naves e os donos de cada planeta batem sem nenhuma
diferença. Batem também as constantes do jogo, a ordem em que as coisas acontecem no turno, a
fórmula de velocidade das frotas, a rotação orbital dos planetas e a destruição de frotas que
cruzam o sol.

A única diferença conhecida são os cometas, que o jogo real cria em órbitas elípticas e que a
nossa reimplementação não modela. Na hora de jogar, o agente simplesmente não escolhe cometas
como alvo, já que a física circular que ele conhece preveria a órbita elíptica de forma
errada.

---

## 4. O pipeline de treino

O treino usa PPO com alguns cuidados que fizeram diferença.

- Cada planeta é tratado como uma decisão própria, com seu próprio crédito de recompensa, em
  vez de somar todas as jogadas do turno numa ação única. Isso deixa o aprendizado muito mais
  limpo, porque cada planeta é premiado ou punido pela sua própria escolha.
- A rede tem duas decisões por planeta. Uma escolhe para onde atacar, comparando todos os
  planetas entre si por atenção, e a outra escolhe quantas das naves livres enviar.
- As entradas da rede foram desenhadas com cuidado, com informações como a rentabilidade de
  um alvo, a distância ao inimigo mais próximo, o valor que ainda resta a capturar e, a partir
  da v3, as frotas que já estão em voo, para a rede ter noção dos reforços a caminho.
- Como o aprendizado só acontece pelo lado de um dos jogadores, todo estado é girado para a
  perspectiva desse jogador antes de entrar na rede, e a jogada é girada de volta na saída.
  Acertar essa rotação foi decisivo, porque sem ela a rede jogava mal quando calhava de ser o
  segundo jogador.
- O treino é por self-play com uma liga de versões antigas. O agente enfrenta cópias de si
  mesmo de várias gerações, com preferência por oponentes do nível certo de dificuldade, e
  guarda versões permanentes para não esquecer o que já aprendeu. De vez em quando ainda
  enfrenta a heurística, para não esquecer como bater os bots clássicos.
- A recompensa principal é vencer ou perder a partida. Além dela, há pequenos incentivos
  intermediários por ter vantagem de naves, de produção e de número de planetas, e uma
  punição por desperdiçar frota. Esses incentivos são pequenos de propósito, para orientar
  sem dominar o objetivo de vencer.
- Para não treinar só as aberturas, muitas partidas começam já no meio do jogo, e a coleta
  aproveita os dois lados de cada partida.

Os cuidados de memória para rodar tudo numa placa de 12 GB dentro do WSL2 estão documentados
em `rl/GPU_WSL.md`.

---

## 5. Arquiteturas testadas

| # | Modelo | Ideia principal | Params | Papel |
|---|---|---|---|---|
| 1 | **DeepSets** | processa os planetas de forma simétrica, sem atenção entre eles | ~380 k | Bate greedy e random, mas passivo contra heurísticas fortes. Base histórica. |
| 2 | **Transformer v1** | atenção entre planetas, duas camadas | 307 k | Primeiro transformer, introduz a comparação de cada planeta com os outros. |
| 3 | **Transformer v3** ⭐ | atenção com duas camadas, mais informação das frotas em voo | 307 k | Modelo final. Tem redes separadas para as partidas de duas e de quatro equipes. |
| 4 | **SP2** | rede mais profunda e mais informação de tempo das frotas | 435 k | Revisão maior do pipeline. Melhor na avaliação interna, mas não superou a v3 no motor real. |
| 5 | **XL** (planejado) | versão bem maior da mesma ideia | ~602 k | Configurado mas nunca treinado, removido nesta limpeza. |

As diferenças da v3 para a SP2, que acabaram não compensando no motor real, foram uma camada
de atenção a mais, informação sobre quanto tempo cada frota em voo leva para chegar, uma
normalização extra na saída, uma rede dedicada para estimar quão boa é a situação atual e uma
forma mais elaborada de decidir quantas naves enviar.

Para rodar no Kaggle sem depender de JAX, o modelo final foi reescrito em numpy puro e teve os
pesos guardados dentro do próprio arquivo de submissão. Essa reescrita foi conferida contra a
versão em JAX e dá o mesmo resultado.

---

## 6. Por que a v3 se saiu melhor no motor real

A v3 é menor e mais simples que a SP2, e ainda assim venceu no motor oficial. As razões
prováveis, ancoradas no que foi medido.

1. **A avaliação interna não é comparável ao motor real.** Os ganhos da SP2 foram medidos num
   ambiente rápido de treino, que gera mapas diferentes do jogo oficial para a mesma semente.
   Ir melhor nesse ambiente interno não garante ir melhor no jogo de verdade, e foi
   exatamente o que aconteceu.
2. **Treinar demais contra si mesmo especializou a rede em enfrentar cópias de si mesma.** A
   SP2 passou mais tempo em self-play puro, com menos exposição à heurística. Isso costuma
   gerar uma política forte contra si mesma, mas menos robusta contra outros estilos de jogo e
   contra o motor real. A v3 manteve mais treino contra a heurística.
3. **A forma de decidir quantas naves enviar ficou mais complexa sem ganho real.** A v3 usa
   uma escolha simples entre poucas opções, enquanto a SP2 experimentou formas mais elaboradas
   que só adicionaram instabilidade sem melhorar o resultado.
4. **Vantagem concreta nas partidas de quatro jogadores.** A v3 carrega uma rede treinada
   especificamente para quatro equipes e a usa nessas partidas. A SP2 só tinha a rede de duas
   equipes, adaptada às de quatro por rotação, tratando os três inimigos como um bloco só.

A lição é que mais parâmetros e mais informação ajudaram nas métricas de treino, mas o
gargalo real era a capacidade de generalizar para o jogo de verdade, e não o tamanho da rede.
O modelo mais simples e mais bem ancorado transferiu melhor.

---

## 7. Layout do repositório

```
OrbitWarsRL/
├── rl_submission_v3.py     # MODELO FINAL, submissão Kaggle em numpy puro com pesos embutidos
├── submission.py           # Bot heurístico, baseline e sparring
├── README.md
└── rl/                     # Pipeline de RL
    ├── jax_env.py, jax_env4.py       # motor JAX para 2 e 4 equipes, paridade auditada
    ├── jax_ppo.py, jax_ppo4_tf.py    # treino por reforço para 2 e 4 equipes
    ├── tf_numpy.py                    # versão numpy da rede, usada na exportação
    ├── owenv.py, features.py          # apoio à avaliação
    ├── ckpt_tf3.pkl, ckpt_tf3_4p.pkl  # pesos da v3, para 2 e 4 equipes
    ├── train_loop_tf3.sh, *_4p.sh     # scripts que treinaram a v3
    └── GPU_WSL.md, requirements-gpu.txt
```

---

## 8. Como rodar

O modelo final é um único arquivo autocontido. No Kaggle basta submeter `rl_submission_v3.py`,
que já traz os pesos embutidos e não depende de nada além de numpy.

Para treinar do zero em GPU, veja o setup do ambiente em `rl/GPU_WSL.md`.

```bash
cd rl && source .venv/bin/activate
./train_loop_tf3.sh        # transformer v3, para duas equipes
./train_loop_tf3_4p.sh     # v3, para quatro equipes
```

Para checar que o motor em JAX reproduz o jogo oficial.

```bash
cd rl && python3 jax_env.py parity
```
