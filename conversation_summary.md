# Conversation Summary – Budget, Step‑Size & Rewire Radius Choices

## 1. Parameter table (current script)

| Planner (key) | Function | `max_iter` / generations | `goal_sample_rate` | `connect_circle_dist` | `step_size` / `path_resolution` | Curvature `kappa_max` | Post‑process | Stop criterion |
|---|---|---|---|---|---|---|---|---|
| `rrt_star` | `run_rrt_star → run_rrt` | 2000 | 20 | **2.0** | *default of `RRTPlanner` (~0.5 m)* | `config.kappa_max` | – | max‑iter |
| `rrt_star_smooth` | `run_rrt_star_smooth` | 2000 | 20 | **2.0** | *default* | `config.kappa_max` | **B‑spline** (`n_waypoints=20, s=0.015`) | max‑iter + one spline |
| `rrt_star_dubins` | `run_rrt_star_dubins` | 2000 | 20 | **4.5** | `path_resolution=0.05` | `config.kappa_max` | – | max‑iter |
| `modified_dubins_rrt_star_ccpoa` | `run_modified_dubins_rrt_star_ccpoa` | 1500 (RRT*) + 5 (CCPOA) | 20 | **4.5** | `step_size=0.05` (both) | `config.kappa_max` | CCPOA (5 iters) | max‑iter RRT* + CCPOA |
| `rrt_star_asv` | `run_rrt_star_asv` | 5000 | 30 | **4.5** | `step_size=0.05` | `config.kappa_max` | – | max‑iter |
| `de2d_nurbs` | `run_de2d_nurbs` | 450 gen × 100 pop = 45 k evals | – | – | – (DE) | `config.kappa_max` | – | fixed generations |
| `pso2d_nurbs` | `run_pso2d_nurbs` | 500 gen × 50 pop = 25 k evals | – | – | – (PSO) | `config.kappa_max` | – | fixed generations |

## 2. Step‑size adequacy
- **Dubins‑based planners** (`path_resolution` / `step_size = 0.05 m`) → appropriate (≈ 0.7 × robot radius, matches minimum turning radius 0.73 m).
- **Plain RRT\*** uses the *default* step‑size of `RRTPlanner` (~0.5 m). For a 4.2 × 4.2 m workspace with obstacles as small as 0.08 m, this is **too large**; it can skip narrow passages. Recommended: **0.10–0.15 m** (≈ 1.5–2 × robot radius).

## 3. Rewire radius (`connect_circle_dist`)
- Theoretical asymptotically optimal radius for 2‑D RRT\* with 2000 nodes ≈ 3–4 m. Current **2.0 m** is conservative → slower convergence.
- For Dubins (3‑D state) 4.5 m is near the theoretical value.
- **Worth testing variations** to show the choice is not arbitrary.

### Suggested grid
| Planner | `connect_circle_dist` values |
|---|---|
| `rrt_star` / `rrt_star_smooth` | 1.5 m, **2.0 m**, 3.0 m |
| Dubins variants | 3.5 m, **4.5 m**, 6.0 m |

## 4. Budget justification (already documented)
- Sampling‑based planners: fixed `max_iter` (2000 / 5000) → comparable wall‑clock (~2 s vs 30‑100 s).
- Evolutionary planners: fixed evaluations (≈ 30‑45 k) → literature‑based convergence region.

## 5. Next steps (to be implemented)
1. Reduce RRT* step‑size to 0.10 m (expose via `extra_kw`).
2. Add new planner entries in `PLANNERS` for each `connect_circle_dist` variant (and optionally step‑size variants).
3. Keep all other parameters identical so the only difference is the hyper‑parameter under test.
4. Re‑run the *progressive* experiment (30 seeds) – extra ≈ 5 min total.

---

## 6. Modificações aplicadas nesta sessão (2026‑10‑16)

### 6.1 Correção de *divide‑by‑zero* na curvatura numérica
`compare_rrt_de2d_nurbs.py`, `max_curvature_numerical` (linha ~50):
- **Antes:** `k = np.where(denom > 1e-12, 2.0 * cross / denom, 0.0)`.
- **Problema:** `np.where` avalia **os dois ramos**, então `2.0 * cross / denom`
  era calculado também onde `denom == 0` (pontos repetidos/colineares),
  gerando `RuntimeWarning: invalid value encountered in divide`.
- **Depois:** divisão segura só onde o denominador é válido:
  ```python
  k = np.zeros_like(denom)
  np.divide(2.0 * cross, denom, out=k, where=denom > 1e-12)
  ```

### 6.2 `extended_start_goal` (garantia de orientação no RRT)
Objetivo: garantir que a trajetória chegue ao `start` e saia do `goal`
exatamente com `th_start` / `th_goal`, acrescentando trechos retos tangentes.

**Parâmetros (já existentes em `scenario_config.py`):**
- `config.extended_start_goal` (bool, default `False`).
- `config.extended_start_goal_dist` (float, default `None` → usa `2 * config.radius`).

**Construção (atualizada — S_ext à frente, G_ext atrás):**
- `S_ext = start + d·[cos(th_start), sin(th_start)]` (à frente do start).
- `G_ext = goal  − d·[cos(th_goal),  sin(th_goal)]`  (atrás do goal).
- Caminho final: `S_ext → start → (caminho do RRT) → goal → G_ext`, com os
  trechos retos amostrados em `step_size`.
- Usa `cos`/`sin` do ângulo desejado, portanto correto nos **4 quadrantes**
  (0°, 90°, 180°, 270°, ...). Validado para th em {0,45,90,135,180,225,270,315}.
- **Atenção:** com `S_ext` à frente, o sentido de deslocamento do trecho
  `S_ext → start` é `th_start + 180°`. Isso foi escolhido explicitamente pelo
  usuário (a convenção do NURBS em `de2d_nurbs.py:25-42`, por outro lado,
  coloca os pontos de controle iniciais no sentido `+th_start`).

**Implementação:**
- `rrt_based/rrt_planner_modified.py`
  - Novo helper `extend_path_start_goal(path, start, goal, th_start, th_goal,
    dist, step_size=0.05)` (seção “1b. EXTENDED START / GOAL”).
  - Novo helper `_extended_start_goal_dist(config)` (default `2*radius`).
  - `RRTPlanner.run`: aplica a extensão quando `config.extended_start_goal`
    está ligado (antes de guardar `self._path`). Cobre todos os planners que
    passam pelo wrapper (`rrt_star`, `rrt_star_dubins`, `bit_star_*`,
    `modified_dubins_rrt_star`, `rrt_star_asv`).
- `compare_rrt_de2d_nurbs.py`
  - Importa `extend_path_start_goal` e `_extended_start_goal_dist`.
  - Novo helper `apply_extended_start_goal(path, config, step_size=0.05)`.
  - Aplicado nos runners que instanciam o planner diretamente:
    `run_modified_dubins_rrt_star`, `run_modified_dubins_rrt_star_ccpoa`
    (no caminho pós‑CCPOA e no retorno antecipado) e `run_rrt_star_asv`.

**Habilitar:**
```python
config.extended_start_goal = True
config.extended_start_goal_dist = None   # None -> 2 * config.radius
```

**Observação:** `rrt_based/rrt_star_dubins.py` (usado por `run_rrtdubins2.py`)
é um arquivo separado e **não** foi alterado.

**Bug de orientação (corrigido):** o `generate_final_course` do **RRT\* puro**
devolve o caminho na ordem **goal→start**, enquanto os planners Dubins/ASV
devolvem **start→goal**. O helper assumia start→goal, então no RRT\* puro a
extensão era anexada invertida (trechos retos trocados). Correção em
`extend_path_start_goal`: se `path[0]` estiver mais perto do `goal` do que do
`start`, o corpo é invertido antes de anexar os segmentos.

### 6.2.1 Ativação para todas as variantes RRT\* não‑Dubins
`compare_rrt_de2d_nurbs.py`:
- `run_rrt` (usado por `rrt_star` e `rrt_star_ccdist_*`) e
  `run_rrt_star_smooth`, `run_rrt_star_smooth_ccdist_1_5/2_5/3_0` passam a
  setar `config.extended_start_goal = True`.
- A lista `PLANNERS` foi reativada para todas as variantes **não‑Dubins**:
  `rrt_star`, `rrt_star_smooth`, `rrt_star_ccdist_{1_5,2_5,3_0}`,
  `rrt_star_smooth_ccdist_{1_5,2_5,3_0}` (as Dubins continuam comentadas).
- **Verificado:** os 8 planners passam a iniciar em
  `S_ext = start − 0.146·[cos th_s, sin th_s]` e terminar em
  `G_ext = goal + 0.146·[cos th_g, sin th_g]` (com `radius = 0.073`).

### 6.3 Correção do bug de suavização (B‑spline) — fallback silencioso
**Sintoma:** o caminho “suavizado” passava exatamente pelos mesmos pontos do
RRT bruto (sem remover quinas).

**Causa raiz:** caminhos do RRT\* podem conter **pontos consecutivos
duplicados**. O `scipy.interpolate.splprep` rejeita isso com
`ValueError: Invalid inputs`. O helper `smooth_path_rrt`
(`compare_rrt_de2d_nurbs.py`) engolia a exceção (`except Exception: return
None`) e os runners caiam no fallback, **devolvendo o caminho bruto**.

**Correção:** `RRTPlanner.smooth_path` (`rrt_based/rrt_planner_modified.py`):
- Remove pontos consecutivos duplicados antes do `splprep`:
  ```python
  keep = np.concatenate(([True], np.any(np.abs(np.diff(full, axis=0)) > 0, axis=1)))
  full = full[keep]
  ```
- Faz *downsampling* por **comprimento de arco** (em vez de por índice), para
  representar quinas (regiões esparsas) de forma uniforme.
- Protege contra entradas degeneradas (`< 2` pontos).

**Verificação:** nos cenários onde antes o suavizado era idêntico ao bruto
(seeds 1 e 3 do teste), agora o B‑spline devolve 500 pontos com desvio
máximo de ~0.07–0.17 m em relação ao raw, arredondando as quinas.

**Caveat (a decidir):** como o B‑spline pode cortar caminho, o caminho
suavizado pode **invadir um obstáculo** que o raw contornava. Os runners já
marcam `collision_free`. Se desejado, pode‑se adicionar suavização
*consciente de colisão* (refinar/expulsar pontos do obstáculo).

### 6.4 Medição de tempo justa sob `joblib` (`n_jobs=21` em 32 cores)
**Problema:** cada runner mede tempo de **parede** (`time.perf_counter`). Com
`joblib` em paralelo, a contenção de CPU infla o tempo de cada tarefa de forma
desigual → comparação injusta (ex.: `rrt_star` parecia mais lento que
`rrt_star_smooth`, mas ambos fazem exatamente o mesmo planejamento).

**Soluções aplicadas (`compare_rrt_de2d_nurbs.py`):**
1. **Tempo de CPU** (`time.process_time`) medido em `_run_single_task`, em
   torno da chamada do runner. É insensível à contenção (conta só a CPU do
   processo). Contudo, o custo de *wall* continua registrado.
   - Novas colunas no `record`: `cpu_time` e `wall_time`.
   - `elapsed` passou a ser **o tempo de CPU** (métrica justa entre planners).
2. **Evitar oversubscription de BLAS/OpenMP:** no topo do módulo (antes do
   `import numpy`) fixamos com `setdefault`:
   ```
   OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = NUMEXPR_NUM_THREADS = 1
   ```
   Sem isso, 21 workers × N threads BLAS = centenas de threads, degradando
   wall-clock e a própria medida.
3. `backend='loky'` explícito; `print_summary` agora mostra
   `avg_cpu_time` e `avg_wall_time`.
4. Log verboso por tarefa passou a exibir `cpu=…s wall=…s`.

**Justificativa observada:** instrumentando, `rrt_star` e `rrt_star_smooth`
criam a mesma `RRTStar` com `max_iter=3000`, `connect_circle_dist=2.0`,
`expand_dis=0.3` e **o mesmo nº de nós (2135)** — não há diferença
algorítmica; qualquer diferença de tempo era ruído de medição.

**Análise/plots (`analyze_comparison.py`):**
- Coluna unificada `time_s` criada em `main`: usa `cpu_time` (com
  `fillna(elapsed)`) quando existir; senão cai para `elapsed` (dados antigos).
  Mensagem impressa indica qual foi usada.
- `rank_within_seed`, `print_time_stats`, `print_time_boxplot`,
  `print_latex_table` passaram a usar `time_s`. O boxplot agora rotula eixo
  como "CPU time (s)" (arquivo `boxplot_elapsed.pdf` mantido por
  compatibilidade).
- Observação: `plot_comparison.py` **não** trata tempos (não plota/usa
  `elapsed`/`cpu_time`); nada a alterar lá.

### 6.5 ALERTA: contenção de memória infla o tempo (n_jobs alto)
Medição controlada com a **mesma tarefa** (`rrt_star`, 90 tarefas, 24 cores):

| n_jobs | mean CPU (s) | mean wall (s) | inflação |
|-------:|-------------:|--------------:|---------:|
| 1      | 2.88         | 2.88          | —        |
| 4      | 2.87         | 2.87          | ~0 %     |
| 8      | 4.05         | 4.51          | ~40 %    |
| 21     | 5.45         | 5.46          | ~88 %    |

- `cpu == wall` em todos os casos → **não** é multithreading; é **contenção de
  banda de memória/cache**: 21 processos com arrays numpy grandes disputando
  memória gastam mais ciclos de CPU para o mesmo trabalho.
- Portanto `time.process_time()` **não é imune** a carga; só elimina o efeito
  de multithreading. A banda satura por volta de **4 workers** neste problema.
- **Recomendação para tempos do artigo:** rodar com `n_jobs` baixo (≤4) ou
  sequencial. Com `n_jobs=21` o tempo por tarefa é ~2× maior e a penalização
  não é uniforme entre planners.
- Os summaries antigos `comparison_results_paper` (~5.3–5.85 s) e
  `comparison_results_paper2` foram gravados sob essa inflação; o valor real
  (n_jobs=1) é ~2.9 s. O antigo `comparison_results/progressive` (~1.5 s) usou
  **outro pool de obstáculos** (comprimentos de caminho diferentes), logo não é
  comparação válida.

### 6.6 Redução (½) dos pontos da curva usados em colisão/curvatura
Onde estava definida a densidade de pontos (usada em **colisão** e
**curvatura**):

| Algoritmo | Variável | Onde | Antes → Agora |
|-----------|----------|------|---------------|
| RRT suavizado (B-spline) | `n_points` | `RRTPlanner.smooth_path` (`rrt_based/rrt_planner_modified.py:2018`, `u=linspace(0,1,n_points)`) | 500 → **250** |
| DE2D / PSO (NURBS) | `nsampling` | `_run_single_task` → `config.nsampling` → `de2d_nurbs.py:compute_nurbs_basis` (colisão linha 280, curvatura linha 255) | 100 → **50** |
| RRT busca (amostragem de aresta/nó) | `path_resolution` / `step_size` | `compare_rrt_de2d_nurbs.py` (0.05 / 0.10) | inalterado |

- `smooth_path_rrt` ganhou parâmetro `n_points=250` (passado a
  `smooth_path`); todos os runners de RRT suavizado se beneficiam sem tocar no
  default global do planner.
- Efeito medido (`prog05_seed0`, n_jobs=1): DE2D 0.96s → 0.66s (~31%),
  PSO ~3.8s → ~3.4s (~11%). Para os RRT o `n_points` só afeta o
  pós-processamento (a busca domina), então o ganho é pequeno lá.
- Cuidado: menos pontos = discretização mais grosseira para colisão/curvatura
  (pode mascarar penetrações finas). Reverter = `n_points=500`, `nsampling=100`.

### 6.7 `step_size` vs `expand_dis` vs `connect_circle_dist` (RRT* não-Dubins)
- No branch não-Dubins do `RRTPlanner` (`rrt_planner_modified.py:1917-1933`)
  o **`step_size` nunca é lido**: só `expand_dis` e `path_resolution`. O nome
  `step_size` existe apenas para os planners **Dubins/ASV**.
- `expand_dis` = comprimento de extensão ("steer") da árvore em `steer()`
  (`:366-386` / `:540-560`). Altera **qualidade da busca**, não o tempo (com
  `search_until_max_iter=True` o tempo é limitado por `max_iter`).
- `path_resolution` = resolução de interpolação do segmento para checagem de
  colisão; não altera a topologia da árvore.
- `connect_circle_dist` (`ccdist`) = **raio de vizinhança do RRT***:
  `find_near_nodes` usa `r = ccdist * sqrt(log(nnode)/nnode)` (limitado a
  `ccdist`), alimentando `choose_parent` e `rewire` (`:566-604`). Maior raio →
  mais reconexões → caminho tende a ficar mais curto (melhor otimalidade) e
  tempo por iteração maior. **Altera a qualidade do path.**
- **Correção aplicada** em `compare_rrt_de2d_nurbs.py`: as variantes
  `rrt_star_ccdist_*` e `rrt_star_smooth_ccdist_*` e `run_rrt_star_smooth`
  passaram a usar `expand_dis=RRT_EXPAND_DIS` (constante `=0.3`, igual à base)
  em vez do `step_size=0.10` inócuo. Assim as variantes isolam o efeito do
  `ccdist`.
- Medição `prog07_seed0` (n_jobs=1): `ccdist` 1.5/2.5/3.0 → length
  3.80/3.68/3.68 e `cv` 3.6e3/7.6e3/7.6e3 (base `ccdist=2.0`: 3.744/199);
  tempo 1.44/1.65/1.81s. Confirma que `ccdist` muda length e curvatura.

### 6.8 Sweep de `expand_dis` + refatoração dos runners (`compare`)
- Refatorados os runners suavizados com helpers genéricos:
  `_run_rrt_star_smooth_kw(config, kw, ...)` + `_smooth_kw(**overrides)`, e
  `_rrt_star_kw(**overrides)` para os não-suavizados. As funções antigas
  (duplicadas) viraram wrappers curtos.
- Novas variantes (steer step, base = 0.3):
  `rrt_star_expand_0_2`, `rrt_star_expand_0_5`,
  `rrt_star_smooth_expand_0_2`, `rrt_star_smooth_expand_0_5`.
- Nomes registrados em `PLANNERS` (`compare_rrt_de2d_nurbs.py`), em
  `PLANNER_STYLE` de `plot_comparison.py` e `analyze_comparison.py`
  (labels "RRT* step 0.2/0.5" e "+ B-spline step ..."), e na lista default de
  `planners` do `plot_comparison.py` (também removida a entrada duplicada
  `rrt_star_smooth_ccdist_3_0`).
- Verificação (`prog03_seed0`, n_jobs=1): as 6 variantes rodam; base
  `rrt_star` len 3.757/cv 199, `expand_0_2` 3.715/1720, `expand_0_5`
  3.744/4898 (steer altera qualidade; tempo ~1.1–2.5s).
- Lembrete: para o `analyze_comparison.py` **exibir** essas variantes, é
  preciso incluí-las em `SELECTED_PLANNERS` (hoje ele filtra por essa lista e
  não contém nem as `ccdist`).

--- 

*Atualizado na sessão de 2026‑10‑16 (extended_start_goal, correções de
curvatura/suavização, medição de tempo sob joblib, redução da densidade de
pontos de colisão/curvatura, correção do parâmetro de passo do RRT* e sweep de
`expand_dis`).*