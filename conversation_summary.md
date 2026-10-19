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
| `pso2d_nurbs` | `run_pso2d_nurbs` | 400 gen × 100 pop = 40 k evals | – | – | – (PSO) | `config.kappa_max` | – | fixed generations |

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

### 6.9 Sweep do grid completo (substitui as variantes *tuned*)
**Objetivo:** em vez de rodar apenas a configuração *recomendada* (tuned) de
cada variante, testar **todas as combinações** do grid
(`ccdist × expand_dis × max_iter`) lado a lado.

**`compare_rrt_de2d_nurbs.py` — alterações:**
- Novas constantes de grid (mesmos valores de `tune_rrt_star.py`):
  ```python
  SWEEP_CCDIST  = [1.0, 2.0, 3.0]
  SWEEP_EXPAND  = [0.1, 0.2, 0.3, 0.4]
  SWEEP_MAXITER = [500, 1000, 2000, 4000]   # 8000 removido para encurtar o grid
  ```
- Novas factories que encapsulam `run_rrt` / `_run_rrt_star_smooth_kw` com um
  parâmetro fixo do grid:
  - `_make_rrt_star_grid_runner(ccdist, expand_dis, max_iter)`
  - `_make_rrt_star_smooth_grid_runner(ccdist, expand_dis, max_iter)`
  - `_grid_name(variant, ccdist, expand_dis, max_iter)` →
    `"{variant}_cc{cc}_ed{ed}_mi{mi}"` (ex.: `rrt_star_cc1.0_ed0.1_mi500`).
- `build_grid_planners()` gera a lista de pares `(nome, runner)` e
  `GRID_PLANNERS` materializa tudo no import.
- Em `PLANNERS`, as duas entradas `rrt_star_tuned` / `rrt_star_smooth_tuned`
  foram substituídas por `*GRID_PLANNERS`.
- **Total:** 3 × 4 × 4 = 48 configs por variante × 2 variantes = **96 planners**
  (as funções `run_rrt_star_tuned` / `run_rrt_star_smooth_tuned` continuam
  definidas, mas não são mais usadas na lista).

**Custo esperado:** no experimento `progressive` (10 cenários, `max_obs=9`) →
**960 tasks**. Rodar com `python compare_rrt_de2d_nurbs.py -j 4` (ou `-j 1`
para tempos justos; ver §6.5).

**Como restringir/expandir:** basta editar `SWEEP_CCDIST`, `SWEEP_EXPAND` e/ou
`SWEEP_MAXITER` (ex.: re-adicionar `8000`).

**Nota:** os nomes novos (`rrt_star_cc..._ed..._mi...`) ainda **não** estão
registrados em `PLANNER_STYLE` de `plot_comparison.py` / `analyze_comparison.py`
nem em `SELECTED_PLANNERS`; para plotar/analisar todas as variantes é preciso
tratá-las lá (ex.: agrupar por prefixo `rrt_star` / `rrt_star_smooth`).

### 6.10 Sweep de parâmetros das variantes Dubins (exceto `bit_star_dubins`)
**Objetivo:** variar os parâmetros que de fato definem cada família Dubins,
substituindo as entradas comentadas em `PLANNERS` por um grid.

**Parâmetros por família (descobertos no código):**
- `RRTStarDubins` (`rrt_planner_modified.py:666`) honra
  `random_yaw_strategy` em `planning` (`:702-708`) → `uniform` vs `toward_goal`.
- `ModifiedDubinsRRTStar` (`:939`) tem `eta1` (Algorithm 9 / `ModifiedSample`) e
  `safety_radius_factor` (Theorem 1). O runner em `compare` passava
  `safety_radius_factor=0.0` (**desliga** o *safety margin*; default da classe = 2.0).
- `RRTStarASV` (`:1191`) tem pesos de custo de giro `cost_a`, `cost_b`.

**`compare_rrt_de2d_nurbs.py` — alterações:**
- Refatoração (sem mudar comportamento default):
  - `_dubins_kw(**overrides)` + `_run_dubins_raw_kw(...)`;
    `run_rrt_star_dubins` virou wrapper.
  - `_run_dubins_smooth_kw(config, kw, seed, n_waypoints, s)`;
    `run_rrt_dubins_smooth` virou wrapper.
  - `run_modified_dubins_rrt_star` e `..._ccpoa` passaram a aceitar `eta1=0.3`
    (antes fixo 0.3) além de `safety_radius_factor`.
- Novas constantes de grid:
  ```python
  SWEEP_DUBINS_CCDIST = [3.5, 4.5, 6.0]
  SWEEP_DUBINS_YAW    = ['uniform', 'toward_goal']
  SWEEP_MDR_ETA1      = [0.1, 0.3, 0.5]
  SWEEP_MDR_SAFETY    = [0.0, 1.0, 2.0]
  SWEEP_ASV_COST_A    = [1.0, 1.2]
  SWEEP_ASV_COST_B    = [1.0, 1.5, 2.0]
  ```
- Factories: `_make_dubins_grid_runner(ccdist, yaw, smooth)`,
  `_make_mdr_grid_runner(eta1, safety, ccpoa)`,
  `_make_asv_grid_runner(cost_a, cost_b)`; agregadas em
  `build_dubins_grid_planners()` / `DUBINS_GRID_PLANNERS`.
- Nomes: `rrt_star_dubins_cc{cc}_yaw{yaw}`, `rrt_dubins_smooth_cc{cc}_yaw{yaw}`,
  `modified_dubins_rrt_star_eta{e}_safe{sf}` (+ sufixo `_ccpoa`),
  `rrt_star_asv_ca{ca}_cb{cb}`.
- Em `PLANNERS`, o bloco Dubins comentado foi substituído por
  `*DUBINS_GRID_PLANNERS`.

**Total:** RRT*Dubins `2×(3×2)=12` + MDR/CCPOA `2×(3×3)=18` + ASV `2×3=6` =
**36 planners Dubins**. Somando aos 96 não-Dubins + `de2d_nurbs` →
**`PLANNERS` com 145 entradas** (verificado importando o módulo).

**Verificação:** `len(GRID_PLANNERS)=96`, `len(DUBINS_GRID_PLANNERS)=36`,
`len(PLANNERS)=145`; nomes gerados conferidos.

**Nota:** mesma ressalva de `PLANNER_STYLE` / `SELECTED_PLANNERS` do §6.9.

### 6.11 Checkpoint periódico + retomada (proteção contra queda de energia)
**Problema:** antes, os resultados só eram gravados por `save_results`, chamado
**após** `run_comparison` terminar. Um desligamento no meio perdia todo o
progresso e obrigava a recomeçar do zero.

**Solução (`compare_rrt_de2d_nurbs.py`):**
- Constantes no topo: `CHECKPOINT_SAVE_EVERY = 20` e `CHECKPOINT_NAME =
  'checkpoint.pkl'`.
- Helpers:
  - `_result_key(res)` → `(scenario, planner)` de um resultado.
  - `_save_checkpoint(output_dir, results)` → grava em `checkpoint.pkl.tmp`,
    `f.flush()` + `os.fsync()`, e depois `os.replace()` (**escrita atômica**,
    então uma queda durante a gravação não corrompe o arquivo).
  - `_load_checkpoint(output_dir)` → carrega se existir (tolerante a arquivo
    corrompido).
- `run_comparison` ganhou os parâmetros `checkpoint=True`,
  `save_every=CHECKPOINT_SAVE_EVERY`, `resume=True`:
  - No início, se `resume`, carrega o checkpoint, monta o conjunto
    `done = {_result_key(...)}` e **filtra** `task_list`, rodando só as tarefas
    que faltam (imprime `resuming: N already done, M remaining`).
  - A cada `save_every` resultados novos, grava o checkpoint (`_flush`).
  - **Paralelo:** passa a usar `Parallel(..., return_as=...)` para receber os
    resultados **conforme terminam** e gravar checkpoints incrementalmente.
    Escolha adaptativa: tenta `'generator_unordered'` (joblib ≥ 1.4), cai para
    `'generator'` (joblib ≥ 1.3; aqui é o 1.3.2) e, se nenhum existir, faz um
    *batch* único. `except (TypeError, ValueError)` cobre a rejeição do parâmetro.
    Se a iteração do gerador for interrompida, o checkpoint é gravado antes de
    propagar a exceção.
  - Checkpoint final garantido com `_flush(force=True)`.
- `main`: parsing manual de `-j`, `--fresh` (ignora/recomeça o checkpoint),
  `--no-checkpoint` e `--save-every N`; repassa tudo a `run_comparison`.
  O `checkpoint.pkl` fica em `comparison_results_paper/<exp>/`.
- **Perda máxima em caso de queda:** as tarefas concluídas mas ainda não
  gravadas — no pior caso `save_every + n_workers` tarefas.

**Verificação:** teste com runners falsos rápidos, `save_every=1`, 2 cenários ×
2 planners: (1) grava 1→4/4 incrementalmente; (2) reexecutar com `-j 2` mostra
`resuming: 4 task(s) already done, 0 remaining` e devolve os 4 registros sem
recalcular.

**Uso:**
```
python compare_rrt_de2d_nurbs.py -j 4                 # retoma automaticamente
python compare_rrt_de2d_nurbs.py -j 4 --save-every 5  # salva mais frequente
python compare_rrt_de2d_nurbs.py --fresh              # recomeça do zero
python compare_rrt_de2d_nurbs.py --no-checkpoint      # desliga checkpoint
```


---

### 6.12 Correção do critério de factibilidade de curvatura + recomputação offline

**Problema encontrado:** no run do paper
(`comparison_results_paper/progressive`) a factibilidade dos planejadores com
curvatura (Dubins/MDR/ASV) saía ~0 %, embora o Dubins respeite a curvatura
(`max_kappa = κ_max`). Causa: o `_numerical_curvature_violation` usava
`np.gradient` com **espaçamento de índice** e mantinha **pontos duplicados**;
caminhos Dubins têm vértices repetidos nas junções reta/arco e amostragem
não-uniforme, gerando picos espúrios de curvatura.

**Comparação de estimadores** (factibilidade de curvatura, ignorando
obstáculos), por variante representativa:

| variante | índice (antigo) | arco (dedupe) | reamostragem | **Menger** |
|---|---|---|---|---|
| `rrt_star` (cru) | 0.00 | 0.18 | 0.00 | 0.02 |
| `rrt_star_dubins` | 0.00 | 0.52 | 0.00 | 0.96 |
| `rrt_star_smooth` | 0.36 | 0.35 | 0.36 | 0.35 |
| `rrt_dubins_smooth` | 0.27 | 0.27 | 0.27 | 0.27 |
| `modified_dubins_rrt_star` | 0.00 | 0.36 | 0.00 | 1.00 |
| `rrt_star_asv` | 0.00 | 0.25 | 0.00 | 1.00 |
| `de2d_nurbs` | 0.85 | 0.84 | 0.05 | 0.44 |

**Decisão (escolhida pelo usuário):** usar o **Menger**
(`max_curvature_numerical`) com **tolerância relativa de 5 %**
(`KAPPA_REL_TOL`). O Menger é exato para arcos e separa corretamente os C1
(Dubins/ASV/MDR ≈ 1.0) do RRT* cru (≈ 0); os 5 % absorvem o erro de
discretização (sem tolerância o DE2D cai para 0.44; com 5 % vai a 0.99).

**Mudanças em `compare_rrt_de2d_nurbs.py`:**
- Novo `curvature_metrics(pts, kappa_max, tol=KAPPA_REL_TOL)` (Menger + tol).
- `posthoc_metrics` e `is_feasible` centralizam a regra pós-hoc (run novo e
  recomputação usam a mesma função).
- `max_curvature_numerical` agora aceita ≥3 pontos.
- `_numerical_curvature_violation` mantido apenas como referência.

**Recomputação offline (sem reexecutar as 14500 tarefas):** novo
`recompute_feasibility.py` reconstrói `cv`/`feasible` a partir de `paths.pkl` +
`obstacles.pkl` + `config.pkl` (todos indexados por `progXX_seedY`). Original
preservado em `summary_pre_cvfix.pkl`. Uso:
```
python recompute_feasibility.py comparison_results_paper/progressive
python recompute_feasibility.py <exp_dir> --dry-run
```
Factibilidade global do run: 3.2 % → **14.1 %**.

**`analyze_comparison.py` (reorganizado nesta sessão):**
- `RESULTS_ROOT` = `comparison_results_paper` (via `--root`).
- `main` reordenado: `ensure_feasible` → `prog_group` → `time_s` → prefixo →
  `select_best_grid_configs` → `filter_planners` → análises → heatmaps.
- `select_best_grid_configs`: escolhe a melhor config de cada variante por
  rank médio com **zona de equivalência Nemenyi** (prefere mais factível, menor
  comprimento, menor curvatura, mais rápida) — alinhado ao `tune_rrt_star.py`.
- `print_block_heatmaps`: heatmaps por bloco (rank, factibilidade, comprimento,
  tempo) em `comparison_results_paper/plots/block_heatmaps.pdf`.
- `bit_star_dubins` removido também do `plot_comparison.py`.

**Tabela final (Menger + 5 %; 100 cenários):**

| Planner | Feas | Tempo (CPU) | Best 10 | Norm. Len | Avg Rank |
|---|---|---|---|---|---|
| DE2D_NURBS (proposto) | 0.99 | 0.78 s | 7/10 | 1.002 | 2.35 |
| RRT* ASV | 1.00 | 20.00 s | 1/10 | 1.033 | 2.62 |
| PSO2D_NURBS (baseline) | 0.69 | 7.29 s | 1/10 | 1.000 | 4.08 |
| RRT* Dubins | 0.44 | 48.12 s | 0/10 | --- | 6.86 |
| RRT* + B-spline | 0.21 | 0.24 s | 0/10 | --- | 7.35 |
| RRT* Dubins + B-spline | 0.17 | 50.76 s | 1/10 | 1.000 | 7.87 |
| Dubins-RRT* | 0.24 | 335.30 s | 0/10 | --- | 8.00 |
| Modified Dubins-RRT* + CCPOA | 0.17 | 301.60 s | 0/10 | --- | 8.29 |
| RRT* | 0.18 | 0.31 s | 0/10 | --- | 8.31 |

Configs selecionadas por variante: `rrt_star` ← `cc2.0_ed0.4_mi1000`;
`rrt_star_smooth` ← `cc1.0_ed0.1_mi1000`; `rrt_star_dubins` ←
`cc6.0_yawuniform`; `rrt_dubins_smooth` ← `cc6.0_yawtoward_goal`;
`modified_dubins_rrt_star` ← `eta0.3_safe0.0`; `..._ccpoa` ← `eta0.1_safe0.0`;
`rrt_star_asv` ← `ca1.0_cb1.5`.

**Orçamento mínimo das variantes varridas (2026-10-19):** por decisão do usuário,
o orçamento de **500 iterações foi excluído** da seleção das variantes varridas
(`_MIN_SWEEP_MAX_ITER = 1000` em `analyze_comparison.select_best_grid_configs`):
só entram configs com `max_iter > 500` (1000/2000/4000). As demais variantes não
têm sufixo `_mi` e ficaram inalteradas.

**Exclusão do `cc = 3.0` (2026-10-19):** por decisão do usuário, o raio de
rewiring **`cc = 3.0` também foi retirado da busca** (`_EXCLUDED_SWEEP_CC = {3.0}`
no mesmo filtro), para deixar RRT* e B-spline em regimes de custo comparáveis e
facilitar a defesa com o revisor. O filtro extrai `_cc([0-9.]+)_ed..._mi...` e
remove `cc = 3.0`; configs sem `_mi` (Dubins `cc 3.5/4.5/6.0`, MDR, ASV) não são
afetadas. Com isso, a busca efetiva vira `cc ∈ {1.0, 2.0}` × `expand_dis` ×
`max_iter ∈ {1000, 2000, 4000}` (24 configs × raw/smooth).
Resultado: `rrt_star` ← `cc2.0_ed0.4_mi1000` (**feas 0.18, 0.31 s**; `cc=1.0` dá
0.00 e `mi` não muda a qualidade nesse `cc`), `rrt_star_smooth` ←
`cc1.0_ed0.1_mi1000` (**feas 0.21, 0.24 s**). Agora o B-spline domina o RRT*
puro em factibilidade (0.21 vs 0.18), tempo (0.24 vs 0.31 s) e rank (7.35 vs
8.31). Emparelhado na mesma base: `cc1.0_ed0.1_mi1000` raw 0.2394 s/feas 0.00 →
smooth 0.2396 s/feas 0.21; `cc2.0_ed0.4_mi1000` raw 0.3075 s/feas 0.18 → smooth
0.3077 s/feas 0.13. Ou seja, o smoothing só ajuda quando a base é serrilhada
(`cc=1.0`) e atrapalha quando a base já é mais reta (`cc=2.0`).

**Correção do desempate na seleção (2026-10-19):** o desempate entre configs
dentro da zona de equivalência (CD Nemenyi) usava `mean_length` antes de
`mean_time`; diferenças de comprimento no nível de ruído (ex.: 3.3062594 vs
3.3063358, ~0.002 %) faziam escolher uma config 60× mais lenta porém idêntica em
qualidade (ex.: `rrt_star_smooth_cc1.0_ed0.1_mi4000` a 3.76 s em vez de `mi500` a
0.06 s). `select_best_grid_configs` agora, dentro do conjunto equivalente,
mantém as de maior factibilidade, trata comprimento/curvatura dentro de 1 %
(`_SELECT_QUAL_TOL`) como empate e escolhe a **mais rápida**.

**Piso de tempo do pós-processamento — removido (2026-10-19):** chegou-se a impor
que `rrt_star_smooth` gastasse ≥ ao tempo médio de `rrt_star`
(`_VARIANT_MIN_TIME_VS`), o que levava o representante de `mi500` (0.06 s) a
`cc1.0_ed0.1_mi1000` (**0.24 s**). Como `search_until_max_iter=True` roda o
orçamento inteiro, `mi500` e `mi1000` produzem resultados **bit-idênticos**
(mesma feas/length/cv nos 100 cenários) e a `mi1000` é **estritamente dominada**
(4× mais lenta, mesmo resultado) — logo não havia justificativa técnica para o
piso. Decisão: **reverter para `mi500`** (0.06 s) e justificar a diferença de
tempo pela **comparação emparelhada na mesma base** (ver abaixo). O código
`_VARIANT_MIN_TIME_VS` foi removido de `select_best_grid_configs`.

**Comparação emparelhada RRT* vs RRT*+B-spline (estado atual):** na mesma base,
os tempos são praticamente idênticos (o smoothing adiciona ~0.2 ms): base
`cc1.0_ed0.1_mi1000` → raw 0.2394 s/feas 0.00 / smooth 0.2396 s/feas 0.21; base
`cc2.0_ed0.4_mi1000` → raw 0.3075 s/feas 0.18 / smooth 0.3077 s/feas 0.13. O
tempo da linha reportada do RRT\* (0.31 s, `cc=2.0`, `ed=0.4`) ser maior que o do
RRT\*+B-spline (0.24 s, `cc=1.0`, `ed=0.1`) deve-se à **base escolhida** (raio de
rewiring e passo), **não** ao custo do smoothing. O smoothing só ajuda a
factibilidade quando a base é serrilhada (`cc=1.0`: 0.00→0.21); com base já mais
reta (`cc=2.0`) ele corta cantos para dentro de obstáculos (0.18→0.13). Com
`cc=3.0` fora da busca, o B-spline domina o RRT\* puro em factibilidade (0.21 vs
0.18), tempo (0.24 vs 0.31 s) e rank (7.35 vs 8.31). Tabela emparelhada em
`response_to_reviewer.md` (B.4) e nota no bullet "Avg Time" (Part C).

**Explicação do `cc` para o revisor (2026-10-19):** em B.2 (definições) foi
detalhado que `connect_circle_dist` é o **raio de vizinhança do rewiring**
(`find_near_nodes`/`choose_parent`/`rewire`): dentro da bola escolhe o melhor pai e
reconecta vizinhos através do novo nó quando isso reduz o custo. Raio maior →
reconexão mais agressiva → o caminho final usa **menos segmentos, mais longos e
melhor alinhados** (menos mudanças de direção), o que reduz a curvatura e por isso
alcança factibilidade em casos onde `cc` pequeno não alcança; em troca, maior
**custo por iteração** (mais vizinhos e mais checagens de colisão) → 0.099 s
(`cc=1.0`) vs 0.308 s (`cc=2.0`) com `ed=0.4, mi=1000` (o `cc=3.0` é agora
excluído da busca). Em B.9 foi acrescentado que
o B-spline usa **15 pontos de controle** (ajuste rígido), por isso corta cantos e
pode **colidir** quando o caminho bruto passa rente a obstáculos; melhoria futura =
**aumentar o número de pontos de controle** para seguir melhor o corredor livre.


**Factibilidade como critério primário + correção do "Best 10" (2026-10-19):**
`select_best_grid_configs` passou a filtrar primeiro as configs com a **maior
taxa de factibilidade** da variante (antes de aplicar CD/comprimento/tempo), de
modo que a config escolhida de `rrt_star` (`cc3.0_ed0.4_mi500`, feas 0.39) e de
`rrt_star_smooth` (`cc1.0_ed0.1_mi500`, feas 0.21) são as de maior factibilidade
de cada grid. Além disso, `print_latex_table` contava vitória de bloco comparando
o comprimento médio apenas das sementes bem-sucedidas — uma variante com
factibilidade 0.21 aparecia "vencendo" 7/10 blocos por só ser factível nas
sementes fáceis (curtas). Agora um planner só vence um bloco se for factível em
**todas** as 10 sementes dele; `Norm. Length` usa o mesmo conjunto elegível
(`--` quando nunca resolveu um bloco inteiro). Efeito (naquela altura, sem o PSO
de 100 pop): DE2D 8/10, RRT\*-ASV 1/10, RRT\*-Dubins+B-spline 1/10, demais 0/10.
Com o PSO de 100 pop o DE2D fica 7/10 e o PSO ganha 1/10.

**Nota (RRT\* vs RRT\*+B-spline):** com `cc=3.0` excluído, o `rrt_star_smooth`
agora tem factibilidade máxima 0.21, acima do `rrt_star` (0.18, em `cc=2.0`). O
smoothing **encurva** os caminhos e pode jogá-los dentro de obstáculos (colisão),
e as duas variantes maximizam em regimes diferentes de `connect_circle_dist`
(1.0 vs 2.0), então não são comparáveis config-a-config. Emparelhadas na mesma
base, `raw_feas` e `smooth_feas` divergem: em `cc=2.0` o raw é melhor (0.18 vs
0.13) e em `cc=1.0` o smooth é melhor (0.00 vs 0.21).

**B-spline: versão anterior vs atual (`response_to_reviewer.md` B.9):** o
RRT\*+B-spline roda o **mesmo** RRT\* e passa o melhor caminho pelo estágio de
smoothing (não gera trajetória diferente). Na versão anterior os parâmetros
eram **fixos** e, quando o B-spline resultava em **colisão** (ou a interpolação
falhava), o código **retornava o caminho original do RRT\*** — nesses casos o
"RRT\*+B-spline" era literalmente a trajetória do RRT\* e sem custo extra de
spline, o que derrubava a média e o desvio do tempo abaixo do RRT\* puro. Na
versão atual: parâmetros varridos/selecionados; a tabela mostra o RRT\*+B-spline
em **0.06 s** vs RRT\* **0.12 s**, mas isso **não** é efeito da velocidade do
smoothing — são regimes de `cc` diferentes (`1.0` vs `3.0`); emparelhado na mesma
base, os tempos batem dentro de ~0.1 ms (ver acima).

**Bug de tempo do DE2D sem obstáculos (2026-10-19):** em `de2d_nurbs.py:272`,
`is_circle` exigia `len(expanded_obs) > 0`; com `prog00` (0 obstáculos) caía no
ramo `detailed_collision_with_polygons`, que mesmo com lista vazia cria um
`LineString` do Shapely **por segmento** (~120/indivíduo × 40 000 indivíduos) →
~10 s/cenário, contra ~0.4–1.1 s com obstáculos. Não afetava qualidade (só
tempo; comprimentos bit-idênticos). Correção: pular a colisão quando
`expanded_obs` é vazio. Re-medidos os 100 cenários DE2D (backup
`summary_pre_de2dtime.pkl`): `prog00` caiu de **9.97 s → 0.375 s**, heatmap agora
monotônico (0.375 → 1.155 s), e a média IQR do DE2D passou de 0.80 → **0.78 s**.
Tabela/gráficos regenerados (`combined_summary.csv`, `plots/`).

**PSO2D-NURBS adicionado ao experimento (2026-10-19):** rodados os 100 cenários
com **400 gerações × 50 pop = 20 000 FEs** (configuração baseline do PSO) e
gravadas as 100 linhas em `summary.pkl`/`summary.csv` + os caminhos em
`paths.pkl` (backups `summary_pre_pso400.pkl` / `paths_pre_pso400.pkl`; total
agora 14 600 linhas). `run_pso2d_nurbs` ganhou `n_generations`/`pop_size`
opcionais (defaults 400/50, sem mudar o comportamento anterior). Resultado:
**feas 0.72**, tempo **3.62 s ± 0.29 s**, Best 10 **0/10**, Norm. Len **---**,
Avg Rank **4.02**. O `ensure_feasible` foi corrigido para **usar o flag
`feasible` gravado** (regra estrita de curvatura) em vez de recalcular só por
`cv`: 6 caminhos do PSO ficam logo acima de `κ_max·1.05` (ex.: 1.4410 vs limite
1.4384), tinham `cv` arredondado para 0 e apareciam indevidamente como factíveis
(0.78 → **0.72**). Para os demais planejadores o flag gravado e o recálculo
coincidiam, então nada mais mudou. `combined_summary.csv` e `plots/` regenerados.

**PSO2D-NURBS com 100 indivíduos (2026-10-19):** por decisão do usuário, o PSO
passou a usar **400 gerações × 100 pop = 40 000 FEs** (mesmo orçamento do DE).
`run_pso2d_nurbs` teve o default `pop_size` alterado de 50 → 100. Os 100 cenários
foram re-rodados (`/tmp/opencode/add_pso100pop.py`, `n_jobs=8`; backups
`summary_pre_pso100pop.pkl`/`paths_pre_pso100pop.pkl`). Resultado: **feas 0.69**,
tempo **7.29 s ± 0.57 s**, Best 10 **1/10**, Norm. Len **1.000**, Avg Rank
**4.08**. Com isso o PSO passa a vencer um bloco (prog01, factível nas 10 seeds) e
o DE2D cai de 8/10 para **7/10** (Norm. Len 1.002, rank 2.35); os ranks dos demais
quase não mudam. Tabela, `combined_summary.csv` e `plots/` regenerados.


---

## 7. Resposta ao revisor (movida)

A resposta ao revisor (em inglês) foi movida para **`response_to_reviewer.md`**,
com uma versão **sintética** (Part A), uma **completa** (Part B, com o
significado de cada variável) e uma **Part C** com a tabela de resultados em
LaTeX e os parâmetros efetivamente usados por cada algoritmo.

---

*Atualizado na sessão de 2026‑10‑16 (extended_start_goal, correções de
curvatura/suavização, medição de tempo sob joblib, redução da densidade de
pontos de colisão/curvatura, correção do parâmetro de passo do RRT*, sweep de
`expand_dis`, sweep do grid completo em substituição às variantes tuned, sweep
de parâmetros das variantes Dubins, checkpoint periódico com retomada, correção
do critério de factibilidade de curvatura (Menger + 5 %) com recomputação
offline, reorganização do `analyze_comparison.py` e resposta ao revisor em
`response_to_reviewer.md`).*
