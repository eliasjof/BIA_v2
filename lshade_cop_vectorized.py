import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
import imageio
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import qmc

# def generate_sobol_population(pop_size, dim, xmin_pop, xmax_pop, static_params=None, is_using_nurbs=True, repeated_individuals=1):
#     """
#     Gera uma população usando a sequência de Sobol (amostragem quase-aleatória).
#     """
#     # Verifica se a dimensão é válida (Sobol requer dimensão < 40)
#     if dim > 40:
#         print("Atenção: A dimensão é muito alta para a sequência de Sobol. Usando amostragem uniforme.")
#         return xmin_pop + np.random.rand(pop_size, dim) * (xmax_pop - xmin_pop)
        
#     # 1. Gerar a sequência de Sobol no espaço [0, 1)
#     sampler = qmc.Sobol(d=dim, scramble=True)
#     sobol_points = sampler.random(n=pop_size)
    
#     # 2. Mapear para o espaço de busca [xmin_pop, xmax_pop]
#     # qmc.scale faz o mapeamento linear
#     initial_pop = qmc.scale(sobol_points, xmin_pop, xmax_pop)

#     if static_params is not None:
#         if is_using_nurbs:
#             for pop_i in initial_pop:
#                 weight_normalized = pop_i[static_params['num_free_ctrlpts']*static_params['space_dim']:-1]
#                 # weight_normalized = weight_normalized/np.max(weight_normalized)
#                 pop_i[static_params['num_free_ctrlpts']*static_params['space_dim']:-1] = weight_normalized.tolist()


#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()

#             weights_i  = np.array(static_params['initial_weights'][static_params['num_static_ctrlpts']//2:static_params['num_static_ctrlpts']//2+static_params['num_free_ctrlpts']]).tolist()

#             #np.ones(static_params['num_free_ctrlpts']).reshape(-1).tolist()
#             pop_i = ctrlpts_i + weights_i 
#             pop_i.append(0.0)  # initial deltav   
            
#             for iii in range(repeated_individuals):  
#                 initial_pop[iii] = pop_i
#                 # print(pop_i)
#         # print(np.array(pop_i).reshape(-1,2))
#         else:       
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()        
#             pop_i = ctrlpts_i  
#             pop_i.append(0.0)  # initial deltav   
#             for iii in range(repeated_individuals):  
#                 initial_pop[iii] = pop_i
    
#     return initial_pop
# def generate_sobol_population(pop_size, dim, xmin_pop, xmax_pop, static_params=None, is_using_nurbs=True, repeated_individuals=1, init_closer=False):
#     """
#     Gera uma população inicial com Sobol + variação progressiva sobre a solução base.
#     - 1 dimensão variada, depois 2, ..., até todas as dimensões.
#     """
#     # base Sobol
#     if dim > 40:
#         sobol_points = np.random.rand(pop_size, dim)
#     else:
#         sampler = qmc.Sobol(d=dim, scramble=True)
#         sobol_points = sampler.random(n=pop_size)
    
#     initial_pop = qmc.scale(sobol_points, xmin_pop, xmax_pop)

#     # ------------------------
#     # Montar solução base
#     # ------------------------
#     if static_params is not None:
#         if is_using_nurbs:
#             # vetor "zero" de ctrlpts livres
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             # ctrlpts_i = np.array(static_params['initial_ctrlpts'][
#                 # static_params['num_static_ctrlpts']//2 : static_params['num_static_ctrlpts']//2+static_params['num_free_ctrlpts']
#             # ,:]).reshape(-1).tolist()
#             # print(np.array(static_params['initial_ctrlpts']),ctrlpts_i)
#             # pesos iniciais (fixos da base)
#             weights_i = np.array(static_params['initial_weights'][
#                 static_params['num_static_ctrlpts']//2 : static_params['num_static_ctrlpts']//2+static_params['num_free_ctrlpts']
#             ]).tolist()
#             pop_base = ctrlpts_i + weights_i
#             pop_base.append(static_params['vel_init'])  # deltav inicial
#         else:
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             pop_base = ctrlpts_i
#             pop_base.append(static_params['vel_init'])

#         pop_base = np.array(pop_base)

#         # ------------------------
#         # Inserir base repetida
#         # ------------------------
#         for iii in range(min(repeated_individuals, pop_size)):
#             initial_pop[iii] = pop_base.copy()

#         # ------------------------
#         # Estratégia progressiva
#         # ------------------------
#         # rng = np.random.default_rng()
#         # for k in range(1, min(dim, pop_size - repeated_individuals) + 1):
#         #     ind = pop_base.copy()
#         #     # sorteia k índices únicos
#         #     idxs = rng.choice(dim, size=k, replace=False)
#         #     # aplica variação randômica em cada índice
#         #     variation = rng.uniform(xmin_pop[idxs]/5, xmax_pop[idxs]/5, size=k) * (xmax_pop[idxs] - xmin_pop[idxs])
#         #     ind[idxs] += variation
#         #     # garante limites
#         #     ind = np.clip(ind, xmin_pop, xmax_pop)
#         #     initial_pop[repeated_individuals + k - 1] = ind
#         # rng = np.random.default_rng()
#         # print(f'xmin={xmin_pop[-1]}, xmax={xmax_pop[-1] }')

#         # for k in range(1, min(dim, pop_size - repeated_individuals) + 1):
#         #     ind = pop_base.copy()

#         #     print(f'ind before = {ind}')
#         #     # sorteia k índices únicos
#         #     idxs = rng.choice(dim, size=k, replace=False)

#         #     # calcula margens até os limites
#         #     lower_margin = ind[idxs] - xmin_pop[idxs]
#         #     upper_margin = xmax_pop[idxs] - ind[idxs]
#         #     # lower_margin = xmin_pop[idxs]
#         #     # upper_margin = xmax_pop[idxs]

#         #     # define variação proporcional à menor margem (para não ultrapassar)
#         #     max_step = np.minimum(lower_margin, upper_margin) / 2.0

#         #     # sorteia variação dentro de [-max_step, max_step]
#         #     variation = rng.uniform(-1, 1, size=k) * max_step

#         #     # aplica a variação
#         #     ind[idxs] += variation

#         #     # garante numericamente (raro) que está dentro do intervalo
#         #     # ind[idxs] = np.maximum(np.minimum(ind[idxs], xmax_pop[idxs]), xmin_pop[idxs])
#         #     ind = np.clip(ind, xmin_pop, xmax_pop)
#         #     print(f'ind after = {ind}')

#         #     initial_pop[repeated_individuals + k - 1] = ind
#         rng = np.random.default_rng()

#         # print(f"xmin={xmin_pop[-1]}, xmax={xmax_pop[-1]}")

#         # Cria a população inicial como cópias do indivíduo base
#         initial_pop = np.tile(pop_base, (pop_size, 1))

#         # Gera uma matriz de variações aleatórias no intervalo [-1, 1]
#         random_signs = rng.uniform(-1, 1, size=initial_pop.shape)

#         # Calcula margens até os limites inferior e superior
#         # lower_margin = initial_pop - xmin_pop
#         # upper_margin = xmax_pop - initial_pop
#         lower_margin =  xmin_pop
#         upper_margin = xmax_pop 
#         # Define passo máximo proporcional às margens
#         max_step = np.minimum(lower_margin, upper_margin) / 2.0

#         # Cria máscara para aplicar a variação apenas do índice k até o final
#         mask = np.zeros_like(initial_pop, dtype=bool)
#         mask[:, repeated_individuals:] = True

#         # Aplica variação **somente nas colunas permitidas**
#         initial_pop = np.where(mask, initial_pop + random_signs * max_step, initial_pop)

#         # Garante que todos estão dentro dos limites
#         initial_pop = np.clip(initial_pop, xmin_pop, xmax_pop)

#         # # Aplica variação segura a todos os indivíduos e dimensões
#         # initial_pop += random_signs * max_step

#         # # Garante numericamente que está dentro dos limites (redundante, mas seguro)
#         # initial_pop = np.clip(initial_pop, xmin_pop, xmax_pop)

#         # print(f'initial_pop[-1] = {initial_pop[-30:]}')

#         # print("População inicial gerada com sucesso!")

#     return initial_pop
def generate_sobol_population(
    pop_size,
    dim,
    xmin_pop,
    xmax_pop,
    static_params=None,
    is_using_nurbs=True,
    repeated_individuals=1,
    init_closer=False,
    freeze_dim=0
):
    """
    Gera uma população inicial com amostragem Sobol e variação segura sobre uma solução base.

    Parâmetros
    ----------
    pop_size : int
        Tamanho da população.
    dim : int
        Dimensão do vetor de decisão.
    xmin_pop, xmax_pop : np.ndarray
        Limites inferior e superior (shape = (dim,)).
    static_params : dict, opcional
        Parâmetros estáticos usados para construir a solução base (NURBS, velocidade, etc.).
    is_using_nurbs : bool, padrão=True
        Se True, assume estrutura de parâmetros de curvas NURBS.
    repeated_individuals : int, padrão=1
        Número de indivíduos idênticos à solução base.
    init_closer : bool, padrão=False
        Se True, gera amostras mais próximas da base.
    freeze_dim : int, padrão=0
        Número de dimensões iniciais que não serão modificadas (mantidas fixas).

    Retorna
    -------
    np.ndarray
        População inicial (shape = (pop_size, dim)).
    """

    # -----------------------------
    # 1. Amostragem Sobol
    # -----------------------------
    if dim > 40:
        sobol_points = np.random.rand(pop_size, dim)
    else:
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n=pop_size)

    initial_pop = qmc.scale(sobol_points, xmin_pop, xmax_pop)

    # -----------------------------
    # 2. Construir solução base
    # -----------------------------
    if static_params is not None:
        if is_using_nurbs:
            num_free = static_params['num_free_ctrlpts']
            dim_space = static_params['space_dim']
            mid = static_params['num_static_ctrlpts'] // 2

            # pontos de controle livres (iniciais como zeros)
            ctrlpts_i = np.zeros((num_free, dim_space)).reshape(-1)
            # pesos iniciais
            weights_i = np.array(static_params['initial_weights'][mid:mid + num_free])
            pop_base = np.concatenate([ctrlpts_i, weights_i, [static_params['vel_init']]])
        else:
            num_free = static_params['num_free_ctrlpts']
            dim_space = static_params['space_dim']
            ctrlpts_i = np.zeros((num_free, dim_space)).reshape(-1)
            pop_base = np.concatenate([ctrlpts_i, [static_params['vel_init']]])

    else:
        pop_base = np.zeros(dim)

    # -----------------------------
    # 3. Injetar base repetida
    # -----------------------------
    initial_pop[:repeated_individuals] = pop_base

    # -----------------------------
    # 4. Gerar variação sobre base
    # -----------------------------
    rng = np.random.default_rng()
    varied_pop = np.tile(pop_base, (pop_size, 1))

    # Gera variações uniformes em [-1, 1]
    random_signs = rng.uniform(-1, 1, size=varied_pop.shape)

    # Ajuste de amplitude: se init_closer=True, reduz variação
    variation_scale = 0.25 if init_closer else 0.5

    # Passo máximo proporcional ao tamanho do intervalo
    max_step = (xmax_pop - xmin_pop) * variation_scale

    # Máscara: as primeiras `freeze_dim` colunas não variam
    mask = np.zeros_like(varied_pop, dtype=bool)
    mask[:, freeze_dim:] = True

    # Aplica variação apenas nas dimensões liberadas
    varied_pop = np.where(mask, varied_pop + random_signs * max_step, varied_pop)

    # Garante que está dentro dos limites
    varied_pop = np.clip(varied_pop, xmin_pop, xmax_pop)

    # Sobrescreve os primeiros indivíduos pela base
    varied_pop[:repeated_individuals] = pop_base

    return varied_pop
# def generate_sobol_population(pop_size, dim, xmin_pop, xmax_pop, static_params=None, is_using_nurbs=True, repeated_individuals=1, init_closer=False):
#     """
#     Gera população inicial apenas variando progressivamente em torno da solução base.
#     - começa com alguns indivíduos idênticos à base
#     - depois sorteia 1 dimensão e varia, depois 2, ..., até N
#     """
#     # ------------------------
#     # Montar solução base
#     # ------------------------
#     if static_params is not None:
#         if is_using_nurbs:
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             weights_i = np.array(static_params['initial_weights'][
#                 static_params['num_static_ctrlpts']//2 : static_params['num_static_ctrlpts']//2+static_params['num_free_ctrlpts']
#             ]).tolist()
#             pop_base = ctrlpts_i + weights_i
#             pop_base.append(0.0)  # deltav inicial
#         else:
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             pop_base = ctrlpts_i
#             pop_base.append(0.0)
#     else:
#         pop_base = np.zeros(dim).tolist()

#     # print("gerando nova população: ", xmin_pop[-1], xmax_pop[-1])
#     pop_base = np.array(pop_base)

#     if init_closer:    

#         # ------------------------
#         # Construir população
#         # ------------------------
#         # inicializa todos com a base
#         initial_pop = np.tile(pop_base, (pop_size, 1))
#         rng = np.random.default_rng()

#         # 1) base repetida
#         for i in range(min(repeated_individuals, pop_size)):
#             initial_pop[i] = pop_base.copy()

#         # 2) progressivamente variando mais dimensões
#         filled = repeated_individuals
#         k = 1
#         # while filled < pop_size:
#         #     ind = pop_base.copy()
#         #     # sorteia k índices
#         #     idxs = rng.choice(dim, size=min(k, dim), replace=False)
#         #     variation = rng.uniform(-xmax_pop[idxs]/5, xmax_pop[idxs]/5, size=len(idxs)) * (xmax_pop[idxs] - xmin_pop[idxs])
#         #     ind[idxs] += variation
#         #     # ind[idxs] = rng.uniform(xmin_pop[idxs], xmax_pop[idxs], size=len(idxs))
#         #     ind = np.clip(ind, xmin_pop, xmax_pop)
#         #     initial_pop[filled] = ind
#         #     filled += 1
#         #     k += 1
#         #     if k > dim:  # se já variou todas dimensões, recomeça com k=1
#         #         k += 1
#         while filled < pop_size:
#             ind = pop_base.copy()
#             # sorteia k índices
#             idxs = rng.choice(dim, size=min(k, dim), replace=False)

#             if len(idxs) > 1:
#                 # todos menos o último divididos por 5
#                 variation = np.zeros(len(idxs))
#                 variation[:-1] = rng.uniform(-xmax_pop[idxs[:-1]]/5,
#                                             xmax_pop[idxs[:-1]]/5,
#                                             size=len(idxs)-1) * (xmax_pop[idxs[:-1]] - xmin_pop[idxs[:-1]])
#                 # último sem divisão
#                 variation[-1] = rng.uniform(-xmax_pop[idxs[-1]]/2,
#                                             xmax_pop[idxs[-1]]/2,
#                                             size=1) * (xmax_pop[idxs[-1]] - xmin_pop[idxs[-1]])
#             else:
#                 # caso só tenha 1 índice, não divide por 5
#                 variation = rng.uniform(-xmax_pop[idxs],
#                                         xmax_pop[idxs],
#                                         size=1) * (xmax_pop[idxs] - xmin_pop[idxs])

#             ind[idxs] += variation
#             ind = np.clip(ind, xmin_pop, xmax_pop)

#             initial_pop[filled] = ind
#             filled += 1
#             k += 1
#             if k > dim:  # se já variou todas dimensões, recomeça com k=1
#                 k = 1
#     else:
#         # base Sobol
#         if dim > 40:
#             sobol_points = np.random.rand(pop_size, dim)
#         else:
#             sampler = qmc.Sobol(d=dim, scramble=True)
#             sobol_points = sampler.random(n=pop_size)
        
#         initial_pop = qmc.scale(sobol_points, xmin_pop, xmax_pop)

#         for iii in range(repeated_individuals):
#             initial_pop[iii] = pop_base.copy()
#             # print(pop_i)
    
#             # Estratégia progressiva
#             # ------------------------
#             rng = np.random.default_rng()
#             for k in range(1, min(dim, pop_size - repeated_individuals) + 1):
#                 ind = pop_base.copy()
#                 # sorteia k índices únicos
#                 idxs = rng.choice(dim, size=k, replace=False)
#                 # aplica variação randômica em cada índice
#                 # variation = rng.uniform(-0.2, 0.2, size=k) * (xmax_pop[idxs] - xmin_pop[idxs])
#                 # ind[idxs] += variation
#                 ind[idxs] = rng.uniform(xmin_pop[idxs], xmax_pop[idxs], size=len(idxs))
#                 # garante limites
#                 ind = np.clip(ind, xmin_pop, xmax_pop)
#                 initial_pop[repeated_individuals + k - 1] = ind

#     return initial_pop

# import itertools
# import numpy as np

# import itertools
# import numpy as np

# def generate_sobol_population(pop_size, dim, xmin_pop, xmax_pop,
#                                     static_params=None, is_using_nurbs=True, repeated_individuals=1):
#     """
#     Gera população inicial variando progressivamente em torno da solução base.
#     - Todos os indivíduos da população são preenchidos (sem sobrar base pura).
#     - Progressivo: começa variando 1D, depois 2D, até dimD.
#     - Combinações escolhidas de forma aleatória.
#     """
#     # ------------------------
#     # Montar solução base
#     # ------------------------
#     if static_params is not None:
#         if is_using_nurbs:
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             weights_i = np.array(static_params['initial_weights'][
#                 static_params['num_static_ctrlpts']//2 : static_params['num_static_ctrlpts']//2+static_params['num_free_ctrlpts']
#             ]).tolist()
#             pop_base = ctrlpts_i + weights_i
#             pop_base.append(0.0)  # deltav inicial
#         else:
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             pop_base = ctrlpts_i
#             pop_base.append(0.0)
#     else:
#         pop_base = np.zeros(dim).tolist()

#     pop_base = np.array(pop_base)

#     # ------------------------
#     # Construir população
#     # ------------------------
#     rng = np.random.default_rng()
#     initial_pop = np.zeros((pop_size, dim))

#     filled = 0

#     # Progressivamente variando k dimensões
#     for k in range(1, dim + 1):
#         # todas as combinações possíveis
#         all_combos = list(itertools.combinations(range(dim), k))
#         rng.shuffle(all_combos)  # embaralha as combinações

#         for idxs in all_combos:
#             if filled >= pop_size:
#                 return initial_pop
#             ind = pop_base.copy()
#             idxs = np.array(idxs)
#             # variação global nas dimensões escolhidas
#             ind[idxs] = rng.uniform(xmin_pop[idxs], xmax_pop[idxs], size=len(idxs))
#             initial_pop[filled] = ind
#             filled += 1

#     # Se ainda sobrar espaço (pop_size maior que nº de combinações possíveis),
#     # completa com indivíduos globais totalmente aleatórios
#     while filled < pop_size:
#         ind = rng.uniform(xmin_pop, xmax_pop, size=dim)
#         initial_pop[filled] = ind
#         filled += 1

#     return initial_pop



# def generate_sobol_population(pop_size, dim, xmin_pop, xmax_pop,
#                                     static_params=None, is_using_nurbs=True, repeated_individuals=1):
#     """
#     Gera população inicial variando progressivamente em torno da solução base.
#     Estratégia: primeiro varia 1 dimensão, depois pares, depois trios... até todas.
#     A variação é global: valores sorteados no intervalo [xmin, xmax].
#     """
#     # ------------------------
#     # Montar solução base
#     # ------------------------
#     if static_params is not None:
#         if is_using_nurbs:
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             weights_i = np.array(static_params['initial_weights'][
#                 static_params['num_static_ctrlpts']//2 : static_params['num_static_ctrlpts']//2+static_params['num_free_ctrlpts']
#             ]).tolist()
#             pop_base = ctrlpts_i + weights_i
#             pop_base.append(0.0)  # deltav inicial
#         else:
#             ctrlpts_i = np.zeros((static_params['num_free_ctrlpts'], static_params['space_dim'])).reshape(-1).tolist()
#             pop_base = ctrlpts_i
#             pop_base.append(0.0)
#     else:
#         pop_base = np.zeros(dim).tolist()

#     pop_base = np.array(pop_base)

#     # ------------------------
#     # Construir população
#     # ------------------------
#     initial_pop = np.tile(pop_base, (pop_size, 1))  # todos começam como base
#     rng = np.random.default_rng()

#     filled = repeated_individuals

#     # gera combinações de 1 até dim dimensões
#     for k in range(1, dim + 1):
#         for idxs in itertools.combinations(range(dim), k):
#             if filled >= pop_size:
#                 return initial_pop
#             ind = pop_base.copy()
#             # sorteia novos valores globais para essas dimensões
#             idxs = np.array(idxs)
#             ind[idxs] = rng.uniform(xmin_pop[idxs], xmax_pop[idxs], size=len(idxs))
#             initial_pop[filled] = ind
#             filled += 1

#     return initial_pop


# Create custom colormap: yellow to blue
colors = ["darkblue", "yellow"]
custom_cmap = LinearSegmentedColormap.from_list("yellow_blue_cmap", colors)

class LSHADE_COP:
    def __init__(self, pop_size, dim, max_fes=None, xmin=None, xmax=None, func=None, H=5, tolerance=1e-5, initial_pop=None, type_mean=2, type_mutation=1, type_sharing='best', static_params = None, dimension=2, color='blue', id=0, DEBUG=0, n_generations=None, reset_evolution=True):
        self.pop_size = pop_size
        self.max_pop_size = pop_size
        self.min_pop_size = 4
        self.r_arc = 1.5#2.6*3
        self.dim = dim        
        self.reset_evolution = reset_evolution
        self.n_generations = n_generations
        if n_generations is not None:
            self.max_fes = n_generations * pop_size
        else:
            self.max_fes = max_fes if max_fes is not None else 0

        self._init_pop_size = pop_size
        self.xmin = xmin
        self.xmax = xmax
        self.H = H
        self.tolerance = tolerance
        self.fes = 0
        self.gen = 0
        self.type_sharing = type_sharing
        self.DEBUG = DEBUG
        self.color = color
        self.dimension = dimension
        self.id = id

        self.type_mean = type_mean
        self.type_mutation = type_mutation
        if func is None:
            self.func = self.fitness
        else:
            self.func = func

        if initial_pop is None:
            # Initialize population and archive
            self.P_c = xmin + np.random.rand(pop_size, dim) * (xmax - xmin)
        else:
            self.P_c = initial_pop.copy()

        self.shared_individual = self.P_c[0].copy() # shared individual in cooperative schema
        self.static_params = static_params.copy()

        # self.P_c[0, :] = xmin
        # self.P_c[1, :] = xmax
        # self.P_c[2, :] = (xmin + xmax) / 2
        self.A = np.array([])
        # Atualiza valor f+cv
        # self.fcv_A =  np.array([])

        # Initialize memory
        self.M_CR = np.ones(H) * 0.5
        self.M_F = np.ones(H) * 0.5
        self.k = 0

        self._init_M_CR = self.M_CR.copy()
        self._init_M_F = self.M_F.copy()
        
        # Initialize constraints and mutation settings
        self.Tc = 0.8 * self.max_fes
        self.epsilon_0 = None
        self.epsilon = None
        self.nfeasible = 0
        self.S_F = []
        self.S_CR = []
        self.best_unfesible = []

        # log
        self.log_best_solutions = []
        self.log_best_f = []
        self.log_best_CV = []
        self.log_A = []
        self.log_population = []
        self.log_mean_F = []
        self.log_mean_CR = []
        self.log_mutation_choise = [0,0]

    def set_archive(self, A):
        self.A = A.copy()

    def get_shared_individual(self):
        # if self.type_sharing in 'best':
        self.shared_individual = self.P_c[0].copy()

    def init_population(self, shared_pop=[], update=True) :
        self.shared_pop = shared_pop
        # Evaluate initial population
        self.evaluate_population(shared_pop=shared_pop, update=update)
        self.get_shared_individual()

    def evaluate_population(self, shared_pop=[], update=True):
        # Placeholder for objective and constraint evaluation function
        
        self.f, self.g, self.h = self.func(self.P_c, shared_pop=shared_pop, static_params=self.static_params)

        self.CV = self.get_CV(self.g, self.h)

        # self.f += self.CV
        self.printD('eval = ', self.f.shape, self.g.shape, self.h.shape, self.CV.shape)
        self.fes += self.pop_size
        self.epsilon_0 = 1.0 * np.mean(np.sort(self.CV)[-int(0.2 * self.pop_size):]) / 1  # Adjust for multi-objective case
        
        self.epsilon_0 = 0.0
        self.printD(f'epsilon0 = {self.epsilon_0}')
        self.epsilon = self.epsilon_0
        # print(f'eps = {self.epsilon}')
        # self.printD(f'CV = {self.CV[0]}')
        if update:
            self.P_c, self.P_fe, self.f, self.g, self.h, self.CV, self.best_unfesible = self.sort_pop_CV(
                self.P_c, self.f, self.g, self.h, self.CV, self.epsilon)
            self.nfeasible = len(self.P_fe)

    def get_CV(self, g, h):
        g[g < 1e-18] = 0  # Set small values to zero
        
        # Calculate the constraint violation (CV)
        CV = np.sum(np.maximum(0, g), axis=1) + np.sum(np.maximum(0, np.abs(h) - self.tolerance), axis=1)  # Line sum
        return CV

    # def sort_pop_CV(self, pop, f, g, h, CV, epsilon=0):

    #     # print('sort = ', f.shape, g.shape, h.shape, pop.shape, CV.shape)
    #     #     # Ensure f, g, h, CV are 1D arrays
    #     # if f.ndim > 1:
    #     #     f = f.flatten()  # Flatten f to 1D
    #     # if g.ndim > 1:
    #     #     g = g.flatten()  # Flatten g to 1D (if needed)
    #     # if h.ndim > 1:
    #     #     h = h.flatten()  # Flatten h to 1D (if needed)
    #     # if CV.ndim > 1:
    #     #     CV = CV.flatten()  # Flatten CV to 1D

    #     # Check if all arrays have the same length
    #     # assert len(f) == len(CV) == len(g) == len(h) == pop.shape[0], "Dimension mismatch between arrays"

    #     # Initial index
    #     idx = np.arange(pop.shape[0], dtype=int)  # Ensure idx is of type int

    #     # Masks for CV conditions
    #     mask_CV_zero = (CV == 0)
    #     mask_CV_epsilon = ((CV <= epsilon) & (CV > 0))
    #     mask_CV_positive = (CV > epsilon)

    #     # Sort by f when CV == 0
    #     if np.any(mask_CV_zero):  # Check if there are any feasible solutions
    #         order_zero = np.argsort(f[mask_CV_zero])
    #         idx_feasible = idx[mask_CV_zero][order_zero].copy()
    #     else:
    #         idx_feasible = np.array([], dtype=int)  # No feasible solutions

    #     # Sort by f when 0 < CV <= epsilon
    #     if np.any(mask_CV_epsilon):  # Check if there are any epsilon feasible solutions
    #         order_epsilon = np.argsort(f[mask_CV_epsilon])
    #         idx_feasible_epsilon = idx[mask_CV_epsilon][order_epsilon].copy()
    #     else:
    #         idx_feasible_epsilon = np.array([], dtype=int)  # No epsilon feasible solutions

    #     # Sort by CV when CV > epsilon
    #     if np.any(mask_CV_positive):  # Check if there are any unfeasible solutions
    #         order_positive = np.argsort(CV[mask_CV_positive])
    #         idx_unfeasible = idx[mask_CV_positive][order_positive].copy()
    #     else:
    #         idx_unfeasible = np.array([], dtype=int)  # No unfeasible solutions

    #     # List of the best individuals in the population
    #     pbest_list = np.concatenate([idx_feasible, idx_feasible_epsilon, idx_unfeasible])

    #     if len(idx_unfeasible):
    #         best_unfesible = pop[idx_unfeasible[0],:].copy()
    #     else:
    #         best_unfesible = []

    #     # Get the sorted data
    #     pop_sorted = pop[pbest_list, :].copy() if pbest_list.size > 0 else pop.copy()
    #     f_sorted = f[pbest_list].copy() if pbest_list.size > 0 else f.copy()
    #     g_sorted = g[pbest_list,:].copy() if pbest_list.size > 0 else g.copy()
    #     h_sorted = h[pbest_list,:].copy() if pbest_list.size > 0 else h.copy()
    #     CV_sorted = CV[pbest_list].copy() if pbest_list.size > 0 else CV.copy()

    #     return pop_sorted, pop[idx_feasible], f_sorted, g_sorted, h_sorted, CV_sorted, best_unfesible

    def printD(self, *args, **kwargs):
        if self.DEBUG:
            print(*args, **kwargs)

    

    def sort_pop_CV(self, pop, f, g, h, CV, epsilon=0):

        # 1. Preparação (Robustez e Índices)
        n_pop = pop.shape[0]
        idx = np.arange(n_pop, dtype=int)
        
        # 2. Definição das Máscaras de Factibilidade (usando np.isclose para robustez do float)
        
        # Tolerância estrita (factível ou quase zero, e.g., 1e-9)
        # Nota: A condição CV > 0 é matematicamente redundante se já estamos usando CV <= 0 ou CV <= epsilon
        mask_feasible = (CV <= 1e-5) 
        
        # Epsilon-factível: 1e-9 < CV <= epsilon (Se epsilon > 0)
        # Se epsilon for 0, esta máscara deve ser vazia.
        if epsilon > 1e-9:
            mask_epsilon = (CV > 1e-5) & (CV <= epsilon)
        else:
            mask_epsilon = np.zeros(n_pop, dtype=bool)

        # Não-factível: CV > epsilon
        mask_unfeasible = (CV > epsilon)

        # 3. Ordenação e Captura de Índices

        # A) FACTIVEIS (CV <= 1e-9): Ordenar por F
        if np.any(mask_feasible):
            order_feasible = np.argsort(f[mask_feasible])
            idx_feasible = idx[mask_feasible][order_feasible]
        else:
            idx_feasible = np.array([], dtype=int)

        # B) EPSILON-FACTIVEIS (1e-9 < CV <= epsilon): Ordenar por F
        if np.any(mask_epsilon):
            order_epsilon = np.argsort(f[mask_epsilon])
            idx_epsilon = idx[mask_epsilon][order_epsilon]
        else:
            idx_epsilon = np.array([], dtype=int)

        # C) NÃO-FACTIVEIS (CV > epsilon): Ordenar por CV
        if np.any(mask_unfeasible):
            order_unfeasible = np.argsort(CV[mask_unfeasible]) # Ordenação por CV (menor é melhor)
            idx_unfeasible = idx[mask_unfeasible][order_unfeasible]
        else:
            idx_unfeasible = np.array([], dtype=int)

        # 4. Concatenação Final (A ORDEM DE PRIORIDADE)
        # Prioridade: Feasible > Epsilon > Unfeasible
        pbest_list = np.concatenate([idx_feasible, idx_epsilon, idx_unfeasible])

        # 5. Organização da Saída
        
        # Captura do melhor indivíduo não-factível (para uso em PSO, por exemplo)
        best_unfeasible = pop[idx_unfeasible[0], :].copy() if idx_unfeasible.size > 0 else []

        # Aplica a ordenação a todos os arrays de dados
        pop_sorted = pop[pbest_list, :]
        f_sorted = f[pbest_list]
        g_sorted = g[pbest_list, :]
        h_sorted = h[pbest_list, :]
        CV_sorted = CV[pbest_list]

        # Retorno: pop_sorted, pop_feasible (subconjunto), f_sorted, g_sorted, h_sorted, CV_sorted, best_unfeasible
        return pop_sorted, pop[idx_feasible, :], f_sorted, g_sorted, h_sorted, CV_sorted, best_unfeasible

    # def sort_pop_CV(self, P_c, f, g, h, CV, epsilon):
    #     idx = np.lexsort((f, CV))  # Sort by CV first, then by fitness
    #     P_c = P_c[idx]
    #     f = f[idx]
    #     g = g[idx]
    #     h = h[idx]
    #     CV = CV[idx]
    #     P_fe = P_c[CV <= epsilon]  # Feasible solutions
    #     return P_c, P_fe, f, g, h, CV
    @staticmethod
    def handle_boundaries(trial, xmin, xmax):
        trial = np.where(trial < xmin, 2 * xmin - trial, trial)
        trial = np.where(trial > xmax, 2 * xmax - trial, trial)
        trial = np.clip(trial, xmin, xmax)
        return trial
    def run(self, shared_pop):
        if self.reset_evolution:
            self.reset()
            self.init_population(shared_pop=shared_pop, update=True)
        self.S_CR = 0.5*np.ones(self.pop_size)
        self.S_F  = 0.5*np.ones(self.pop_size)
        if self.n_generations is not None:
            for _ in range(self.n_generations):
                self.evolve(shared_pop)
        else:
            while self.fes <= self.max_fes:
                self.evolve(shared_pop)

    def reset(self):
        self.fes = 0
        self.gen = 0
        self.pop_size = self._init_pop_size
        self.A = np.array([])
        self.M_CR = self._init_M_CR.copy()
        self.M_F = self._init_M_F.copy()
        self.k = 0
        self.epsilon = None
        self.nfeasible = 0
        self.S_F = []
        self.S_CR = []
        self.best_unfesible = []
        self.log_best_solutions = []
        self.log_best_f = []
        self.log_best_CV = []
        self.log_A = []
        self.log_population = []
        self.log_mean_F = []
        self.log_mean_CR = []
        self.log_mutation_choise = [0, 0]
        self.P_c = self.xmin + np.random.rand(self._init_pop_size, self.dim) * (self.xmax - self.xmin)
        self.shared_individual = self.P_c[0].copy()
        self.f = None
        self.g = None
        self.h = None
        self.CV = None

    def evolve(self, shared_pop=[], keep_individual=None):
        self.shared_pop = shared_pop
        if keep_individual is not None:
            self.P_c[1] = keep_individual[0]
            self.f[1] = keep_individual[1]
            self.g[1] = keep_individual[2]
            self.h[1] = keep_individual[3]
            self.CV[1] = keep_individual[4]

        trial = np.copy(self.P_c)
        self.f_old = self.f.copy()
        self.gen += 1
        self.printD(f'fes = {self.fes}, fbest = {self.f[0]}')

        # =========================================================
        # 1. CR / F generation  (vectorized)
        # =========================================================
        ki = np.random.randint(self.H, size=self.pop_size)
        CR = np.where(
            self.M_CR[ki] != -1,
            np.clip(np.random.normal(self.M_CR[ki], 0.1), 0, 1),
            0.0,
        )
        F = np.clip(cauchy.rvs(loc=self.M_F[ki], scale=0.1), 0, 1.1)

        # =========================================================
        # 2. Mutation  (vectorized)
        # =========================================================
        # Fixed p / p_size per generation (as agreed)
        p = np.random.uniform(0.05, 0.95)
        p_size = max(1, min(self.P_c.shape[0],
                   int(np.round(p * (self.nfeasible if self.nfeasible > 0 else self.pop_size)))))

        # p-best indices
        pbest_idx = np.random.randint(p_size, size=self.pop_size)
        ind_p_best = self.P_c[pbest_idx, :]

        # r1 (from current population)
        r1_idx = np.random.randint(self.pop_size, size=self.pop_size)
        r1 = self.P_c[r1_idx, :]

        # r2 (from current + archive)
        if len(self.A):
            archive_combined = np.vstack([self.P_c, self.A])
        else:
            archive_combined = self.P_c.copy()
        r2_idx = np.random.randint(len(archive_combined), size=self.pop_size)
        r2 = archive_combined[r2_idx, :]

        # Precompute common mutation components (broadcast F to column)
        F_col = F[:, np.newaxis]          # shape (pop_size, 1)

        # component used in multiple mutation types: P_c + F * (pbest - P_c) + F * (r1 - r2)
        mut_pbest = self.P_c + F_col * (ind_p_best - self.P_c) + F_col * (r1 - r2)

        if self.type_mutation == 1:
            trial = mut_pbest

        elif self.type_mutation == 2:
            trial = self.P_c + (F_col / 2) * (ind_p_best - self.P_c) + (F_col / 2) * (r1 - r2)

        elif self.type_mutation == 3:
            # mask: use self as base (True) or archive-mean as base (False)
            use_self = (np.random.rand(self.pop_size) > F) | (len(self.A) == 0)
            mut_mean = np.zeros_like(self.P_c)
            if len(self.A) and np.any(~use_self):
                a_idx = np.random.randint(len(self.A), size=np.sum(~use_self))
                mut_mean[~use_self] = (self.A[a_idx, :] + self.P_c[~use_self]) / 2
            mut_mean[use_self] = self.P_c[use_self]
            trial = mut_mean + (F_col / 2) * (ind_p_best - mut_mean) + (F_col / 2) * (r1 - r2)

        elif self.type_mutation == 4:
            has_best_unf = not isinstance(self.best_unfesible, list) and len(self.best_unfesible) > 0
            mask_m4 = (np.random.rand(self.pop_size) > F) & has_best_unf
            self.log_mutation_choise[0] = int(np.sum(mask_m4))
            self.log_mutation_choise[1] = self.pop_size - self.log_mutation_choise[0]

            if np.any(mask_m4):
                trial4a = F_col * (r1 + self.best_unfesible[np.newaxis, :] - r2)
                trial = np.where(mask_m4[:, np.newaxis], trial4a, mut_pbest)
            else:
                trial = mut_pbest

        elif self.type_mutation == 5:
            has_best_unf = not isinstance(self.best_unfesible, list) and len(self.best_unfesible) > 0
            dim_mask = np.random.rand(self.pop_size, self.dim) > 0.5
            mask_m5 = (np.random.rand(self.pop_size) > F) & has_best_unf
            self.log_mutation_choise[0] = int(np.sum(mask_m5))
            self.log_mutation_choise[1] = self.pop_size - self.log_mutation_choise[0]

            trial = self.P_c.copy()
            if np.any(mask_m5):
                trial5a = F_col * (r1 + self.best_unfesible[np.newaxis, :] - r2)
                trial[mask_m5] = np.where(dim_mask[mask_m5], trial5a[mask_m5], self.P_c[mask_m5])
            trial5b = mut_pbest
            trial[~mask_m5] = np.where(dim_mask[~mask_m5], trial5b[~mask_m5], self.P_c[~mask_m5])

        # =========================================================
        # 3. Boundary handling  (vectorized, full population)
        # =========================================================
        trial = self.handle_boundaries(trial, self.xmin, self.xmax)

        # =========================================================
        # 4. Binomial crossover  (vectorized)
        # =========================================================
        j_rand = np.random.randint(self.dim, size=self.pop_size)
        CR_mask = np.random.rand(self.pop_size, self.dim) <= CR[:, np.newaxis]
        CR_mask[np.arange(self.pop_size), j_rand] = True
        trial = np.where(CR_mask, trial, self.P_c)

        # =========================================================
        # 5. Evaluation
        # =========================================================
        f_trial, g_trial, h_trial = self.func(trial, shared_pop, self.static_params)
        self.fes += self.pop_size
        CV_trial = self.get_CV(g_trial, h_trial)

        # =========================================================
        # 6. Selection  (vectorized masks)
        # =========================================================
        P_c_next = np.copy(self.P_c)
        s = np.zeros(self.pop_size, dtype=bool)

        cond1 = (f_trial < self.f) & (CV_trial == 0) & (self.CV == 0)
        s[cond1] = True

        cond2 = (CV_trial < self.CV) & (self.CV > 0)
        s[cond2] = True

        cond3 = (CV_trial == 0) & (self.CV > 0)
        s[cond3] = True

        P_c_next[s] = trial[s]
        self.f[s] = f_trial[s]
        self.g[s, :] = g_trial[s, :]
        self.h[s, :] = h_trial[s, :]
        self.CV[s] = CV_trial[s]
        self.S_F[s] = F[s].copy()
        self.S_CR[s] = CR[s].copy()

        # Update population and archive
        if len(self.A):
            self.A = np.vstack([self.A, self.P_c[s == 1, :]])

            
        else:
            self.A =  self.P_c[s == 1, :].copy()
        self.P_c = P_c_next

        # # Update memory
        # if len(self.S_F) > 0 and len(self.S_CR) > 0:
        #     w = np.abs(self.f - self.f_old) / (np.sum(np.abs(self.f - self.f_old)) + 1e-9)
        #     self.M_CR[self.k] = np.average(self.S_CR, weights=w) if np.any(self.S_CR) else -1
        #     self.M_F[self.k] = np.average(self.S_F, weights=w)
        # self.k = (self.k + 1) % self.H
        # Update memory
        
        if len(self.S_F) > 0 and len(self.S_CR) > 0:
            w = np.abs(self.f[s==1] - self.f_old[s==1]) / (np.sum(np.abs(self.f[s==1] - self.f_old[s==1])) + 1e-9)
            if self.M_CR[self.k-1] == -1 or max(self.S_CR) == 0:
                self.M_CR[self.k-1] = -1
            else:
                # if self.fes > self.Tc:
                #     self.type_mean = 2
                # print(self.pop_size, np.array(self.S_CR).shape, w.shape)
                self.M_CR[self.k-1] = self.mean_Lehmer(np.array(self.S_CR)[s==1], w, norm=self.type_mean)
                # self.M_CR[self.k-1] = np.mean(self.S_CR)


                
            # print(len(w), len(self.S_CR))
            # print(len(self.M_CR))
            # # Check if w is 1D and its length matches S_CR
            # if len(self.S_CR) == len(w):
            self.M_F[self.k-1] = self.mean_Lehmer(np.array(self.S_F)[s==1], w, norm=self.type_mean)
            # self.M_F[self.k-1] =  np.mean(self.S_F)
            
            # Increment k
            self.k += 1
            if self.k > self.H:
                self.k = 1
                
        # else:
        #     self.M_CR[self.k] = -1
        #     self.M_F[self.k] = -1
        
        

        # Update epsilon
        if self.fes < self.Tc:
            self.epsilon = self.epsilon_0 * (1 - self.fes / self.Tc)**5
        else:
            self.epsilon = 0

        # Sort population based on constraints
        self.P_c, self.P_fe, self.f, self.g, self.h, self.CV, self.best_unfesible = self.sort_pop_CV(
            self.P_c, self.f, self.g, self.h, self.CV, self.epsilon)
        self.nfeasible = len(self.P_fe)
        self.printD(f'nfeasible = {self.nfeasible}/{self.pop_size}')
        # self.printD(f'CV = {self.get_CV(np.array([self.g[0]]), np.array([self.h[0]]))}')
        
        # Resizing population and archive size
        if self.fes > 0.0 * self.max_pop_size:
            # Calculate the new population size based on the current fes
            ps = max(
                round(self.min_pop_size + (1 - (self.fes / self.max_fes)) * (self.max_pop_size - self.min_pop_size)), 
                self.min_pop_size
            )
            self.pop_size = ps
            
            # Ensure ps does not exceed the current population size
            if ps > len(self.P_c):
                ps = len(self.P_c)

            # Resize the population and other arrays
            self.P_c = self.P_c[:ps, :].copy()  # Resize the population matrix
            self.S_CR = self.S_CR[:ps]
            self.S_F = self.S_F[:ps]

            # Assuming ps and r_arc are defined earlier in the code
            max_as = round(ps * self.r_arc)  # Calculate max allowable size

            # A is assumed to be a numpy array representing the population
            as_size = self.A.shape[0]  # Get the current size of population A

            # If the current size exceeds the maximum allowable size, reduce it
            if as_size > max_as:
                # Randomly select indices to remove from the population
                idx = np.random.permutation(as_size)
                idx = idx[:as_size - max_as]  # Select the number of indices to remove
                # if 0 in idx: idx.remove(0)
                self.A = np.delete(self.A, idx, axis=0)  # Remove the selected rows from A
            

            # Resize the corresponding fitness and constraint arrays
            self.f = self.f[:ps].copy()  # Resize fitness array
            self.g = self.g[:ps, :].copy()  # Resize constraint array g
            self.h = self.h[:ps, :].copy()  # Resize constraint array h
            self.CV = self.CV[:ps].copy()  # Resize constraint violation array
        # print(np.array(self.P_c[0]) - np.array(self.P_fe[0]))
        self.get_shared_individual()
        self.log_best_solutions.append(self.P_c[0])
        self.log_best_f.append(self.f[0])
        self.log_best_CV.append(self.CV[0])
        self.log_population.append(self.P_c)
        self.log_A.append(self.A)
        self.log_mean_CR.append(np.mean(CR))
        self.log_mean_F.append(np.mean(F))

    def update_fitness(self, shared_pop=[]):
        self.shared_pop = shared_pop
        self.f, self.g, self.h = self.func(self.P_c, self.shared_pop, self.static_params)
        self.fes += self.pop_size
        # print('trial1 = ', f_trial.shape, g_trial.shape, h_trial.shape)    
        self.CV = self.get_CV(self.g, self.h)
        self.P_c, self.P_fe, self.f, self.g, self.h, self.CV, self.best_unfesible = self.sort_pop_CV(
            self.P_c, self.f, self.g, self.h, self.CV, self.epsilon)
        self.nfeasible = len(self.P_fe)

    def mean_Lehmer(self, S, w, norm=2):
        S = np.array(S) if not isinstance(S, np.ndarray) else S

        num = np.sum(w*(S**norm))
        den = np.sum(w*S)
        if den == 0: return 0        
        return num/den
    
    @staticmethod
    def fitness(pop, shared_pop=[], static_params=None, dimension=2):
        f = np.sin(pop[:,0]/2)**2 + pop[:,1]*np.cos(pop[:,1]/2)**4 # Replace with actual function logic

        g = np.zeros((pop.shape[0],1))  # Assuming you have one constraint
        h = np.zeros((pop.shape[0],1))  # Replace with the number of your constraints

        # Add new constraint: sum(pop) > 2
        a = pop[:,0]**2+ (pop[:,1]-2)**2 - 4.1

        g[:,0] = a   # g will be positive if sum(pop) <= 2, and negative if sum(pop) > 2
        # g[:,1] = -2 + pop[:,1]   # g will be positive if sum(pop) <= 2, and negative if sum(pop) > 2
        
        # Return the objective function and constraints
        return f, g, h  # Ensure you return f, g, and h
    