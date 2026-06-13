import multiprocessing
import math
import random
import time
import numpy as np
import pandas as pd
import pulp
from pathlib import Path

# EQUIPOS

ALL_TEAMS = [
    ("River Plate",-34.5453,-58.4497,"CABA","Monumental"),
    ("Boca Juniors",-34.6353,-58.3648,"CABA","Bombonera"),
    ("Rosario Central",-32.9197,-60.6722,"Santa Fe","Gigante"),
    ("Newell's Old Boys",-32.9536,-60.6781,"Santa Fe","Bielsa"),
    ("Talleres",-31.3680,-64.2411,"Cordoba","Kempes"),
    ("Godoy Cruz",-32.8897,-68.8631,"Mendoza","Malvinas"),
    ("Atlético Tucumán",-26.8014,-65.2228,"Tucuman","Fierro"),
    ("Racing Club",-34.6617,-58.3642,"Buenos Aires","Cilindro"),
    ("Independiente",-34.6656,-58.3617,"Buenos Aires","LDA"),
    ("Banfield",-34.7397,-58.3989,"Buenos Aires","Sola"),
    ("Estudiantes LP",-34.9133,-57.9819,"Buenos Aires","UNO"),
    ("Gimnasia LP",-34.9167,-57.9833,"Buenos Aires","Zerillo"),
    ("San Lorenzo",-34.6464,-58.4381,"CABA","Gasometro"),
    ("Huracan",-34.6447,-58.4064,"CABA","Duco"),
    ("Velez",-34.6439,-58.5267,"CABA","Amalfitani"),
    ("Argentinos",-34.5892,-58.4669,"CABA","Maradona"),
    ("Lanus",-34.7047,-58.3947,"Buenos Aires","Fortaleza"),
    ("Tigre",-34.4493,-58.5422,"Buenos Aires","Dellagiovanna"),
    ("Colon",-31.6272,-60.6953,"Santa Fe","Lopez"),
    ("Union",-31.6353,-60.7150,"Santa Fe","15 Abril"),
]

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)

    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)

    return R*2*math.asin(math.sqrt(a))

def seleccionar_equipos(N, seed):

    random.seed(seed)

    selected = random.sample(ALL_TEAMS, N)

    names = [x[0] for x in selected]

    D = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i!=j:

                D[i,j] = haversine(selected[i][1], selected[i][2], selected[j][1], selected[j][2])

    mu = D[D>0].mean()

    return selected, names, D, mu

def round_robin_schedule(teams):

    lista = list(teams)
    N     = len(lista)

    if N % 2 != 0:
        lista.append('BYE')
        N += 1

    n_fechas = N - 1
    fixture  = []

    for ronda in range(n_fechas):
        partidos = []
        for k in range(N // 2):
            equipo_a = lista[k]
            equipo_b = lista[N - 1 - k]

            if equipo_a == 'BYE' or equipo_b == 'BYE':
                continue

            if ronda % 2 == 0:
                local, visitante = equipo_a, equipo_b
            else:
                local, visitante = equipo_b, equipo_a

            partidos.append((local, visitante))

        fixture.append(partidos)

        lista = [lista[0]] + [lista[-1]] + lista[1:-1]

    return fixture


def evaluar_fixture(fixture, N, D, mu):

    # secuencia[i][t] = ('local'/'visitante', índice_local_o_None)
    secuencia = {i: [] for i in range(N)}

    for fecha in fixture:
        for (local, visitante) in fecha:
            secuencia[local].append(('local', None))
            secuencia[visitante].append(('visitante', local))

    f1 = 0
    for i in range(N):
        seq = secuencia[i]
        for t in range(len(seq) - 1):
            if seq[t][0] == seq[t + 1][0]:
                f1 += 1

    f2_km = 0.0
    for i in range(N):
        seq = secuencia[i]
        for t in range(len(seq) - 1):
            cond_t,  dest_t  = seq[t]
            cond_t1, dest_t1 = seq[t + 1]
            if cond_t == 'visitante' and cond_t1 == 'visitante':
                if dest_t is not None:
                    f2_km += D[i][dest_t]
                if t == len(seq) - 2 and dest_t1 is not None:
                    f2_km += D[i][dest_t1]

    f2_norm = f2_km / mu if mu > 0 else 0.0
    obj     = f1 + f2_norm

    return f1, f2_km, f2_norm, obj


def solve_round_robin(teams, D, mu):

    N = len(teams)
    t0 = time.time()

    fixture = round_robin_schedule(teams)
    f1, f2_km, f2_norm, obj = evaluar_fixture(fixture, N, D, mu)

    return {
        'metodo'           : 'Round Robin',
        'N'                : N,
        'status'           : 'feasible',
        'objetivo'         : obj,
        'f1_breaks'        : float(f1),
        'f2_distancia_km'  : float(f2_km),
        'f2_normalizada'   : float(f2_norm),
        'fixture'          : fixture,
        'tiempo_s'         : time.time() - t0,
        'nodos_explorados' : 0,
        'lp_resueltos'     : 0,
        'cortes_agregados' : 0,
    }

class FixtureBILP:

    def __init__(self, D, use_distances=True):

        self.D = D
        self.use_distances = use_distances
        self.N = D.shape[0]
        self.F = self.N-1
        self.mu = D[D>0].mean()
        self.TEAMS = range(self.N)
        self.DATES = range(self.F)
        self.model = pulp.LpProblem("Fixture", pulp.LpMinimize)
        self._variables()
        self._objective()
        self._constraints()

    # VARIABLES

    def _variables(self):

        T = self.TEAMS
        F = self.DATES
        self.x = pulp.LpVariable.dicts("x", ((i,j,t) for i in T for j in T if i!=j for t in F), cat="Binary")
        self.h = pulp.LpVariable.dicts("h", ((i,t) for i in T for t in F), cat="Binary")
        self.cv = pulp.LpVariable.dicts( "cv", ((i,t) for i in T for t in range(self.F-1)), cat="Binary")
        self.cl = pulp.LpVariable.dicts("cl", ((i,t) for i in T for t in range(self.F-1)), cat="Binary")

        if self.use_distances:

            self.w = pulp.LpVariable.dicts("w", ((i,j,t) for i in T for j in T if i!=j for t in self.DATES ), cat="Binary")
        
        else:

            self.w = None

    # OBJETIVO

    def _objective(self):

        breaks = pulp.lpSum(self.cv[i,t] + self.cl[i,t] for i in self.TEAMS for t in range(self.F-1))

        if not self.use_distances:
            self.model += breaks

        else:
            dist = pulp.lpSum(self.D[i,j] * self.w[i,j,t] for i in self.TEAMS for j in self.TEAMS if i!=j for t in self.DATES)
            self.model += (breaks + dist/self.mu)

    # RESTRICCIONES

    def _constraints(self):

        T = self.TEAMS
        F = self.DATES

        # R1 + R3

        for i in T:
            for t in F:

                away = pulp.lpSum(self.x[j,i,t] for j in T if j!=i)
                home = pulp.lpSum(self.x[i,j,t] for j in T if j!=i)
                self.model += (self.h[i,t] + away == 1)
                self.model += (self.h[i,t] == home)

        # R2

        for i in T:
            for j in T:
                if i<j:

                    self.model += (pulp.lpSum(self.x[i,j,t] + self.x[j,i,t] for t in F) == 1)

        # R4 + R5
 
        for i in T:
            for t in range(self.F-1):

                h1 = self.h[i,t]
                h2 = self.h[i,t+1]

                self.model += self.cl[i,t] >= h1+h2-1
                self.model += self.cl[i,t] <= h1
                self.model += self.cl[i,t] <= h2

                self.model += self.cv[i,t] >= 1-h1-h2
                self.model += self.cv[i,t] <= 1-h1
                self.model += self.cv[i,t] <= 1-h2


        # R6 + R7

        for i in T:
            for t in range(self.F-2):

                s = (self.h[i,t] + self.h[i,t+1] + self.h[i,t+2])
                self.model += s <=2
                self.model += s >=1

        # R8

        self.model += (self.h[0,0]==1)

        # R9

        if self.use_distances:

            for i in T:
                for j in T:
                    if i==j:

                        continue

                    # fecha 0

                    self.model += (self.w[i,j,0] <= self.x[j,i,0])
                    self.model += (self.w[i,j,0] <= self.cv[i,0])
                    self.model += (self.w[i,j,0] >= self.x[j,i,0] + self.cv[i,0] -1)

                    # intermedias

                    for t in range(1,self.F-1):

                        self.model += (self.w[i,j,t] <= self.x[j,i,t])
                        self.model += (self.w[i,j,t] <= self.cv[i,t] + self.cv[i,t-1])
                        self.model += (self.w[i,j,t] >= self.x[j,i,t] + self.cv[i,t] + self.cv[i,t-1] -1)

                    # ultima

                    last = self.F-1
                    self.model += (self.w[i,j,last] <= self.x[j,i,last])
                    self.model += (self.w[i,j,last] <= self.cv[i,last-1])
                    self.model += (self.w[i,j,last] >= self.x[j,i,last] + self.cv[i,last-1] -1)

    def solve(self, time_limit=None, threads=1, verbose=False):

        start = time.perf_counter()
        solver = pulp.PULP_CBC_CMD(msg=verbose, threads=threads, timeLimit=time_limit)

        self.model.solve(solver)
        elapsed = (time.perf_counter() - start)

        return {
            "status": pulp.LpStatus[self.model.status],
            "objetivo": pulp.value(self.model.objective),
            "tiempo_s": elapsed,
            "fixture": self.extract_fixture()
        }

    def extract_fixture(self):

        fixture=[]

        for t in self.DATES:

            matches=[]

            for i in self.TEAMS:
                for j in self.TEAMS:
                    if i==j:

                        continue

                    val = pulp.value(self.x[i,j,t])

                    if val is not None and val>0.5:

                        matches.append((i,j))

            fixture.append(matches)

        return fixture

def calcular_distancia_fixture(fixture, N, D, ):

    secuencia = {i: [] for i in range(N)}

    for fecha in fixture:
        for local, visitante in fecha:

            secuencia[local].append(("L",None))
            secuencia[visitante].append(("V",local))


    distancia = 0

    for equipo in range(N):

        seq = secuencia[equipo]

        for t in range(len(seq)-1):

            tipo1,dest1 = seq[t]
            tipo2,dest2 = seq[t+1]

            if tipo1=="V" and tipo2=="V":

                distancia += D[equipo][dest1]

                if t==len(seq)-2:

                    distancia += D[equipo][dest2]

    return distancia

def run_instance(N, seed, time_limit=300, threads=1):

    teams, names, D, mu = seleccionar_equipos(N, seed)

    results = []

    rr = solve_round_robin(teams=list(range(N)), D=D, mu=mu)

    results.append({
        "metodo": "RoundRobin",
        "N": N,
        "seed": seed,
        "status": rr["status"],
        "objetivo": rr["objetivo"],
        "f1_breaks": rr["f1_breaks"],
        "f2_distancia_km": rr["f2_distancia_km"],
        "f2_norm": rr["f2_normalizada"],
        "tiempo_s": rr["tiempo_s"],
        "n_equipos": N,
        "equipos": ";".join(names)
    })

    model = FixtureBILP(D, use_distances=False)

    bilp = model.solve(time_limit=time_limit, threads=threads, verbose=False)

    f1 = 0

    for i in model.TEAMS:
        for t in range(model.F-1):

            f1 += (model.cv[i,t].value() + model.cl[i,t].value())

    f2_km = calcular_distancia_fixture(bilp["fixture"], N, D)

    results.append({
        "metodo":"Branch&Bounsd(con Breaks)",
        "N":N,
        "seed":seed,
        "status":bilp["status"],
        "objetivo":bilp["objetivo"],
        "f1_breaks":float(f1),
        "f2_distancia_km":float(f2_km),
        "f2_norm":float(f2_km/mu),
        "tiempo_s":bilp["tiempo_s"],
        "n_equipos":N,
        "equipos":";".join(names)
    })

    model = FixtureBILP(D)

    bilp = model.solve(time_limit=time_limit, threads=threads, verbose=False)

    f1 = 0

    for i in model.TEAMS:
        for t in range(model.F-1):

            f1 += (model.cv[i,t].value() + model.cl[i,t].value())

    f2_km = calcular_distancia_fixture(bilp["fixture"], N, D)

    results.append({

        "metodo": "Branch&Bound(con Distancias)",
        "N": N,
        "seed": seed,
        "status": bilp["status"],
        "objetivo": bilp["objetivo"],
        "f1_breaks": float(f1),
        "f2_distancia_km": float(f2_km),
        "f2_norm": float(f2_km/mu),
        "tiempo_s": bilp["tiempo_s"],
        "n_equipos": N,
        "equipos": ";".join(names)
    })

    return results

def benchmark(Ns, seeds, output_csv="benchmark.csv", time_limit=300, threads=1):

    all_results=[]
    total = (len(Ns) * len(seeds))
    count=0

    for N in Ns:
        for seed in seeds:

            count += 1

            print(f"[{count}/{total}]", f"N={N}", f"seed={seed}")

            try:

                out = run_instance(N=N, seed=seed, time_limit=time_limit, threads=threads)
                all_results.extend(out)

            except Exception as e:

                print("ERROR:", N, seed, e)


    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print("\nCSV guardado:", output_csv)

    return df

def imprimir_benchmark(df):

    sep = "="*110

    print("\n"+sep)
    print("RESULTADOS")
    print(sep)
    print(

        f"{'Metodo':<15}"
        f"{'N':>4}"
        f"{'Seed':>6}"
        f"{'Status':<12}"
        f"{'Obj':>10}"
        f"{'f1':>8}"
        f"{'f2_distancia_km':>12}"
        f"{'Tiempo(s)':>12}"
    )

    print("-"*110)

    for _, row in df.iterrows():

        print(
            f"{row['metodo']:<15}"
            f"{row['N']:>4}"
            f"{row['seed']:>6}"
            f"{row['status']:<12}"
            f"{row['objetivo']:>10.3f}"
            f"{row['f1_breaks']:>8.1f}"
            f"{row['f2_distancia_km']:>12.1f}"
            f"{row['tiempo_s']:>12.4f}"
        )

    print(sep)

def imprimir_fixture(fixture, nombres):

    N = len(nombres)

    print("\n"+"="*60)
    print("FIXTURE")
    print("="*60)

    secuencia = {i: [] for i in range(N)}

    for fecha_idx, fecha in enumerate(fixture):

        print(f"\nFecha {fecha_idx+1}")
        print("-"*60)

        for local, visitante in fecha:

            print(f"L: {nombres[local]:<25}"
                  f" vs "
                  f"V: {nombres[visitante]}"
            )

            secuencia[local].append("L")
            secuencia[visitante].append("V")


    print("\n"+"="*60)
    print("SECUENCIA LOCALIA")
    print("="*60)

    for i in range(N):

        sec = " ".join(secuencia[i])

        print(f"{nombres[i]:<25}"
              f"{sec}"
        )

def resumen(df):

    summary = (df.groupby(["metodo","N"]).agg({"objetivo": ["mean","std"],
                                               "tiempo_s": ["mean","std"],
                                               "f1_breaks": "mean",
                                               "f2_norm": "mean"
                                               }).round(5))
    return summary

# EJEMPLO

N = 4
seed = 0

teams, names, D, mu = seleccionar_equipos(N, seed)

result_heuristico = solve_round_robin(list(range(N)), D, mu)

print(result_heuristico["status"])
print(result_heuristico["objetivo"])
print(result_heuristico["tiempo_s"])
imprimir_fixture(result_heuristico["fixture"], names)


model_breaks = FixtureBILP(D, use_distances=False)

result_breaks = model_breaks.solve(time_limit=None, threads=4, verbose=False)

print(result_breaks["status"])
print(result_breaks["objetivo"])
print(result_breaks["tiempo_s"])
imprimir_fixture(result_breaks["fixture"], names)

model_full = FixtureBILP(D, use_distances=True)

result_full = model_full.solve(time_limit=None, threads=4, verbose=False)

print(result_full["status"])
print(result_full["objetivo"])
print(result_full["tiempo_s"])
imprimir_fixture(result_full["fixture"], names)
'''
def _run_wrapper(args):
    N, seed, time_limit, threads = args
    try:
        return run_instance(N=N, seed=seed,
                            time_limit=time_limit, threads=threads)
    except Exception as e:
        print(f"ERROR: N={N} seed={seed} → {e}")
        return []
        
# MAIN EXPERIMENTOS
if __name__ == "__main__":

    Ns         = [4,]# 6, 8]
    seeds      = list(range(1))
    TIME_LIMIT = np.inf
    THREADS    = 4        # hilos CBC cuando se corre en serie
    PARALLEL_DESDE_N = 8  # umbral: a partir de qué N paralelizar

    all_results = []

    for N in Ns:

        if N < PARALLEL_DESDE_N:
            print(f"\nN={N} → serie")
            for seed in seeds:
                print(f"  seed={seed}")
                try:
                    out = run_instance(N=N, seed=seed,
                                       time_limit=TIME_LIMIT,
                                       threads=THREADS)
                    all_results.extend(out)
                except Exception as e:
                    print(f"  ERROR seed={seed}: {e}")

        else:
            n_workers = min(len(seeds), multiprocessing.cpu_count())
            print(f"\nN={N} → paralelo ({n_workers} workers, 1 hilo CBC c/u)")
            tareas = [(N, seed, TIME_LIMIT, 1) for seed in seeds]

            with multiprocessing.Pool(processes=n_workers) as pool:
                lotes = pool.map(_run_wrapper, tareas)

            for lote in lotes:
                if lote:
                    all_results.extend(lote)

    df = pd.DataFrame(all_results)
    df.to_csv("resultados.csv", index=False)
    imprimir_benchmark(df)

    print("\n")
    print("="*70)
    print("PROMEDIOS")
    print("="*70)

    resumen = (df.groupby(["metodo","N"]).agg({"objetivo":"mean",
                                               "tiempo_s":"mean",
                                               "f1_breaks":"mean",
                                               "f2_distancia_km":"mean"
                                               }).round(5))

    print(resumen)
    '''
