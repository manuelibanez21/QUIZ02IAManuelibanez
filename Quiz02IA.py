"""
pso_drones_global_time_scaled.py

- Área: 0..5000 m (x,y)
- 10 drones, radio detección R = 200 m
- Tiempo máximo real = 120 min (7200 s)
- Emulación en tiempo simulado: 120 s (factor de escala S=60)
- PSO optimiza rutas de los drones
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
import csv

# ---------------------------
# Parámetros generales
# ---------------------------
L = 5000.0                     # tamaño del área en metros
Ngrid = 120                    # resolución del mapa
xvec = np.linspace(0, L, Ngrid)
yvec = np.linspace(0, L, Ngrid)
XX, YY = np.meshgrid(xvec, yvec)

# mapa de probabilidades de ejemplo (3 hotspots gaussianos)
prob_map = np.zeros_like(XX)
hotspots = [(1500, 4000, 400), (3500, 1200, 600), (2600, 2600, 400)]
for (x0, y0, s) in hotspots:
    prob_map += np.exp(-((XX - x0)**2 + (YY - y0)**2) / (2 * s**2))
prob_map /= np.sum(prob_map)   # normalizar

# drones y misión
num_drones = 10
R = 200.0                      # radio de detección (m)

# ---------------------------
# Escalado temporal
# ---------------------------
S = 60                         # 1 seg sim = 60 seg reales
Tmax_total = 120.0             # límite en segundos de simulación (120 min reales)
v_real = 10.0                  # velocidad real (m/s)
v_dron = v_real * S            # velocidad simulada equivalente

print(f"[INFO] Escalado temporal aplicado: S={S}")
print(f"[INFO] Tmax_total = {Tmax_total} s de simulación = 7200 s reales")
print(f"[INFO] Velocidad real = {v_real} m/s  ->  Velocidad simulada = {v_dron} m/s")

# ---------------------------
# Configuración de rutas
# ---------------------------
M = 3                          # waypoints por dron
D = num_drones * M * 2         # dimensión del vector de decisión
ds = 100.0                     # paso de muestreo a lo largo de la ruta (m)
constraint_mode = 'global'     # restricción global de tiempo

# ---------------------------
# Funciones auxiliares
# ---------------------------
def route_length(route_xy):
    diffs = np.diff(route_xy, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

def sample_route_points(route_xy, ds_local=ds):
    pts = []
    for i in range(route_xy.shape[0]-1):
        a = route_xy[i]
        b = route_xy[i+1]
        d = np.linalg.norm(b - a)
        if d == 0:
            pts.append(a.copy())
            continue
        n = max(1, int(np.ceil(d / ds_local)))
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(a + t * (b - a))
    pts.append(route_xy[-1].copy())
    return np.array(pts)

def coverage_of_sampled_positions(all_sampled_pts, XX, YY, prob_map, R):
    if all_sampled_pts.size == 0:
        return 0.0
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    samp = all_sampled_pts
    Ncells = grid_pts.shape[0]
    minDistSq = np.full(Ncells, np.inf)
    block_g = 3000
    block_s = 500
    for i in range(0, Ncells, block_g):
        gchunk = grid_pts[i:i+block_g]
        min_d2_chunk = np.full(gchunk.shape[0], np.inf)
        for j in range(0, samp.shape[0], block_s):
            ssub = samp[j:j+block_s]
            dx = gchunk[:,0,None] - ssub[:,0][None,:]
            dy = gchunk[:,1,None] - ssub[:,1][None,:]
            d2 = dx**2 + dy**2
            min_d2_chunk = np.minimum(min_d2_chunk, np.min(d2, axis=1))
        minDistSq[i:i+block_g] = min_d2_chunk
    covered_mask = (minDistSq <= R**2)
    covered_prob = np.sum(prob_map.ravel()[covered_mask])
    return covered_prob

def evaluate_particle(particle):
    particle = particle.reshape(num_drones, M, 2)
    sampled_list = []
    times = np.zeros(num_drones)
    for d in range(num_drones):
        route = particle[d]
        Lroute = route_length(route)
        t = Lroute / v_dron
        times[d] = t
        pts = sample_route_points(route)
        sampled_list.append(pts)
    all_sampled = np.vstack(sampled_list) if len(sampled_list)>0 else np.zeros((0,2))
    covered = coverage_of_sampled_positions(all_sampled, XX, YY, prob_map, R)
    total_time = np.sum(times)
    max_time_per_drone = np.max(times)
    if constraint_mode == 'global':
        excess = max(0.0, total_time - Tmax_total)
    else:
        excess = np.sum(np.maximum(0.0, times - Tmax_total))
    penalty = excess / (1.0 + Tmax_total)
    fitness = covered - penalty
    return max(fitness,0.0), covered, total_time, max_time_per_drone

# ---------------------------
# PSO (idéntico al anterior)
# ---------------------------
np.random.seed(42)
nParticles = 50
maxIter = 150
w = 0.72; c1 = 1.5; c2 = 1.5

lb = np.zeros(D)
ub = np.ones(D) * L
X = np.random.rand(nParticles, D) * (ub - lb) + lb
V = np.zeros((nParticles, D))
pbest = X.copy()
pbest_val = np.zeros(nParticles)
pbest_info = [None]*nParticles

for i in range(nParticles):
    val, cov, ttotal, tmax = evaluate_particle(X[i])
    pbest_val[i] = val
    pbest_info[i] = (cov, ttotal, tmax)
gbest_idx = np.argmax(pbest_val)
gbest = pbest[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]
gbest_info = pbest_info[gbest_idx]

print("Iniciando PSO (con tiempo escalado)...")
t0 = time()
for it in range(maxIter):
    for i in range(nParticles):
        r1 = np.random.rand(D)
        r2 = np.random.rand(D)
        V[i] = w*V[i] + c1*r1*(pbest[i] - X[i]) + c2*r2*(gbest - X[i])
        X[i] = np.clip(X[i] + V[i], lb, ub)
        val, cov, ttotal, tmax = evaluate_particle(X[i])
        if val > pbest_val[i]:
            pbest_val[i] = val
            pbest[i] = X[i].copy()
            pbest_info[i] = (cov, ttotal, tmax)
        if val > gbest_val:
            gbest_val = val
            gbest = X[i].copy()
            gbest_info = (cov, ttotal, tmax)
    if it % 20 == 0 or it == maxIter-1:
        print(f"Iter {it}/{maxIter}  best={gbest_val:.4f}  covered={gbest_info[0]:.4f}  total_time(sim)={gbest_info[1]:.1f}s")

elapsed = time() - t0
print(f"PSO terminado en {elapsed:.1f}s. Mejor fitness={gbest_val:.4f}, covered={gbest_info[0]:.4f}, total_time(sim)={gbest_info[1]:.1f}s")

# ---------------------------
# Visualización
# ---------------------------
best_routes = gbest.reshape(num_drones, M, 2)
plt.figure(figsize=(7,6))
plt.imshow(prob_map, extent=[0,L,0,L], origin='lower', cmap='viridis')
plt.colorbar(label='Probabilidad')
plt.title('Mapa y rutas optimizadas (escala temporal aplicada)')
for d in range(num_drones):
    r = best_routes[d]
    plt.plot(r[:,0], r[:,1], '-w', linewidth=1)
    plt.plot(r[:,0], r[:,1], 'wo', markersize=4, markerfacecolor='w')
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.show()

