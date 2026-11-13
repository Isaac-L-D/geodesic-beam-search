import os
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
from ODEClass import ODE

df = pd.read_csv(os.path.join(PROJECT_ROOT, 'geodesicBeamSearch', 'output', 'allUniqueHits.csv'))
coefCols = [c for c in df.columns if c.startswith('dx_') or c.startswith('dy_') or c.startswith('dz_')]
W = df.iloc[0][coefCols].to_numpy(dtype=np.float64).reshape(3, 27)  # any hit works - only the 4D/5D time ratio matters here
W = W / np.linalg.norm(W)
ode = ODE(W=W, doFitToCube=False)

TAU = 500.0
SPAN = 50000.0
initial = np.random.default_rng(1078).uniform(-1, 1, size=3).tolist()


def rhs4(t, s):
    v = ode.eval(t, s[:3])
    speed = np.linalg.norm(v)
    return np.array([v[0], v[1], v[2], speed])


def rhs5(t, s):
    v = ode.eval(t, s[:3])
    speed = np.linalg.norm(v)
    dm = (speed - s[4]) / TAU
    return np.array([v[0], v[1], v[2], speed, dm])


t0 = time.time()
sol4 = solve_ivp(rhs4, (0, SPAN), initial + [0.0], method='LSODA', t_eval=np.linspace(0, SPAN, 5000))
time4 = time.time() - t0

t0 = time.time()
sol5 = solve_ivp(rhs5, (0, SPAN), initial + [0.0, 0.0], method='LSODA', t_eval=np.linspace(0, SPAN, 5000))
time5 = time.time() - t0

print(f"4D (x,y,z,s):   {time4:.3f}s")
print(f"5D (x,y,z,s,m): {time5:.3f}s")
print(f"ratio: {time5/time4:.3f}x")

# Confirm the trajectory itself is unaffected - m only reads velocity, never feeds back.
xyzDiff = np.abs(sol4.y[:3] - sol5.y[:3]).max()
print(f"\nmax |x,y,z difference| between 4D and 5D runs: {xyzDiff:.2e}")
print(f"m ranges from {sol5.y[4].min():.4f} to {sol5.y[4].max():.4f} over the run")
