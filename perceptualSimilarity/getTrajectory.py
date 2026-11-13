import os
import sys
import random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

from ODEClass import ODE
import sharedConstants as ls

# Shared by both perceptual-similarity approaches (render+embed, ICP alignment) - a single real,
# fresh integration per system, using the same EVAL_REFERENCE_NORM scaling convention the rest of
# the pipeline uses (see buildOdeForAnyLabel in makeDiscoveryDashboard.py).


def getTrajectory(W, maxAttempts=6, seed=None):
    if seed is not None:
        random.seed(seed)
    scaleBy = max(1, ls.EVAL_REFERENCE_NORM / np.linalg.norm(W))
    ode = ODE(W=W * scaleBy, doFitToCube=False)
    for _ in range(maxAttempts):
        candidate = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        result = ode.integrate(initial_pos=candidate, endTime=1000.0 / scaleBy)
        if result is not None:
            t, y = result
            return t, y
    return None


def getWByLabel(label, allHitsPath):
    import pandas as pd
    allHits = pd.read_csv(allHitsPath)
    row = allHits[allHits['Label'] == label].iloc[0]
    return ls.invertFlatten(row[ls.W_COLUMNS].to_numpy(dtype=np.float64))