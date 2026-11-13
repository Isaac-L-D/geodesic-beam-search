import os
import sys
import time
import itertools
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))
os.chdir(PROJECT_ROOT)

from attractorSimilarity import loadDatasetAttractors
import sharedConstants as ls
from getTrajectory import getTrajectory
from renderAndEmbed import renderAllViews, CAMERA_ANGLES
from momentSimilarity import zernikeFeatures

# Same 36-known-attractor validation methodology as validatePerceptualMethods.py, applied to
# Zernike moments instead - no PyTorch anywhere in this file, so every step (including
# rendering+moments) is fork-parallelize-safe.

known = loadDatasetAttractors()
names = sorted(n for n in known if n not in ls.EXCLUDED_ATTRACTOR_NAMES)
print(f'{len(names)} known attractors')


def _integrateTask(args):
    i, name = args
    tA = getTrajectory(known[name], seed=1000 + i)
    tB = getTrajectory(known[name], seed=2000 + i)
    if tA is None or tB is None:
        return name, None
    return name, (tA[1], tB[1])


t0 = time.time()
trajectories = {}
failed = []
with Pool() as pool:
    tasks = list(enumerate(names))
    for name, result in tqdm(pool.imap(_integrateTask, tasks), total=len(tasks),
                              desc='Integrating known attractors (2 runs each)'):
        if result is None:
            failed.append(name)
        else:
            trajectories[name] = result
print(f'Integrated {len(trajectories)} attractors in {time.time()-t0:.1f}s'
      + (f' - failed: {failed}' if failed else ''))

reliableNames = sorted(trajectories.keys())


# Render + compute both moment types once per trajectory (rendering is the shared cost) - no
# PyTorch, so this runs in the Pool too, unlike the ResNet embedding step in
# validatePerceptualMethods.py.
def _momentsTask(name):
    runA, runB = trajectories[name]
    imagesA = renderAllViews(runA, CAMERA_ANGLES)
    imagesB = renderAllViews(runB, CAMERA_ANGLES)
    zerA = [zernikeFeatures(img) for img in imagesA]
    zerB = [zernikeFeatures(img) for img in imagesB]
    return name, (zerA, zerB)


t0 = time.time()
moments = {}
with Pool() as pool:
    for name, result in tqdm(pool.imap(_momentsTask, reliableNames), total=len(reliableNames),
                              desc='Rendering + computing Zernike moments (2 runs each)'):
        moments[name] = result
print(f'Computed moments for {len(moments)} attractors in {time.time()-t0:.1f}s')


def _dist(featsX, featsY):
    return float(np.mean([np.linalg.norm(featsX[i] - featsY[i]) for i in range(len(featsX))]))


zerSelf = [_dist(moments[n][0], moments[n][1]) for n in reliableNames]

t0 = time.time()
zerCross = []
crossPairs = list(itertools.combinations(reliableNames, 2))
for nameA, nameB in tqdm(crossPairs, desc='Computing cross-distances'):
    zerCross.append(_dist(moments[nameA][0], moments[nameB][0]))
print(f'Cross-distances ({len(zerCross)} pairs) in {time.time()-t0:.1f}s')


def analyze(label, selfD, crossD):
    selfD, crossD = np.array(selfD), np.array(crossD)
    meanSelf, meanCross = selfD.mean(), crossD.mean()
    combined = np.concatenate([selfD, crossD])
    ranks = np.argsort(np.argsort(combined)) + 1
    n1, n2 = len(selfD), len(crossD)
    auc = (ranks[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)
    threshold = meanSelf + 0.25 * (meanCross - meanSelf)
    falsePositive = (selfD > threshold).mean()
    falseNegative = (crossD <= threshold).mean()
    print(f'\n{label}:')
    print(f'  meanSelf={meanSelf:.4f}  meanCross={meanCross:.4f}  ratio={meanCross/meanSelf:.2f}x')
    print(f'  AUC = {auc:.4f}')
    print(f'  at threshold={threshold:.4f}: falsePositive={falsePositive:.1%}, falseNegative={falseNegative:.1%}')


analyze('Hu moments', huSelf, huCross)
analyze('Zernike moments', zerSelf, zerCross)

np.savez(os.path.join(SCRIPT_DIR, 'momentValidationResults.npz'),
         names=reliableNames, huSelf=huSelf, huCross=huCross, zerSelf=zerSelf, zerCross=zerCross)
print('\nSaved to perceptualSimilarity/momentValidationResults.npz')