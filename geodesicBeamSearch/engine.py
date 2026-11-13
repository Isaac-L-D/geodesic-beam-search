import os
import sys
import re
import glob
import shutil
import time
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# This module is the search pipeline's machinery: candidate generation, integration and chaos
# classification, the novelty check, checkpointing, and diagnostics. It has no entry point of its
# own - it's imported as a library by a driver module that supplies the seeding strategy and owns
# the round loop (see driver.py).

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))

from ODEClass import ODE
from attractorSimilarityTopN import attractorDistanceBatch, loadDatasetAttractors, descriptorForW
from zernikeIcpCascade import zernikeDistanceBatch, icpAlign, SHORTLIST_K, ICP_RESTARTS
from sharedConstants import (invertFlatten, EVAL_REFERENCE_NORM, MONOMIAL_NAMES, W_COLUMNS,
                                     RELIABLE_MEAN_SELF_DISTANCE, _RELIABLE_MEAN_CROSS_DISTANCE,
                                     NOVELTY_DISTANCE_THRESHOLD, RELIABLE_MEAN_SELF_DISTANCE_CURVATURE,
                                     _RELIABLE_MEAN_CROSS_DISTANCE_CURVATURE,
                                     NOVELTY_DISTANCE_THRESHOLD_CURVATURE, EXCLUDED_ATTRACTOR_NAMES)


def _silenceFortranStdout():
    devnullFd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnullFd, 1)
    os.close(devnullFd)


# computeChaosSignals (like computeLargestLyapunovExponent) can hang without raising on some
# points - runs in a subprocess so a timeout can kill and reclaim it. One call does 3-5 full
# draws (baseline + escalation), not just one, so the budget is much larger than a single-draw
# LLE check would need. Empirically the slowest legitimate (non-stuck) run across 501 real
# round 1/2 search candidates took 82s - 300s gives comfortable margin above that while still
# catching a genuine stall within minutes instead of indefinitely.
CHAOS_CLASSIFIER_TIMEOUT_SECONDS = 300
CHAOS_CLASSIFIER_TIMEOUT_RETRIES = 1
CHAOS_CLASSIFIER_HELPER_PATH = os.path.join(PROJECT_ROOT, "chaosClassifierWorker.py")


# Returns (resultDict, status) - resultDict is None on subprocess failure/timeout, otherwise a
# dict with isChaotic/nDraws/escalated/llePassRate/lleMean/cdMean/cdStd/lleDraws/cdDraws/
# reasonsNotChaotic. On success, status is the worker's own reported status (about the chaos
# computation itself, e.g. "JitCODE Success"). On failure, it instead describes the subprocess
# call breaking down before any real result came back at all (e.g. 'Timed Out', 'Subprocess
# Crashed', 'JSON Parse Error') - nothing to do with the candidate's dynamics.
def classifyChaosWithTimeout(W, scaleBy=1):
    payload = json.dumps({"W": W.flatten().tolist(), "scaleBy": scaleBy})
    for attempt in range(CHAOS_CLASSIFIER_TIMEOUT_RETRIES + 1):
        try:
            result = subprocess.run(
                [sys.executable, CHAOS_CLASSIFIER_HELPER_PATH],
                input=payload,
                capture_output=True,
                text=True,
                timeout=CHAOS_CLASSIFIER_TIMEOUT_SECONDS,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                return None, 'Subprocess Crashed'
            try:
                parsed = json.loads(result.stdout)
                return parsed, parsed.get('status', 'OK')
            except (json.JSONDecodeError, KeyError):
                return None, 'JSON Parse Error'
        except subprocess.TimeoutExpired:
            if attempt < CHAOS_CLASSIFIER_TIMEOUT_RETRIES:
                continue
            return None, 'Timed Out'
        except Exception:
            return None, 'Subprocess Launch Error'  # e.g. failed fork() under heavy load
    return None, 'Both Methods Failed'


INTEGRATE_HELPER_PATH = os.path.join(SCRIPT_DIR, 'integrateWorker.py')
INTEGRATE_TIMEOUT_SECONDS = 150


def integrateWithFailsafe(W, scaleBy, failsafePos):
    payload = json.dumps({"W": W.flatten().tolist(), "scaleBy": scaleBy, "failsafePos": failsafePos})
    try:
        result = subprocess.run(
            [sys.executable, INTEGRATE_HELPER_PATH],
            input=payload, capture_output=True, text=True,
            timeout=INTEGRATE_TIMEOUT_SECONDS, cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            return False, 'Subprocess Crashed', 'N/A', None, False, None
        parsed = json.loads(result.stdout)
        return (parsed['success'], parsed['status'], parsed['minDist'], parsed.get('lastPoint'),
                parsed.get('usedFailsafe', False), parsed.get('initialPos'))
    except subprocess.TimeoutExpired:
        return False, 'Timed Out', 'N/A', None, False, None
    except Exception:
        return False, 'Subprocess Launch Error', 'N/A', None, False, None


# failsafePos: the parent seed's own known-good point (or None) - only used as a last-resort
# initial condition if every random attempt fails, so it never preempts random basin
# exploration, only rescues candidates that would otherwise be falsely rejected.
#
# Chaos classification uses classifyChaosWithTimeout (multi-draw LLE + correlation dimension,
# see chaosClassifierWorker.py) instead of a bare single-draw LLE>0 check - a single draw was
# found unreliable near the chaos threshold (27% of one round's accepted hits flipped LLE sign on
# redraw), and validation against 501 real search hits found the old check was also blind to
# candidates with consistently positive LLE but curve-like (non-fractal) geometry - the dominant
# failure mode, not the redraw-flip one. See chaos-classification-reliability project notes.
def evaluateCandidateWithFailsafe(WFlat, failsafePos):
    np.random.seed()
    W = invertFlatten(np.asarray(WFlat))
    tmp = ODE(W=W, doFitToCube=False)
    scaleBy = max(1, EVAL_REFERENCE_NORM / np.linalg.norm(tmp.W))
    success, status, minDist, lastPoint, usedFailsafe, initialPos = integrateWithFailsafe(
        tmp.W, scaleBy, failsafePos)
    classifyResult, classifyStatus = None, ''
    if success:
        classifyResult, classifyStatus = classifyChaosWithTimeout(tmp.W, scaleBy=scaleBy)
    isChaotic = bool(classifyResult and classifyResult.get('isChaotic'))
    lleMean = classifyResult.get('lleMean') if classifyResult else None
    llePassRate = classifyResult.get('llePassRate') if classifyResult else None
    cdMean = classifyResult.get('cdMean') if classifyResult else None
    cdStd = classifyResult.get('cdStd') if classifyResult else None
    nDraws = classifyResult.get('nDraws') if classifyResult else 0
    escalated = classifyResult.get('escalated') if classifyResult else False
    # Raw per-draw values and failure reasons kept (JSON-encoded for a clean CSV cell) alongside
    # the summary stats, for auditability - what every individual draw actually returned, not
    # just its mean/std.
    lleDraws = json.dumps(classifyResult.get('lleDraws')) if classifyResult else '[]'
    cdDraws = json.dumps(classifyResult.get('cdDraws')) if classifyResult else '[]'
    reasonsNotChaotic = json.dumps(classifyResult.get('reasonsNotChaotic')) if classifyResult else '[]'
    # Positional return (not a namedtuple) to match _evaluateCandidateTask/Pool's plain-list
    # convention elsewhere in this file. Index layout, since callers slice into this directly:
    #   0 success        5 lleMean       10 escalated         15 lastPoint
    #   1 status          6 llePassRate  11 lleDraws           16 initialPos
    #   2 minDist         7 cdMean       12 cdDraws            17+ W (81 values, flattened)
    #   3 classifyStatus  8 cdStd        13 reasonsNotChaotic
    #   4 isChaotic       9 nDraws       14 usedFailsafe
    return ([success, status, minDist, classifyStatus, isChaotic, lleMean, llePassRate, cdMean,
             cdStd, nDraws, escalated, lleDraws, cdDraws, reasonsNotChaotic, usedFailsafe,
             lastPoint, initialPos] + tmp.W.flatten().tolist())


def _evaluateCandidateTask(args):
    WFlat, failsafePos = args
    return evaluateCandidateWithFailsafe(WFlat, failsafePos)


# Establishes a "known good" point for a seed, for use as its own children's failsafe - same
# worker, no failsafePos of its own (a genuinely valid seed should succeed from a random IC
# almost always; the worker's own internal retries already cover anything less certain). Since
# there's no failsafePos to fall back to here, every attempt is a genuine random IC - so this
# point always starts a fresh chain at depth 0 (see runRound's FailsafeChainDepth handling).
def getKnownGoodPoint(WFlat):
    W = invertFlatten(np.asarray(WFlat))
    tmp = ODE(W=W, doFitToCube=False)
    scaleBy = max(1, EVAL_REFERENCE_NORM / np.linalg.norm(tmp.W))
    success, status, minDist, lastPoint, usedFailsafe, initialPos = integrateWithFailsafe(
        tmp.W, scaleBy, None)
    return {'point': lastPoint, 'depth': 0} if success else None


def _knownGoodPointTask(args):
    label, WFlat = args
    return label, getKnownGoodPoint(WFlat)


# jitcode compiles a fresh C extension per candidate into a tempfile.TemporaryDirectory, which
# normally self-cleans on process exit - except our worker subprocesses get SIGKILLed on
# timeout, which gives no chance for that cleanup to run. Over enough candidates (and enough
# timeouts) these accumulate and can exhaust the filesystem's inode limit outright, independent
# of raw disk space. Sweeps stale ones (a currently-running candidate's directory is always
# recently touched, so age is a safe way to tell "leaked" from "in use").
#
# 150s is the subprocess kill timeout for a whole candidate evaluation (all retries included,
# since they run inside one subprocess invocation) - the real bound on how long a directory can
# still be legitimately in progress. 10x that gives comfortable safety margin before treating
# one as leaked rather than in-use.
_CANDIDATE_TIMEOUT_SECONDS = 150


def cleanupStaleJitcodeTempDirs(maxAgeSeconds=_CANDIDATE_TIMEOUT_SECONDS * 10):
    now = time.time()
    for path in glob.glob('/tmp/jitcxde_*'):
        try:
            if now - os.path.getmtime(path) > maxAgeSeconds:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


# For a deliberate interrupt specifically: no age check at all, since we already know this run
# is ending rather than needing to infer "leaked" from staleness.
def cleanupJitcodeTempDirsUnconditional():
    for path in glob.glob('/tmp/jitcxde_*'):
        shutil.rmtree(path, ignore_errors=True)


# A single round can run for hours, so waiting for round boundaries to sweep isn't enough -
# runs independently in the background at a fixed interval instead. Spends nearly all its time
# in time.sleep() (which releases the GIL and holds no CPU), so this costs essentially nothing.
# 5x the candidate timeout: prompt enough to matter without waking up needlessly often.
def _cleanupLoop(intervalSeconds=_CANDIDATE_TIMEOUT_SECONDS * 5):
    while True:
        time.sleep(intervalSeconds)
        cleanupStaleJitcodeTempDirs()


# From perceptualSimilarity/cascadeValidationResults.npz / validateCascade.py (36 reliable
# attractors, catalog-style self + leave-one-out retrieval test: self = true match present in
# the 36-attractor reference catalog, leave-one-out = true match removed, simulating a genuinely
# new discovery). Same threshold formula as D2/curvature in sharedConstants.py. This
# drives the novelty decision now - see runRound.
RELIABLE_MEAN_SELF_DISTANCE_CASCADE = 0.0608
_RELIABLE_MEAN_CROSS_DISTANCE_CASCADE = 0.1469
NOVELTY_DISTANCE_THRESHOLD_CASCADE = (RELIABLE_MEAN_SELF_DISTANCE_CASCADE
    + 0.25 * (_RELIABLE_MEAN_CROSS_DISTANCE_CASCADE - RELIABLE_MEAN_SELF_DISTANCE_CASCADE))

RANDOM_STATE = 42
N_DIRECTIONS_PER_SEED = 40   # simplex size - tune against compute budget once seed count is fixed
RADII = [0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5]
# Below this geodesic distance (radians) to any already-catalogued point, a new candidate is
# considered a re-exploration (backward toward its parent, or sideways into a sibling's
# territory) and discarded before spending any integration/LLE compute on it. Scaled to the
# round's own radius - a coarse round's points are expected to be spaced ~radius apart, so a
# fraction of that is "too close", not an absolute constant.
MIN_CATALOG_DISTANCE_FRACTION = 0.5

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_LOG_PATH = os.path.join(OUTPUT_DIR, 'latticeSamplerResults.txt')
ALL_UNIQUE_HITS_PATH = os.path.join(OUTPUT_DIR, 'allUniqueHits.csv')
TAPER_PLOT_PATH = os.path.join(OUTPUT_DIR, 'newHitsPerRound.png')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# Discovery count grows exponentially round over round (16 -> 16 -> 69 -> 225 -> 754 across the
# first 5 rounds), so saving every discovery's trajectory is not sustainable - round 6 alone
# exhausted disk space mid-save. Capped per round (not across the whole run) to the
# MAX_SAVED_DISCOVERY_TRAJECTORIES most chaotic (highest LLEMean) discoveries that round found,
# not just the first N encountered - see saveDiscoveryTrajectory. Per-round rather than global so
# a single large round can't evict every earlier round's saved examples outright - each round
# keeps its own top slice regardless of how the others compare.
MAX_SAVED_DISCOVERY_TRAJECTORIES = 1000
# Escape hatch back to the original unbounded behavior (save every discovery's trajectory, no
# cap) - e.g. for a deliberately small/bounded run where disk exhaustion isn't a risk. Off by
# default given what happened without the cap.
UNCAP_DISCOVERY_TRAJECTORIES = False


# ode.integrate samples evenly in TIME (fixed dt) at scipy's default tolerance (rtol=1e-3), but
# discovered systems can have extreme local speed bursts - checked against a real saved discovery
# (NewHit_985, round 5): one single dt=0.005 step covered 101.9 units of straight-line distance
# (implied speed ~20,000, vs a ~0.08 median step elsewhere). Re-integrating just that window at
# rtol=1e-8 recovered 125.6 units of real arc length the original had flattened into a straight
# line - the default tolerance let the solver step through it too coarsely to capture the actual
# curve, not just under-sample an otherwise-correct one. Fixing this needs the solver's own dense
# interpolant (built from its adaptive internal steps, so re-querying it afterward is cheap - no
# re-integration), refined recursively wherever the coarse output still shows too large a spatial
# jump, plus a tighter tolerance (see saveDiscoveryTrajectory's integrate call) so the solver's
# internal stepping resolves fast bursts for real instead of stepping straight through them.
def _refineHighSpeedSegments(t, points, denseSol, maxSpatialStep, maxDepth=12):
    newT = [t[0]]
    newPoints = [points[:, 0]]

    def refine(tA, ptA, tB, ptB, depth):
        if np.linalg.norm(ptB - ptA) <= maxSpatialStep or depth >= maxDepth:
            newT.append(tB)
            newPoints.append(ptB)
            return
        tMid = 0.5 * (tA + tB)
        ptMid = denseSol(tMid)
        refine(tA, ptA, tMid, ptMid, depth + 1)
        refine(tMid, ptMid, tB, ptB, depth + 1)

    for i in range(1, len(t)):
        refine(t[i - 1], points[:, i - 1], t[i], points[:, i], 0)
    return np.array(newT), np.array(newPoints).T


# Once _refineHighSpeedSegments has recovered the real curvature in fast stretches, this
# redistributes the (now-complete) point set evenly in ARC LENGTH (distance actually traveled)
# instead of time, so slow stretches - originally oversampled relative to fast ones - don't bloat
# the output file; final point count matches whatever's requested, independent of how much
# refinement happened.
def _resampleByArcLength(t, points, nPoints=None):
    diffs = np.diff(points, axis=1)
    stepLengths = np.sqrt((diffs ** 2).sum(axis=0))
    arcLength = np.concatenate([[0.0], np.cumsum(stepLengths)])
    if arcLength[-1] < 1e-12:
        return t, points
    targetArc = np.linspace(0.0, arcLength[-1], nPoints or len(t))
    resampledT = np.interp(targetArc, arcLength, t)
    resampledPoints = np.array([np.interp(targetArc, arcLength, points[i]) for i in range(3)])
    return resampledT, resampledPoints


def _discoveryTrajectoryPaths(roundDir):
    trajectoryDir = os.path.join(roundDir, 'discoveryTrajectories')
    return trajectoryDir, os.path.join(trajectoryDir, '_index.json')


def _loadDiscoveryTrajectoryIndex(indexPath):
    if not os.path.exists(indexPath):
        return {}
    with open(indexPath) as f:
        return json.load(f)


def _saveDiscoveryTrajectoryIndex(indexPath, index):
    tmpPath = indexPath + '.tmp'
    with open(tmpPath, 'w') as f:
        json.dump(index, f)
    os.replace(tmpPath, indexPath)


# One dense trajectory per confirmed unique discovery, capped per round (round_NNN/
# discoveryTrajectories/) at the MAX_SAVED_DISCOVERY_TRAJECTORIES most chaotic (by LLEMean)
# discoveries that round found - not every evaluated candidate and not just the first N
# encountered. The index (label -> LLEMean for whatever's currently saved) lives on disk next to
# that round's trajectories, so the cap survives process restarts. ChildFailsafePos was already
# confirmed (by the integration precheck) to produce a stable, non-blowing-up trajectory under
# this candidate's own W, so starting a fresh integration from it here can't blow up. Lets figures
# be regenerated later without re-integrating a stochastic (IC-dependent) system.
#
# Split into two functions because the expensive part (integration) is can be made parallel
# across discoveries, but the eviction bookkeeping (which of the current top-N does a new
# discovery displace) is not - two workers deciding independently would race on the same slot.
# _saveOneTrajectoryTask does only the integration/file-write, no shared state, safe to run in any
# Pool worker. saveDiscoveryTrajectoriesBatch resolves the bookkeeping in two cheap sequential
# passes around the parallel integration: pass 1 filters out candidates that clearly can't make
# the cut (so we don't waste integration time on them) without yet committing any real eviction;
# pass 2, run only after the parallel integration reports which candidates actually succeeded,
# resolves the real evictions against the real index. Pass 2 has to come after, not be folded into
# pass 1's optimistic pre-filter: pass 1 assumes every earlier candidate in the chain succeeds, so
# if one actually fails, everything chained after it in that assumption would be wrong(Ex: 
# a failing middle candidate would cause the wrong file to be evicted when this was
# resolved in one pass). Pass 2 only walks the real successes, so no such phantom dependency
# exists; which specific old file gets deleted at an intermediate step can vary with
# imap_unordered's non-deterministic completion order, but the final kept set is exactly the top-N
# by LLEMean regardless of order (a property of this kind of running top-K/min-eviction).
SAVE_TRAJECTORY_HELPER_PATH = os.path.join(SCRIPT_DIR, 'saveTrajectoryWorker.py')
# The tighter tolerance below (rtol=1e-8 vs ODEClass.integrate's scipy-default rtol=1e-3, needed so
# the solver's own adaptive stepping resolves fast bursts instead of stepping through them - see
# _refineHighSpeedSegments) makes this integration noticeably slower and more failure-prone per
# candidate than the eval-time integration (INTEGRATE_TIMEOUT_SECONDS=150 covers that cheaper
# path). Measured live against real round-3-scale discoveries, this path costs ~24s/trajectory
# typically - 180s gives ~7.5x headroom above that for slower outliers while still bounding a
# genuine solver hang (see saveTrajectoryWorker.py's header comment for why a subprocess,
# not an in-process timeout, is required) to a few minutes instead of indefinitely.
TRAJECTORY_SAVE_TIMEOUT_SECONDS = 180


def _saveOneTrajectoryTask(args):
    label, flatW, initialPos, lleMean, roundDir = args
    trajectoryDir, _ = _discoveryTrajectoryPaths(roundDir)
    W = invertFlatten(flatW)
    scaleBy = max(1, EVAL_REFERENCE_NORM / np.linalg.norm(W))
    payload = json.dumps({
        "flatW": np.asarray(flatW).tolist(), "initialPos": initialPos,
        "scaleBy": scaleBy, "endTime": 1000.0 / scaleBy,
        "label": label, "lleMean": lleMean, "trajectoryDir": trajectoryDir,
    })
    try:
        result = subprocess.run(
            [sys.executable, SAVE_TRAJECTORY_HELPER_PATH],
            input=payload, capture_output=True, text=True,
            timeout=TRAJECTORY_SAVE_TIMEOUT_SECONDS, cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            return None
        return label if json.loads(result.stdout).get("success") else None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def saveDiscoveryTrajectoriesBatch(trajectoryTasks, roundDir):
    trajectoryDir, indexPath = _discoveryTrajectoryPaths(roundDir)
    os.makedirs(trajectoryDir, exist_ok=True)
    index = _loadDiscoveryTrajectoryIndex(indexPath)

    # Pass 1: optimistic pre-filter. Discards only candidates that couldn't make the cut even if
    # every candidate ahead of them succeeds - doesn't commit any real eviction yet.
    workingIndex = dict(index)
    toIntegrate = []
    for label, flatW, initialPos, lleMean in trajectoryTasks:
        if not UNCAP_DISCOVERY_TRAJECTORIES and len(workingIndex) >= MAX_SAVED_DISCOVERY_TRAJECTORIES:
            weakestLabel = min(workingIndex, key=workingIndex.get)
            if lleMean <= workingIndex[weakestLabel]:
                continue
            del workingIndex[weakestLabel]
        workingIndex[label] = lleMean
        toIntegrate.append((label, flatW, initialPos, lleMean, roundDir))

    if not toIntegrate:
        return

    # total=len(toIntegrate) but the bar only advances on a real success, so its live count tracks
    # trajectories actually saved, not attempts made - a run with failures visibly lags behind
    # instead of looking identical to a fully successful one. Once every attempt has been consumed
    # from the pool (the for loop below only ends once that's true, success or fail), the bar is
    # forced to 100% so it closes out cleanly rather than sitting short forever, which would look
    # exactly like the "is it frozen" problem tqdm bars exist to prevent. The real success count
    # isn't lost by that - it's logged separately right after.
    results = []
    with Pool(initializer=_silenceFortranStdout) as pool, tqdm(total=len(toIntegrate), desc='Saving discovery trajectories') as bar:
        for label in pool.imap_unordered(_saveOneTrajectoryTask, toIntegrate):
            results.append(label)
            if label is not None:
                bar.update(1)
        bar.n = bar.total
        bar.refresh()

    lleMeanByLabel = {label: lleMean for label, _, _, lleMean, _ in toIntegrate}
    succeeded = [label for label in results if label is not None]
    if len(succeeded) < len(toIntegrate):
        logResult(f'  {len(toIntegrate) - len(succeeded)}/{len(toIntegrate)} discovery '
                  f'trajectory saves failed (integration failure or timeout) and were skipped.')

    # Pass 2: real evictions, resolved only against candidates that actually succeeded.
    for label in succeeded:
        lleMean = lleMeanByLabel[label]
        evictLabel = None
        if not UNCAP_DISCOVERY_TRAJECTORIES and len(index) >= MAX_SAVED_DISCOVERY_TRAJECTORIES:
            evictLabel = min(index, key=index.get)
            if lleMean <= index[evictLabel]:
                continue  # lost out for real, once real outcomes are known
        if evictLabel is not None:
            evictPath = os.path.join(trajectoryDir, f'{evictLabel}.npz')
            if os.path.exists(evictPath):
                os.remove(evictPath)
            del index[evictLabel]
        index[label] = lleMean

    _saveDiscoveryTrajectoryIndex(indexPath, index)


def logResult(line=""):
    print(line)
    with open(RESULTS_LOG_PATH, 'a') as f:
        f.write(line + "\n")


# --- Equal-angle direction generation --------------------------------------------------------

def _simplexVertexDirections(n):
    # Standard coordinate-simplex construction: n unit vectors in R^n, pairwise equiangular,
    # living in the (n-1)-dim subspace {x : sum(x) = 0} (the n standard basis vectors of R^n,
    # recentered on their own centroid).
    vertices = np.eye(n)
    vecs = vertices - vertices.mean(axis=0)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _randomTangentBasis(p, k, rng):
    # k random orthonormal directions, all exactly orthogonal to unit vector p. QR on a random
    # Gaussian matrix (after removing any p-component from each column) gives an orthonormal
    # basis for a random k-dim subspace of the tangent space at p.
    randomMat = rng.normal(size=(len(p), k))
    randomMat -= np.outer(p, p @ randomMat)
    q, _ = np.linalg.qr(randomMat)
    return q  # (81, k), columns orthonormal and orthogonal to p


def equalAngleTangentDirections(p, n, rng):
    # n pairwise-equiangular unit vectors, each exactly tangent to p. Embedding through a
    # matrix with orthonormal columns preserves all pairwise dot products exactly (for any u,v:
    # (Bu)*(Bv) = u^T B^T B v = u^T v when B^T B = I), so equiangularity survives the embedding
    # with zero distortion - unlike drawing directions in the full ambient space and projecting
    # afterward, which does not preserve angles in general.
    simplex = _simplexVertexDirections(n)      # (n, n)
    basis = _randomTangentBasis(p, n, rng)     # (81, n)
    directions = simplex @ basis.T             # (n, 81), already unit length
    return directions


def tangentProject(p, vec):
    return vec - (vec @ p) * p


def geodesicStep(p, direction, radius):
    # Exact great-circle step (not perturb-then-renormalize), so the result is always exactly
    # unit-norm. direction is assumed already tangent to p and unit-length (as produced by
    # equalAngleTangentDirections), but tangentProject/renormalize defensively in case of drift.
    direction = tangentProject(p, direction)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return p
    v = direction / norm
    return np.cos(radius) * p + np.sin(radius) * v


# --- Backward/sideways catalog filter ---------------------------------------------------------

def filterTooCloseToCatalog(candidates, catalogVecs, minDist):
    # candidates: (M, 81) unit vectors. catalogVecs: (K, 81) unit vectors already tested
    # (parent, siblings, ancestors, everything catalogued so far). Returns a boolean mask, True
    # = keep (far enough from everything in the catalog to be worth evaluating).
    if len(catalogVecs) == 0:
        return np.ones(len(candidates), dtype=bool)
    dots = np.clip(candidates @ catalogVecs.T, -1.0, 1.0)
    angles = np.arccos(dots)
    nearestDist = angles.min(axis=1)
    return nearestDist >= minDist


# --- Seed loading --------------------------------------------------------------------------

def loadKnownAttractorSeeds():
    # The hand-curated, well-established attractors: robust by construction, diverse, and
    # already the reference set distinctness is measured against. EXCLUDED_ATTRACTOR_NAMES are
    # left out - see its definition for why.
    known = loadDatasetAttractors()
    names = [n for n in known if n not in EXCLUDED_ATTRACTOR_NAMES]
    vecs = np.array([known[n].flatten() / np.linalg.norm(known[n].flatten()) for n in names])
    return names, vecs


# --- Candidate generation + evaluation for one round ----------------------------------------

def _descriptorTask(task):
    key, W, seed, failsafePos = task
    rng = np.random.default_rng(seed)
    return key, descriptorForW(W, rng, failsafePos=failsafePos)


# Novelty check against the catalog as it stood at the START of the round - populated as
# module-level globals (not via Pool initargs) before Pool() is created, so forked worker
# processes inherit them via copy-on-write instead of needing the whole reference catalog
# shipped through IPC once per hit. Read-only for the lifetime of the Pool below.
_phase1RefNames = None
_phase1RefZernike = None
_phase1RefIcpPoints = None
_phase1RefCurv = None
_phase1RefD2 = None


def _phase1Task(task):
    i, desc = task
    if (desc is None or desc.get('zernike') is None or desc.get('icpPoints') is None
            or desc.get('curvature') is None or desc.get('d2') is None):
        return i, None

    zernikeDists = zernikeDistanceBatch(desc['zernike'], _phase1RefZernike)
    shortlistIdx = np.argsort(zernikeDists)[:SHORTLIST_K]
    bestResidual, bestName = np.inf, None
    for idx in shortlistIdx:
        residual, _, _ = icpAlign(desc['icpPoints'], _phase1RefIcpPoints[idx], nRestarts=ICP_RESTARTS)
        if residual < bestResidual:
            bestResidual, bestName = residual, _phase1RefNames[idx]

    # Curvature/D2 are secondary disagreement flags now (see attractorSimilarityTopN.py's
    # descriptorForW) - cheap enough (attractorDistanceBatch is a vectorized batch call) that
    # computing them here alongside the cascade check costs almost nothing next to the ICP calls
    # above.
    curvDists = attractorDistanceBatch(desc['curvature'], _phase1RefCurv)
    curvNearestIdx = np.argmin(curvDists)
    d2Dists = attractorDistanceBatch(desc['d2'], _phase1RefD2)
    d2NearestIdx = np.argmin(d2Dists)

    return i, {
        'cascadeResidual': float(bestResidual), 'cascadeNearestName': bestName,
        'isDistinctVsCatalog': bestResidual > NOVELTY_DISTANCE_THRESHOLD_CASCADE,
        'curvNearestName': _phase1RefNames[curvNearestIdx],
        'curvNearestDist': float(curvDists[curvNearestIdx]),
        'curvIsDistinct': curvDists[curvNearestIdx] > NOVELTY_DISTANCE_THRESHOLD_CURVATURE,
        'd2NearestName': _phase1RefNames[d2NearestIdx],
        'd2NearestDist': float(d2Dists[d2NearestIdx]),
        'd2IsDistinct': d2Dists[d2NearestIdx] > NOVELTY_DISTANCE_THRESHOLD,
    }


def buildRoundTasks(seedLabels, seedVecs, radius, nDirections, catalogVecs, rng, explored):
    # A fresh randomly-oriented direction set per seed (not shared across seeds), each filtered
    # against the full catalog before being returned, so evaluation compute is never spent on
    # backward/sideways repeats. Every direction tried (kept or not) is appended to `explored`
    # for later record-keeping - this is pure logging, it doesn't feed back into anything here.
    tasks = []
    for label, p in zip(seedLabels, seedVecs):
        directions = equalAngleTangentDirections(p, nDirections, rng)
        for d in directions:
            explored.append((label, radius, d))
        candidates = np.array([geodesicStep(p, d, radius) for d in directions])
        keep = filterTooCloseToCatalog(candidates, catalogVecs, radius * MIN_CATALOG_DISTANCE_FRACTION)
        for cand in candidates[keep]:
            tasks.append((label, cand))
    return tasks


def saveExploredDirections(roundDir, explored):
    # Pure record-keeping - see this file's direction-generation functions above for how these
    # get used; this file has no effect on any run's behavior, only what was tried and where.
    labels = np.array([e[0] for e in explored])
    radii = np.array([e[1] for e in explored], dtype=np.float64)
    directions = np.array([e[2] for e in explored], dtype=np.float64)
    np.savez(os.path.join(roundDir, 'exploredDirections.npz'),
             seedLabel=labels, radius=radii, direction=directions)


SUCCESSFUL_INITIAL_POSITION_COLUMNS = ['ParentLabel', 'InitialPos', 'UsedFailsafe', 'LastPoint',
                                        'LLEMean', 'LLEPassRate', 'CorrDimMean', 'CorrDimStd',
                                        'NDraws', 'Escalated', 'LLEDraws', 'CorrDimDraws',
                                        'ReasonsNotChaotic', 'ClassifyStatus', 'IsChaotic',
                                        'FailsafeChainDepth']


def saveSuccessfulInitialPositions(roundDir, rows):
    # One row per candidate that succeeded (chaotic or not) - which initial position worked,
    # whether it took the failsafe to get there, and where it settled. Meant to be used
    # alongside exploredDirections.npz (which covers every direction tried, not just the ones
    # that led somewhere). An explicit column list on the empty-rows path keeps the file readable
    # by pd.read_csv even on a round with zero successful candidates - an empty DataFrame with no
    # columns specified writes a blank file with no header at all.
    df = pd.DataFrame(rows, columns=SUCCESSFUL_INITIAL_POSITION_COLUMNS) if not rows else pd.DataFrame(rows)
    df.to_csv(os.path.join(roundDir, 'successfulInitialPositions.csv'), index=False)


def runRound(roundNum, seedLabels, seedVecs, catalogVecs, referenceDescriptors, newHitCount,
             nextSeed, rng, knownFailsafePoints):
    roundDir = os.path.join(OUTPUT_DIR, f'round_{roundNum:03d}')
    os.makedirs(roundDir, exist_ok=True)
    uniqueHitsPath = os.path.join(roundDir, 'latticeSamplerUniqueHits.csv')

    logResult(f"\n{'=' * 20} ROUND {roundNum} {'=' * 20}")
    logResult(f"{len(seedLabels)} seeds this round, {N_DIRECTIONS_PER_SEED} directions each, "
              f"{len(RADII)} radii")

    # One known-good point per seed, supplied to its own children as their failsafe initial
    # condition. Reused from knownFailsafePoints where already known; computed fresh only for
    # seeds with nothing cached yet (round 1's known-attractor seeds).
    knownGoodByLabel = dict(knownFailsafePoints)
    labelsNeedingPoint = [(label, vec) for label, vec in zip(seedLabels, seedVecs)
                           if label not in knownGoodByLabel]
    if labelsNeedingPoint:
        with Pool(initializer=_silenceFortranStdout) as pool:
            computed = dict(tqdm(
                pool.imap_unordered(_knownGoodPointTask,
                                     [(label, vec.tolist()) for label, vec in labelsNeedingPoint]),
                total=len(labelsNeedingPoint), desc=f'Round {roundNum}: seed failsafe points'))
        knownGoodByLabel.update(computed)

    allTasks = []
    explored = []
    for radius in RADII:
        allTasks += buildRoundTasks(seedLabels, seedVecs, radius, N_DIRECTIONS_PER_SEED,
                                     catalogVecs, rng, explored)
    saveExploredDirections(roundDir, explored)
    maxBeforeFilter = len(seedLabels) * N_DIRECTIONS_PER_SEED * len(RADII)
    catalogFilteredCount = maxBeforeFilter - len(allTasks)
    logResult(f"{len(allTasks)} candidates after backward/sideways catalog filtering - "
              f"{catalogFilteredCount} of {maxBeforeFilter} thrown out for being too close to "
              f"an already-catalogued point ({catalogFilteredCount / maxBeforeFilter * 100:.1f}%)")

    if not allTasks:
        logResult(f"\nRound {roundNum}: nothing left to evaluate after filtering. Stopping.")
        return np.empty((0,)), np.empty((0, 81)), newHitCount, nextSeed, {}

    evalTasks = [(cand.tolist(), (knownGoodByLabel.get(label) or {}).get('point'))
                 for label, cand in allTasks]
    with Pool(initializer=_silenceFortranStdout) as pool:
        results = list(tqdm(pool.imap(_evaluateCandidateTask, evalTasks),
                             total=len(evalTasks), desc=f'Round {roundNum}: evaluating candidates'))

    # r[0]=success, r[14]=usedFailsafe - see the index layout comment on
    # evaluateCandidateWithFailsafe's return statement above.
    precheckFailCount = sum(1 for r in results if not r[0])
    precheckSuccessCount = len(results) - precheckFailCount
    failsafeRescueCount = sum(1 for r in results if r[0] and r[14])
    logResult(f"\n{precheckFailCount} of {len(results)} candidates thrown out by the "
              f"blowup/stability precheck ({precheckFailCount / len(results) * 100:.1f}%)")
    if precheckSuccessCount:
        logResult(f"{failsafeRescueCount} of {precheckSuccessCount} successful candidates only "
                  f"succeeded via the parent-position failsafe on the last retry "
                  f"({failsafeRescueCount / precheckSuccessCount * 100:.1f}%)")

    hitRows = []
    successRows = []
    for (parentLabel, wVec), res in zip(allTasks, results):
        (success, _, _, classifyStatus, isChaotic, lleMean, llePassRate, cdMean, cdStd, nDraws,
         escalated, lleDraws, cdDraws, reasonsNotChaotic, usedFailsafe, lastPoint,
         initialPos) = res[:17]
        if not success:
            continue
        # How many consecutive generations the point actually used as the initial condition has
        # gone without a fresh, independent (genuine-random-IC) validation. 0 = validated just
        # now, whether by a real random draw or (for a seed's own point) getKnownGoodPoint, which
        # never takes the failsafe branch itself. Reusing the failsafe adds one more generation on
        # top of however deep the parent's own point already was.
        if usedFailsafe:
            parentDepth = (knownGoodByLabel.get(parentLabel) or {}).get('depth', 0)
            resultDepth = parentDepth + 1
        else:
            resultDepth = 0
        successRows.append({'ParentLabel': parentLabel, 'InitialPos': initialPos,
                             'UsedFailsafe': usedFailsafe, 'LastPoint': lastPoint,
                             'LLEMean': lleMean, 'LLEPassRate': llePassRate,
                             'CorrDimMean': cdMean, 'CorrDimStd': cdStd, 'NDraws': nDraws,
                             'Escalated': escalated, 'LLEDraws': lleDraws, 'CorrDimDraws': cdDraws,
                             'ReasonsNotChaotic': reasonsNotChaotic, 'ClassifyStatus': classifyStatus,
                             'IsChaotic': isChaotic, 'FailsafeChainDepth': resultDepth})
        if isChaotic:
            # Propagate this hit's own settled point only if it was independently validated by a
            # genuine random IC. If it only converged via its own failsafe, initialPos here IS the
            # parent's point (never actually left it) - so propagate that unchanged instead. Despite
            # the name, initialPos is NOT "this candidate's own starting guess" in that branch - see
            # integrateWorker.py's own note on the same variable.
            childFailsafePos = initialPos if usedFailsafe else lastPoint
            hitRows.append({'ParentLabel': parentLabel, 'W': res[17:], 'UsedFailsafe': usedFailsafe,
                             'ChildFailsafePos': childFailsafePos, 'ChildFailsafeDepth': resultDepth,
                             'LLEMean': lleMean, 'LLEPassRate': llePassRate,
                             'CorrDimMean': cdMean, 'CorrDimStd': cdStd, 'NDraws': nDraws,
                             'Escalated': escalated, 'LLEDraws': lleDraws, 'CorrDimDraws': cdDraws})

    saveSuccessfulInitialPositions(roundDir, successRows)

    logResult(f"{len(hitRows)} chaotic hits found this round.")
    if not hitRows:
        return np.empty((0,)), np.empty((0, 81)), newHitCount, nextSeed, {}

    hitWs = [invertFlatten(np.array(row['W'], dtype=np.float64)) for row in hitRows]
    # ChildFailsafePos was already confirmed (by the integration precheck) to produce a stable
    # trajectory under this hit's own W, so it's a safe last-resort IC for any descriptor replica
    # whose own random draw doesn't converge - see arcLengthTrialWorkerJitcode.py's main().
    descTasks = [(('hit', i), W, nextSeed + i, hitRows[i]['ChildFailsafePos'])
                 for i, W in enumerate(hitWs)]
    nextSeed += len(descTasks)
    descriptorsByKey = {}
    # maxtasksperchild=1: descriptorForW's render step measured ~350MB peak RSS per call (down
    # from ~1.7GB before switching renderSilhouette.py to markers-only - see that file's comment),
    # comfortably safe per-task even at one worker per core. But a round can have far more tasks
    # than a worker's own core count, and matplotlib/numpy don't necessarily return every byte to
    # the OS between tasks even after the figure is closed - forcing a fresh process per task
    # guarantees that retained memory can never accumulate across a long-lived worker's many
    # tasks, regardless of how large a round gets.
    with Pool(initializer=_silenceFortranStdout, maxtasksperchild=1) as pool:
        for key, desc in tqdm(pool.imap_unordered(_descriptorTask, descTasks), total=len(descTasks),
                               desc=f'Round {roundNum}: building shape descriptors'):
            descriptorsByKey[key] = desc

    newLabels = []
    newVecs = []
    newFailsafePoints = {}
    uniqueRows = []
    # Every hit where the cascade's verdict disagrees with curvature's or D2's own independent
    # verdict, whether or not the cascade ended up accepting it - logged regardless of outcome so
    # a rejected-but-the-old-methods-would-have-kept-it case is visible too, not just the
    # accepted ones that make it into uniqueRows.
    disagreementRows = []
    # (label, flatW, childFailsafePos, lleMean) for the trajectory-saving pass below - deferred
    # until after the CSVs are safely written (see that pass for why).
    trajectoryTasks = []

    # Phase 1 (parallel): check every hit against the reference catalog as it stood at the START
    # of this round - a frozen snapshot with no intra-round dependency, so this is embarrassingly
    # parallel across hits. This is also the dominant, catalog-size-sensitive cost (the catalog
    # only grows across rounds, never shrinks), which is exactly why it's the part worth
    # parallelizing - the K=5-shortlist-then-ICP-refine cascade itself stays bounded regardless
    # of how large the catalog gets.
    global _phase1RefNames, _phase1RefZernike, _phase1RefIcpPoints, _phase1RefCurv, _phase1RefD2
    _phase1RefNames = list(referenceDescriptors.keys())
    _phase1RefZernike = [referenceDescriptors[n]['zernike'] for n in _phase1RefNames]
    _phase1RefIcpPoints = [referenceDescriptors[n]['icpPoints'] for n in _phase1RefNames]
    _phase1RefCurv = np.array([referenceDescriptors[n]['curvature'] for n in _phase1RefNames])
    _phase1RefD2 = np.array([referenceDescriptors[n]['d2'] for n in _phase1RefNames])

    phase1Tasks = [(i, descriptorsByKey[('hit', i)]) for i in range(len(hitRows))]
    phase1Results = {}
    with Pool(initializer=_silenceFortranStdout) as pool:
        for i, result in tqdm(pool.imap(_phase1Task, phase1Tasks), total=len(phase1Tasks),
                               desc=f'Round {roundNum}: novelty-checking hits vs known catalog'):
            phase1Results[i] = result

    # Phase 2 (serial, restricted to phase-1 survivors only): a hit found later in this round
    # must still be checked against hits already accepted earlier in this SAME round, so two
    # same-round near-duplicates can't both slip through - that's an inherently serial,
    # order-dependent property, not something phase 1's frozen-catalog snapshot can capture.
    # Phase 1's result for "distinct from the catalog as it stood before this round" can't
    # change mid-round, so it's reused here rather than recomputed - only the small, round-local
    # set of hits accepted so far this round needs a fresh shortlist+ICP check.
    survivorIdx = sorted(i for i, r in phase1Results.items() if r is not None and r['isDistinctVsCatalog'])
    roundLocalNames, roundLocalZernike, roundLocalIcpPoints = [], [], []

    for i in tqdm(survivorIdx, desc=f"Round {roundNum}: deduping against this round's own new hits"):
        row = hitRows[i]
        desc = descriptorsByKey[('hit', i)]
        p1 = phase1Results[i]

        if roundLocalNames:
            zernikeDists = zernikeDistanceBatch(desc['zernike'], roundLocalZernike)
            shortlistIdx = np.argsort(zernikeDists)[:SHORTLIST_K]
            bestResidual, bestName = np.inf, None
            for idx in shortlistIdx:
                residual, _, _ = icpAlign(desc['icpPoints'], roundLocalIcpPoints[idx], nRestarts=ICP_RESTARTS)
                if residual < bestResidual:
                    bestResidual, bestName = residual, roundLocalNames[idx]
            isDistinct = bestResidual > NOVELTY_DISTANCE_THRESHOLD_CASCADE
            if bestResidual < p1['cascadeResidual']:
                cascadeNearestName, cascadeNearestDist = bestName, bestResidual
            else:
                cascadeNearestName, cascadeNearestDist = p1['cascadeNearestName'], p1['cascadeResidual']
        else:
            isDistinct = True
            cascadeNearestName, cascadeNearestDist = p1['cascadeNearestName'], p1['cascadeResidual']

        methodsDisagree = isDistinct != p1['curvIsDistinct'] or isDistinct != p1['d2IsDistinct']

        if methodsDisagree:
            disagreementRows.append({'ParentLabel': row['ParentLabel'],
                                      'CascadeVerdict': 'distinct' if isDistinct else 'same',
                                      'CurvatureVerdict': 'distinct' if p1['curvIsDistinct'] else 'same',
                                      'D2Verdict': 'distinct' if p1['d2IsDistinct'] else 'same',
                                      'CascadeNearestReference': cascadeNearestName,
                                      'CascadeDistance': cascadeNearestDist,
                                      'CurvatureNearestReference': p1['curvNearestName'],
                                      'CurvatureDistance': p1['curvNearestDist'],
                                      'D2NearestReference': p1['d2NearestName'],
                                      'D2Distance': p1['d2NearestDist'],
                                      'MeanCurvature': desc['meanCurvature'],
                                      'LLEMean': row['LLEMean'],
                                      **dict(zip(W_COLUMNS, np.array(row['W'], dtype=np.float64)))})

        if isDistinct:
            label = f'NewHit_{newHitCount}'
            referenceDescriptors[label] = desc
            newHitCount += 1
            flatW = np.array(row['W'], dtype=np.float64)
            newLabels.append(label)
            newVecs.append(flatW / np.linalg.norm(flatW))
            newFailsafePoints[label] = {'point': row['ChildFailsafePos'], 'depth': row['ChildFailsafeDepth']}
            roundLocalNames.append(label)
            roundLocalZernike.append(desc['zernike'])
            roundLocalIcpPoints.append(desc['icpPoints'])
            uniqueRows.append({'Label': label, 'ParentLabel': row['ParentLabel'],
                                'NearestReference': cascadeNearestName, 'Distance': cascadeNearestDist,
                                'CurvatureNearestReference': p1['curvNearestName'],
                                'CurvatureDistance': p1['curvNearestDist'],
                                'D2NearestReference': p1['d2NearestName'], 'D2Distance': p1['d2NearestDist'],
                                'MethodsDisagree': methodsDisagree,
                                'MeanCurvature': desc['meanCurvature'],
                                'UsedFailsafe': row['UsedFailsafe'],
                                'FailsafeChainDepth': row['ChildFailsafeDepth'],
                                # Whether descriptor-building itself needed the failsafe rescue on
                                # at least one pooled replica - distinct from UsedFailsafe above,
                                # which is about this hit's own earlier evaluation step, not this.
                                'DescriptorUsedFailsafe': desc['usedFailsafe'],
                                'LLEMean': row['LLEMean'], 'LLEPassRate': row['LLEPassRate'],
                                'CorrDimMean': row['CorrDimMean'], 'CorrDimStd': row['CorrDimStd'],
                                'NDraws': row['NDraws'], 'Escalated': row['Escalated'],
                                'LLEDraws': row['LLEDraws'], 'CorrDimDraws': row['CorrDimDraws'],
                                **dict(zip(W_COLUMNS, flatW))})
            trajectoryTasks.append((label, flatW, row['ChildFailsafePos'], row['LLEMean']))

    logResult(f"{len(newLabels)} of {len(hitRows)} hits this round were distinct from every "
              f"known attractor and every other hit found so far.")
    if disagreementRows:
        logResult(f"{len(disagreementRows)} of {len(hitRows)} hits had the cascade disagree with "
                  f"curvature or D2 on the novelty verdict - see disagreements.csv.")
        pd.DataFrame(disagreementRows).to_csv(os.path.join(roundDir, 'disagreements.csv'), index=False)

    if uniqueRows:
        uniqueDF = pd.DataFrame(uniqueRows)
        uniqueDF.to_csv(uniqueHitsPath, index=False)
        cumulativeDF = uniqueDF.copy()
        cumulativeDF.insert(0, 'Round', roundNum)
        writeHeader = not os.path.exists(ALL_UNIQUE_HITS_PATH)
        cumulativeDF.to_csv(ALL_UNIQUE_HITS_PATH, mode='a', index=False, header=writeHeader)

    # Trajectory-saving happens LAST, after the CSVs above are already safely on disk - it's the
    # part that does real integration + disk I/O per discovery (the thing that crashed mid-round
    # once already, on disk exhaustion). If it fails partway through now, this round's actual
    # discovery data (coefficients, classification stats) is not lost with it - only some
    # trajectory figures would need regenerating. One tradeoff worth knowing: if a future crash
    # happens in this exact window (after the CSV writes above, during this loop) and the round
    # gets redone from checkpoint, the CSVs above will already have this round's rows in them and
    # get a second, duplicate set appended under new NewHit_ labels on the redo - annoying but
    # trivially fixable, unlike the outright data loss this replaces.
    saveDiscoveryTrajectoriesBatch(trajectoryTasks, roundDir)

    return (np.array(newLabels), np.array(newVecs) if newVecs else np.empty((0, 81)),
            newHitCount, nextSeed, newFailsafePoints)


# On resume, each driver's main() already moves a stale round's own directory aside to
# backupPath (search '_interrupted_partial'). That round's rows were also already appended to
# the cumulative CSV before the crash, so call this right after - it moves those same rows out
# of the cumulative CSV and into backupPath too, instead of leaving them to be re-appended a
# second time (under colliding labels) once the round is redone.
def moveRoundRowsFromCumulativeCsv(roundNum, backupPath):
    if not os.path.exists(ALL_UNIQUE_HITS_PATH):
        return
    existing = pd.read_csv(ALL_UNIQUE_HITS_PATH)
    staleRows = existing[existing['Round'] == roundNum]
    if len(staleRows) == 0:
        return
    os.makedirs(backupPath, exist_ok=True)
    staleRows.to_csv(os.path.join(backupPath, 'allUniqueHits_interrupted_partial.csv'), index=False)
    existing[existing['Round'] != roundNum].to_csv(ALL_UNIQUE_HITS_PATH, index=False)
    logResult(f"Moved {len(staleRows)} row(s) for round {roundNum} out of the cumulative discovery "
              f"file into {backupPath}.")


def saveTaperPlot(roundCounts):
    if not roundCounts:
        return
    rounds, counts = zip(*roundCounts)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, counts, marker='o', color='#2a78d6')
    plt.xlabel('Round')
    plt.ylabel('New unique attractors found')
    plt.title('New unique discoveries per round')
    plt.grid(alpha=0.3)
    plt.savefig(TAPER_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close('all')


def _checkpointPath(roundNum):
    return os.path.join(CHECKPOINT_DIR, f'checkpoint_round_{roundNum:03d}.pkl')


# Written after every round completes, holding everything needed to start the round named in the
# filename without recomputing it - most importantly referenceDescriptors, which (unlike
# seeds/catalog) can't be faithfully reconstructed after the fact from the CSVs alone. One file
# per round (not overwritten) so a round history survives even if the newest checkpoint is
# somehow bad. tmp-then-rename makes the write atomic - an interruption mid-write can never leave
# a corrupt, half-written checkpoint behind for the next resume to trip over.
def saveCheckpoint(roundNum, seedLabels, seedVecs, catalogVecs, referenceDescriptors, newHitCount,
                    nextSeed, knownFailsafePoints, roundCounts):
    state = {
        'roundNum': roundNum, 'seedLabels': list(seedLabels), 'seedVecs': np.asarray(seedVecs),
        'catalogVecs': np.asarray(catalogVecs), 'referenceDescriptors': referenceDescriptors,
        'newHitCount': newHitCount, 'nextSeed': nextSeed,
        'knownFailsafePoints': knownFailsafePoints, 'roundCounts': roundCounts,
    }
    path = _checkpointPath(roundNum)
    tmpPath = path + '.tmp'
    with open(tmpPath, 'wb') as f:
        pickle.dump(state, f)
    os.replace(tmpPath, path)


# Picks the highest-numbered checkpoint (the most recent round boundary reached), so a run that
# got through several rounds after its last interruption doesn't fall back further than it has to.
def loadLatestCheckpoint():
    found = {}
    for path in glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_round_*.pkl')):
        m = re.match(r'checkpoint_round_(\d+)\.pkl$', os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    if not found:
        return None
    with open(found[max(found)], 'rb') as f:
        return pickle.load(f)
