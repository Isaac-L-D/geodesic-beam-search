import csv
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from scipy.stats import wasserstein_distance

# ODEClass.py lives one level up, at the project root, shared with other scripts there.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from ODEClass import ODE, logDetails
from zernikeIcpCascade import zernikeDescriptor

# Bounded, fixed-size reference-point subsample per catalog entry for ICP alignment - float32,
# not float64, and modestly larger than icpAlign's own internal nPoints=1500 subsample so that
# internal subsample still has real variety to draw from rather than reusing nearly the whole
# stored array verbatim. Must stay fixed regardless of catalog size, unlike storing every pooled
# replica in full (which would scale checkpoint size with catalog growth - see the project's own
# round-over-round discovery-count growth in engine.py).
N_ICP_REFERENCE_POINTS = 2000


# DVODE writes diagnostics straight to fd 1, bypassing Python's warnings module - redirect
# stdout to /dev/null so it can't break the tqdm bar.
def _silenceFortranStdout():
    devnullFd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnullFd, 1)
    os.close(devnullFd)

TRANSIENT_FRACTION = 0.2     # drop the first 20% of samples as settling-in transient
D2_SAMPLE_PAIRS = 3000       # random point pairs sampled per attractor for its shape distribution
RANDOM_STATE = 0
USE_FULL_DATASET = True  # attractorDataset.csv (55 attractors) instead of selectedStrangeAttractors.csv (9)
ATTRACTOR_DATASET_PATH = os.path.join(PROJECT_ROOT, 'attractorDataset.csv')
RESULTS_LOG_PATH = os.path.join(SCRIPT_DIR, 'attractorSimilarityValidation.txt')
DISTANCE_MATRIX_PATH = os.path.join(SCRIPT_DIR, 'attractorSimilarityMatrix.csv')


# fitToCube=False: integration needs each system's natural dynamics speed, not a
# normalized-for-statistics version.
def loadDatasetAttractors(path=ATTRACTOR_DATASET_PATH):
    known = {}
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            ode, name = ODE.loadFromCSVString(row, fitToCube=False)
            known[name] = ode.W
    return known


def logResult(line=""):
    print(line)
    with open(RESULTS_LOG_PATH, 'a') as f:
        f.write(line + "\n")


# Sampling by fixed TIME isn't scale-of-W invariant - a unit-norm candidate moves slower and
# traces less of its attractor. ARC LENGTH is invariant: scaling W changes speed but not the
# spatial path, so stopping at TARGET_ARC_LENGTH samples the same shape regardless of norm.
# Calibrated via a convergence sweep (see discoveryPipeline/ResearchSummary.txt Phase 6):
# self-distance across known attractors of varying natural speed plateaus by ~30x each one's
# own old-method (fixed-time) arc length, without degrading further at higher multipliers for
# non-flaky attractors. Anchored on lorenz's 30x point (682,741, self-distance 0.018).
TARGET_ARC_LENGTH = 700000
N_OUTPUT_POINTS = 30000
# Backstop for trajectories that never reach the target arc length (e.g. settling onto a
# fixed point has near-zero speed, so the arc-length event would otherwise never fire).
# Sized with real margin above the ~227,000 model-time units a unit-norm lorenz-direction
# candidate needs to reach TARGET_ARC_LENGTH (native lorenz needs ~7,200, x its norm 31.53
# for the unit-norm case). The stall-detection fix in arcLengthTrialWorker.py means most
# genuinely-non-chaotic candidates (settled fixed points) reject in seconds rather than
# consuming this cap, so a moderate value with margin - not a maximal one - is the right call.
MAX_TIME_CAP = 1000000.0

# Same 600s total budget as before, reallocated: arcLengthTimeouts.log showed a 150s timeout
# is near-deterministic (3/163 recovered on retry), so one longer attempt beats four short ones.
ARC_LENGTH_TIMEOUT_SECONDS = 600
ARC_LENGTH_TIMEOUT_RETRIES = 0
ARC_LENGTH_HELPER_PATH = os.path.join(PROJECT_ROOT, "arcLengthTrialWorkerJitcode.py")
ARC_LENGTH_TIMEOUT_LOG_PATH = os.path.join(SCRIPT_DIR, 'arcLengthTimeouts.log')


# Makes timeout frequency observable instead of silently absorbed into "novelty check failed".
# Plain file append, not a return value - Pool workers run with stdout silenced, and this needs
# to work from every caller without changing their signatures.
def _logArcLengthTimeout(attempt, gaveUp):
    line = f"attempt {attempt + 1}/{ARC_LENGTH_TIMEOUT_RETRIES + 1} timed out - "
    line += "giving up" if gaveUp else "retrying"
    with open(ARC_LENGTH_TIMEOUT_LOG_PATH, 'a') as f:
        f.write(line + "\n")


# Runs in a subprocess with an OS-level timeout - solve_ivp has no wall-clock limit, and a
# pathological W can grind for a long real time even with maxTimeCap bounding model time. The
# worker integrates every requested initial condition sequentially in one compiled integrator, so
# the timeout is scaled by batch size - each replica still gets its own full budget, just pooled
# into one subprocess call instead of one call each.
def getAttractorPointsBatchWithTimeout(W, initials, targetArcLength, nPoints, maxTimeCap,
                                        failsafePos=None):
    payload = json.dumps({
        "W": W.flatten().tolist(),
        "initial_positions": initials,
        "targetArcLength": targetArcLength,
        "nPoints": nPoints,
        "maxTimeCap": maxTimeCap,
        "failsafePos": failsafePos,
    })
    batchTimeoutSeconds = ARC_LENGTH_TIMEOUT_SECONDS * len(initials)
    for attempt in range(ARC_LENGTH_TIMEOUT_RETRIES + 1):
        try:
            result = subprocess.run(
                [sys.executable, ARC_LENGTH_HELPER_PATH],
                input=payload,
                capture_output=True,
                text=True,
                timeout=batchTimeoutSeconds,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                return [None] * len(initials), [False] * len(initials)
            try:
                parsed = json.loads(result.stdout)
                points = [np.array(p) if p is not None else None for p in parsed["pointsByReplica"]]
                usedFailsafe = parsed.get("usedFailsafeByReplica", [False] * len(initials))
                return points, usedFailsafe
            except (json.JSONDecodeError, KeyError):
                return [None] * len(initials), [False] * len(initials)
        except subprocess.TimeoutExpired:
            gaveUp = attempt >= ARC_LENGTH_TIMEOUT_RETRIES
            _logArcLengthTimeout(attempt, gaveUp)
            if not gaveUp:
                continue
            return [None] * len(initials), [False] * len(initials)
        except Exception:
            return [None] * len(initials), [False] * len(initials)  # e.g. failed fork() under heavy load


def getAttractorPointsWithTimeout(W, initial, targetArcLength, nPoints, maxTimeCap):
    points, _ = getAttractorPointsBatchWithTimeout(W, [initial], targetArcLength, nPoints, maxTimeCap)
    return points[0]


def getAttractorPointsBatch(W, rng, nRuns, targetArcLength=TARGET_ARC_LENGTH,
                             nPoints=N_OUTPUT_POINTS, maxTimeCap=MAX_TIME_CAP, failsafePos=None):
    initials = [rng.uniform(-1, 1, size=3).tolist() for _ in range(nRuns)]
    pointsByReplica, usedFailsafeByReplica = getAttractorPointsBatchWithTimeout(
        W, initials, targetArcLength, nPoints, maxTimeCap, failsafePos)
    # Per-replica actual length, not the requested nPoints: the arc-length worker's point count
    # now varies per replica (each candidate's fine step adapts to its own dynamics speed rather
    # than landing on exactly nPoints), so a cutoff derived from the requested count would trim
    # the wrong fraction of a replica whose actual length differs from it.
    trimmed = [p[int(len(p) * TRANSIENT_FRACTION):] if p is not None else None for p in pointsByReplica]
    return trimmed, usedFailsafeByReplica


def getAttractorPoints(W, rng, targetArcLength=TARGET_ARC_LENGTH, nPoints=N_OUTPUT_POINTS,
                        maxTimeCap=MAX_TIME_CAP):
    points, _ = getAttractorPointsBatch(W, rng, 1, targetArcLength, nPoints, maxTimeCap)
    return points[0]


# D2 shape distribution (Osada et al.): distances between random point pairs, centered and
# scale-normalized so it compares shape only, independent of position or size.
def d2Descriptor(points, rng, nPairs=D2_SAMPLE_PAIRS):
    centered = points - points.mean(axis=0)
    scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
    if scale < 1e-12:
        return None
    normalized = centered / scale
    n = len(normalized)
    idxA = rng.integers(0, n, size=nPairs)
    idxB = rng.integers(0, n, size=nPairs)
    return np.linalg.norm(normalized[idxA] - normalized[idxB], axis=1)


def attractorDistance(descA, descB):
    return wasserstein_distance(descA, descB)


CURVATURE_SAMPLE_SIZE = D2_SAMPLE_PAIRS  # keeps curvature descriptors the same fixed length as
                                          # D2's, so attractorDistanceBatch's sorted-difference
                                          # shortcut (which requires equal-length descriptors)
                                          # applies to both, not just D2.


# Vectorized evaluation of the 27-monomial cubic basis and its Jacobian at every point at once -
# the same basis ODEClass.eval/computeJacobian use, but sympy's lambdify output isn't array-
# broadcastable for this Matrix-wrapped form, so this is a plain-numpy equivalent instead.
def _monomialsVectorized(x, y, z):
    n = len(x)
    terms = np.empty((27, n))
    for i in range(27):
        d1, d2, d3 = i // 9, (i // 3) % 3, i % 3
        terms[i] = (x ** d1) * (y ** d2) * (z ** d3)
    return terms


def _monomialJacobianVectorized(x, y, z):
    n = len(x)
    jac = np.zeros((27, 3, n))
    for i in range(27):
        d1, d2, d3 = i // 9, (i // 3) % 3, i % 3
        if d1 > 0: jac[i, 0] = d1 * (x ** (d1 - 1)) * (y ** d2) * (z ** d3)
        if d2 > 0: jac[i, 1] = d2 * (x ** d1) * (y ** (d2 - 1)) * (z ** d3)
        if d3 > 0: jac[i, 2] = d3 * (x ** d1) * (y ** d2) * (z ** (d3 - 1))
    return jac


# Curvature kappa = |v x a| / |v|^3, computed analytically at each point directly from W
# (v=f(x), a=J(x)@v via the chain rule) rather than from finite differences between neighboring
# samples - a point's curvature doesn't depend on how densely neighboring points were sampled in
# time, only on its own position, which is what makes it immune to the sampling-density issues
# D2 is exposed to. Time-weighted, not arc-length-weighted: tested both during validation, and
# time-weighting separated known/discovered attractors better - the amount of time a system
# lingers in sharply-curved regions turned out to be real signal here, not just sampling noise.
# Subsampled to a fixed size regardless of point count (which now varies per replica) so every
# descriptor stays the same length, matching D2's own fixed-nPairs convention.
def curvatureDescriptor(W, points, rng, nSample=CURVATURE_SAMPLE_SIZE):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    centered = points - points.mean(axis=0)
    scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
    if scale < 1e-12:
        return None
    v = W @ _monomialsVectorized(x, y, z)
    jFull = np.einsum('oi,ivn->ovn', W, _monomialJacobianVectorized(x, y, z))
    a = np.einsum('ovn,vn->on', jFull, v)
    cross = np.cross(v.T, a.T)
    speed = np.linalg.norm(v, axis=0)
    speedSafe = np.where(speed < 1e-10, 1.0, speed)
    kappa = np.linalg.norm(cross, axis=1) / speedSafe ** 3
    kappa[speed < 1e-10] = 0.0
    kappa = kappa * scale  # dimensionless, relative to this attractor's own size
    idx = rng.integers(0, len(kappa), size=nSample)
    return kappa[idx]


# Wasserstein-1 distance between two EQUAL-LENGTH 1D samples reduces exactly to
# mean(|sorted(a) - sorted(b)|) - every descriptor is always exactly D2_SAMPLE_PAIRS long, so
# this always applies. Checking a new hit against a large, ever-growing reference set (every
# known attractor plus every hit found in every round so far) is one row per reference here
# instead of one scipy call per reference, which measured ~11x faster.
def attractorDistanceBatch(desc, refDescs):
    sortedDesc = np.sort(desc)
    sortedRefs = np.sort(np.asarray(refDescs), axis=1)
    return np.abs(sortedDesc[None, :] - sortedRefs).mean(axis=1)


# Iglewicz & Hoaglin robust-outlier rule: median+MAD instead of mean+std, since MAD isn't
# dragged around by the outliers it's meant to detect.
MAD_OUTLIER_THRESHOLD = 3.5


def modifiedZScores(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-12:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


# Pools several random-initial-condition runs per descriptor: some systems are multistable
# (different initial conditions settle onto different attracting sets), which made single-run
# self-distance misleadingly large for some attractors. Pooling averages that out.
N_RUNS_PER_DESCRIPTOR = 8


def descriptorForW(W, rng, nRuns=N_RUNS_PER_DESCRIPTOR, failsafePos=None):
    pointsByReplica, usedFailsafeByReplica = getAttractorPointsBatch(W, rng, nRuns,
                                                                      failsafePos=failsafePos)
    pooledReplicas = [(p, uf) for p, uf in zip(pointsByReplica, usedFailsafeByReplica)
                      if p is not None and len(p) >= 20]
    if not pooledReplicas:
        return None
    pooled = np.concatenate([p for p, _ in pooledReplicas], axis=0)
    curvDesc = curvatureDescriptor(W, pooled, rng)
    # The Zernike+ICP cascade (see engine.py's runRound) drives the novelty
    # decision. Curvature and D2 are computed alongside as secondary disagreement flags for human
    # review, not to drive the decision themselves - see perceptualSimilarity/
    # cascadeValidationResults.npz and curvatureCascadeComparisonResults.npz for the validation
    # this is based on.
    #
    # pooled is (N,3) - renderSilhouette/zernikeDescriptor want (3,N), the OPPOSITE convention -
    # hence the transpose here. icpReferencePoints below is sliced straight from pooled with NO
    # transpose (icpAlign already expects (N,3)) - this asymmetry has caused two real bugs this
    # project already, so it's called out explicitly rather than left implicit.
    assert pooled.shape[1] == 3
    zernikeDesc = zernikeDescriptor(pooled.T)
    icpReferencePoints = _subsampleForIcp(pooled, rng)
    return {
        'curvature': curvDesc,
        'd2': d2Descriptor(pooled, rng),
        'meanCurvature': float(np.mean(curvDesc)) if curvDesc is not None else None,
        # True if any pooled replica needed the failsafe rescue to converge at all - a
        # reliability signal distinct from whether the candidate's own EARLIER evaluation step
        # used failsafe (see UsedFailsafe elsewhere), since this is specifically about whether
        # descriptor-building itself struggled.
        'usedFailsafe': any(uf for _, uf in pooledReplicas),
        'zernike': zernikeDesc,
        'icpPoints': icpReferencePoints,
    }


# Fixed-size float32 subsample, independent of catalog size (see N_ICP_REFERENCE_POINTS) - drawn
# from the flat pooled cloud (not per-replica), so a multistable system's stored reference
# naturally reflects whatever mix of basins its replicas actually landed in.
def _subsampleForIcp(pooled, rng, nPoints=N_ICP_REFERENCE_POINTS):
    if len(pooled) <= nPoints:
        return pooled.astype(np.float32)
    idx = rng.choice(len(pooled), size=nPoints, replace=False)
    return pooled[idx].astype(np.float32)


# Flags runs that landed somewhere qualitatively different from an attractor's other runs
# (e.g. a different basin under multistability) before pooling. Self-referential per
# attractor, so it adapts instead of needing one global cutoff.
def filterOutlierRuns(runPointSets, rng):
    if len(runPointSets) < 4:
        return list(range(len(runPointSets)))  # too few to judge robustly, keep all
    descriptors = [d2Descriptor(pts, rng) for pts in runPointSets]
    valid = [i for i, d in enumerate(descriptors) if d is not None]
    if len(valid) < 4:
        return valid
    n = len(valid)
    distMat = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            d = attractorDistance(descriptors[valid[a]], descriptors[valid[b]])
            distMat[a, b] = distMat[b, a] = d
    typicalDist = np.array([np.median(np.delete(distMat[i], i)) for i in range(n)])
    z = modifiedZScores(typicalDist)
    kept = [valid[i] for i in range(n) if z[i] <= MAD_OUTLIER_THRESHOLD]
    return kept if kept else valid  # never end up with zero runs


# Every (name, group, run) integration is independent, so all are sent to the pool together
# rather than parallelizing only within one attractor at a time.
def _integrateTask(task):
    name, group, W, seed = task
    rng = np.random.default_rng(seed)
    points = getAttractorPoints(W, rng)
    if points is None or len(points) < 20:
        return name, group, None
    return name, group, points


def main():
    if USE_FULL_DATASET:
        known = loadDatasetAttractors()  # {name: W}
    else:
        known = {name: v[0] for name, v in ODE().strangeAttractors.items()}  # {name: W}
    names = list(known.keys())

    logResult(f"Loaded {len(names)} known attractors: {names}")

    # Two independent groups of pooled runs per attractor give a same-system descriptor variance,
    # used as the noise floor against cross-attractor distances.
    tasks = []
    seed = RANDOM_STATE
    for name in names:
        W = known[name]
        for group in ('A', 'B'):
            for _ in range(N_RUNS_PER_DESCRIPTOR):
                tasks.append((name, group, W, seed))
                seed += 1

    pooledRuns = {(name, group): [] for name in names for group in ('A', 'B')}
    with Pool(initializer=_silenceFortranStdout) as pool:
        for name, group, points in tqdm(pool.imap_unordered(_integrateTask, tasks),
                                         total=len(tasks), desc='Integrating (parallel)'):
            if points is not None:
                pooledRuns[(name, group)].append(points)

    rng = np.random.default_rng(RANDOM_STATE)  # only used for d2Descriptor's pair-sampling now
    descA = {}
    descB = {}
    for name in names:
        # Both groups are nominally the same system, so together they're the reference population
        # for spotting individual runs that look out of place.
        allRuns = pooledRuns[(name, 'A')] + pooledRuns[(name, 'B')]
        groupOfRun = ['A'] * len(pooledRuns[(name, 'A')]) + ['B'] * len(pooledRuns[(name, 'B')])
        if not allRuns:
            descA[name] = None
            descB[name] = None
            logResult(f"  {name}: failed to integrate at all, dropping from validation.")
            continue

        keptIdx = filterOutlierRuns(allRuns, rng)
        nDiscarded = len(allRuns) - len(keptIdx)
        if nDiscarded:
            logResult(f"  {name}: discarded {nDiscarded}/{len(allRuns)} runs as outliers before pooling.")

        aPts = [allRuns[i] for i in keptIdx if groupOfRun[i] == 'A']
        bPts = [allRuns[i] for i in keptIdx if groupOfRun[i] == 'B']
        descA[name] = d2Descriptor(np.concatenate(aPts), rng) if aPts else None
        descB[name] = d2Descriptor(np.concatenate(bPts), rng) if bPts else None

    validNames = [n for n in names if descA.get(n) is not None and descB.get(n) is not None]

    logResult("\n=== Self-distance (same attractor, two different runs) ===")
    selfDistances = {}
    for name in validNames:
        d = attractorDistance(descA[name], descB[name])
        selfDistances[name] = d
        logResult(f"  {name}: {d:.4f}")

    # Flags attractors whose self-distance is a population outlier - even after filtering,
    # its two runs still don't agree, so comparisons involving it aren't trustworthy.
    selfDistArray = np.array([selfDistances[n] for n in validNames])
    selfZScores = modifiedZScores(selfDistArray)
    reliableNames = [n for n, z in zip(validNames, selfZScores) if z <= MAD_OUTLIER_THRESHOLD]
    unreliableNames = [n for n, z in zip(validNames, selfZScores) if z > MAD_OUTLIER_THRESHOLD]
    if unreliableNames:
        logResult(f"\n=== Unreliable (self-distance is a population outlier, z > {MAD_OUTLIER_THRESHOLD}) ===")
        for n, z in zip(validNames, selfZScores):
            if z > MAD_OUTLIER_THRESHOLD:
                logResult(f"  {n}: self-distance {selfDistances[n]:.4f}  z={z:.1f}")
    logResult(f"\n{len(reliableNames)}/{len(validNames)} attractors pass the reliability check.")

    logResult("\n=== Cross-attractor distance matrix (using descA for each) ===")
    matrix = pd.DataFrame(index=validNames, columns=validNames, dtype=float)
    for n1 in validNames:
        for n2 in validNames:
            matrix.loc[n1, n2] = attractorDistance(descA[n1], descA[n2])
    matrix.to_csv(DISTANCE_MATRIX_PATH)
    if len(validNames) <= 15:
        logResult(matrix.round(3).to_string())
    else:
        logResult(f"(matrix too large to print inline - {len(validNames)}x{len(validNames)}, see {DISTANCE_MATRIX_PATH})")

    crossDistances = [matrix.loc[n1, n2] for n1 in validNames for n2 in validNames if n1 != n2]
    meanSelf = np.mean(list(selfDistances.values()))
    meanCross = np.mean(crossDistances)
    logResult(f"\nMean self-distance (same attractor, all {len(validNames)}): {meanSelf:.4f}")
    logResult(f"Mean cross-distance (different attractors, all {len(validNames)}): {meanCross:.4f}")
    logResult(f"Separation ratio, all attractors: {meanCross / meanSelf:.2f}x")

    if len(reliableNames) >= 2:
        reliableCross = [matrix.loc[n1, n2] for n1 in reliableNames for n2 in reliableNames if n1 != n2]
        reliableMeanSelf = np.mean([selfDistances[n] for n in reliableNames])
        reliableMeanCross = np.mean(reliableCross)
        logResult(f"Separation ratio, reliable-only ({len(reliableNames)}): "
                  f"{reliableMeanCross / reliableMeanSelf:.2f}x")

    logResult("\n=== Nearest neighbor for each attractor (excluding itself) ===")
    for n1 in validNames:
        others = matrix.loc[n1].drop(n1)
        nearest = others.idxmin()
        flag = "" if n1 in reliableNames else "  [UNRELIABLE]"
        logResult(f"  {n1} -> closest to {nearest} (distance {others[nearest]:.3f}, "
                  f"vs its own self-distance {selfDistances[n1]:.3f}){flag}")

    print(f"\nDistance matrix saved to {DISTANCE_MATRIX_PATH}")


if __name__ == '__main__':
    main()
