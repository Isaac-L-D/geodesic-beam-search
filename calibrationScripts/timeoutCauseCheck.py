import os
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))

from ODEClass import ODE
from attractorSimilarity import loadDatasetAttractors, _silenceFortranStdout

# Three questions about why ~40% of runs are hitting the 600s timeout in the currently-running
# validation job:
#   1) Is it genuine near-stalling (speed drops toward zero, EMA-detector territory) or genuine
#      stiffness (speed stays reasonable but the solver needs many tiny steps anyway)?
#   2) Does a fixed implicit method (Radau/BDF - bigger steps in stiff regions, but more
#      per-step cost than LSODA's automatic switching) actually finish faster on these systems?
#   3) Does the analytical Jacobian (now wired into ODE.eval's companion computeJacobian) change
#      wall-clock time on these specifically, given it removes LSODA/Radau/BDF's own internal
#      finite-difference Jacobian estimation - extra eval() calls per Jacobian update - on top of
#      making each Jacobian evaluation itself faster?
# All three checked directly on the 5 known-slow attractors already flagged earlier this session
# (bouali2, sprotti, newtonliepnik, wangsun, atmosphericregime) - the same ones almost certainly
# behind a lot of tonight's timeouts, since they're deliberately in that validation set.

KNOWN_SLOW = ['bouali2', 'sprotti', 'newtonliepnik', 'wangsun', 'atmosphericregime']
DIAG_TIME_CAP = 5000.0  # bounded diagnostic window, not the full 700k-arc-length production run -
# just enough model-time to see the speed/step-density pattern without waiting for a full timeout.
METHODS = ['LSODA', 'Radau', 'BDF']
USE_JAC_OPTIONS = [False, True]
MIN_SPEED_THRESHOLD = 0.045  # same value as arcLengthTrialWorker.py's stall cutoff, for comparability
POOL_WORKERS = 3  # kept small - the NOVELTY_DISTANCE_THRESHOLD validation job may still be finishing
OUT_PATH = os.path.join(SCRIPT_DIR, 'timeoutCauseCheck.log')


def _task(args):
    name, W, method, useJac = args
    ode = ODE(W=W, doFitToCube=False)
    initial = np.array([0.1, 0.1, 0.1])
    jac = ode.jacobianForSolveIvp if useJac else None
    t0 = time.time()
    try:
        sol = solve_ivp(lambda t, y: ode.eval(t, y), (0, DIAG_TIME_CAP), initial, method=method, jac=jac)
    except Exception as e:
        return name, method, useJac, {'success': False, 'error': str(e), 'elapsed': time.time() - t0}
    elapsed = time.time() - t0
    if not sol.success:
        return name, method, useJac, {'success': False, 'error': sol.message, 'elapsed': elapsed}
    speeds = np.array([np.linalg.norm(ode.eval(t, sol.y[:, i])) for i, t in enumerate(sol.t)])
    return name, method, useJac, {
        'success': True, 'elapsed': elapsed, 'nSteps': len(sol.t),
        'modelTimeReached': float(sol.t[-1]), 'minSpeed': float(speeds.min()),
        'medianSpeed': float(np.median(speeds)),
        'fracStepsNearStall': float(np.mean(speeds < MIN_SPEED_THRESHOLD)),
    }


def main():
    known = loadDatasetAttractors()
    tasks = [(name, known[name], method, useJac)
             for name in KNOWN_SLOW for method in METHODS for useJac in USE_JAC_OPTIONS]

    log = open(OUT_PATH, 'w')
    def logLine(s):
        print(s)
        log.write(s + "\n")
        log.flush()

    logLine(f"Diagnosing {len(KNOWN_SLOW)} known-slow attractors x {len(METHODS)} methods, "
            f"bounded to {DIAG_TIME_CAP:.0f} model-time-units, {POOL_WORKERS} parallel workers")

    results = {}
    with Pool(POOL_WORKERS, initializer=_silenceFortranStdout) as pool:
        for name, method, useJac, r in pool.imap_unordered(_task, tasks):
            results.setdefault(name, {})[(method, useJac)] = r
            tag = f"{method}{'+jac' if useJac else ''}"
            if r['success']:
                logLine(f"  {name} [{tag}]: {r['elapsed']:.1f}s wall, {r['nSteps']} steps, "
                        f"reached model-time {r['modelTimeReached']:.0f}/{DIAG_TIME_CAP:.0f}, "
                        f"minSpeed={r['minSpeed']:.4f}, medianSpeed={r['medianSpeed']:.4f}, "
                        f"fracNearStall={r['fracStepsNearStall']:.2%}")
            else:
                logLine(f"  {name} [{tag}]: FAILED ({r.get('error','?')}) after {r['elapsed']:.1f}s")

    logLine("\n=== Interpretation ===")
    for name in KNOWN_SLOW:
        r = results.get(name, {}).get(('LSODA', False))
        if not r or not r['success']:
            logLine(f"{name}: LSODA diagnostic run itself failed/incomplete - inconclusive.")
            continue
        if r['fracStepsNearStall'] > 0.3:
            logLine(f"{name}: speed frequently near the stall threshold ({r['fracStepsNearStall']:.0%} "
                     f"of steps) - looks like genuinely SLOW coverage, not pure stiffness. "
                     f"W-scaling should help here.")
        else:
            stepsPerModelTime = r['nSteps'] / max(r['modelTimeReached'], 1e-9)
            logLine(f"{name}: speed stays reasonable (median {r['medianSpeed']:.3f}, rarely near "
                     f"stall) but took {r['nSteps']} steps ({stepsPerModelTime:.2f} steps/model-time-unit) "
                     f"- looks like genuine STIFFNESS. W-scaling likely won't help much (stiffness "
                     f"ratio is scale-invariant); compare wall-clock across methods above instead.")
        for method in ['LSODA', 'Radau', 'BDF']:
            rNoJac = results.get(name, {}).get((method, False))
            rJac = results.get(name, {}).get((method, True))
            if rNoJac and rJac and rNoJac['success'] and rJac['success']:
                speedup = rNoJac['elapsed'] / rJac['elapsed'] if rJac['elapsed'] > 0 else float('nan')
                logLine(f"  {method} jac speedup on {name}: {rNoJac['elapsed']:.1f}s -> "
                        f"{rJac['elapsed']:.1f}s ({speedup:.2f}x)")
        for method in ['Radau', 'BDF']:
            rm = results.get(name, {}).get((method, False))
            if rm and rm['success'] and r['success']:
                speedup = r['elapsed'] / rm['elapsed'] if rm['elapsed'] > 0 else float('nan')
                logLine(f"  {method} vs LSODA on {name}: {rm['elapsed']:.1f}s vs {r['elapsed']:.1f}s "
                        f"({speedup:.2f}x)")
    log.close()


if __name__ == '__main__':
    main()
