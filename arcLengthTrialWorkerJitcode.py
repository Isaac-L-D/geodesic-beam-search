import os
import sys

# Must run before any other import: jitcode's C-compilation step and DVODE/LSODA write raw
# diagnostics straight to fd 1, which would corrupt the JSON result below. Redirect fd 1 to
# /dev/null and write JSON through a preserved duplicate instead.
_realStdoutFd = os.dup(1)
_devnullFd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnullFd, 1)
os.close(_devnullFd)

import json
import glob
import math
import numpy as np
import sympy as sp
from jitcode import jitcode, y as jy, UnsuccessfulIntegration
from ODEClass import ODE

def _writeResult(obj):
    os.write(_realStdoutFd, json.dumps(obj).encode())

BLOWUP_BOUND = 500
TAU = 500.0
MIN_SPEED_THRESHOLD = 0.045
WARMUP = 3 * TAU

# Same reasoning/values as arcLengthTrialWorker.py's stall detector - see that file for the
# calibration behind TAU/MIN_SPEED_THRESHOLD/WARMUP.

# JiTCODE has no event/root-finding system like scipy's solve_ivp, so blowup/arc-length-target/
# stall detection is done by stepping forward in fixed CHECK_INTERVAL increments and checking
# after each - not exact root-finding. This is a deliberate simplification, not a shortcut that
# loses correctness: the target arc length only needs to be "reached," not hit exactly (that's
# the whole reason it's a calibrated threshold, not an exact requirement) - overshooting by up
# to one CHECK_INTERVAL (200 out of a 700,000 target) is negligible. A smaller interval would
# catch blowup/stall faster at the cost of more per-check Python overhead; 200 was chosen as a
# reasonable balance, not deeply tuned.
CHECK_INTERVAL = 200.0

# The 27-monomial basis is the same for every candidate this project evaluates - only the 81
# coefficients differ. Declaring them as control_pars (set at runtime via set_parameters())
# instead of baking numeric values in makes the compiled integrator generic, so compilation
# becomes a one-time cost cached to disk (see _compileOwn) instead of a per-candidate one.
PAR_SYMBOLS = sp.symbols('p0:81')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jitcodeCompiledCache')
CACHE_BASENAME = os.path.join(CACHE_DIR, 'arcLengthGenericCubic')


def _buildGenericAugmentedSystem():
    monomials = [1]
    for i in range(1, 27):
        d1, d2, d3 = i // 9, (i // 3) % 3, i % 3
        monomials.append((jy(0) ** d1) * (jy(1) ** d2) * (jy(2) ** d3))
    dx = sum(PAR_SYMBOLS[i] * monomials[i] for i in range(27))
    dy = sum(PAR_SYMBOLS[27 + i] * monomials[i] for i in range(27))
    dz = sum(PAR_SYMBOLS[54 + i] * monomials[i] for i in range(27))
    speed = sp.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    dm = (speed - jy(4)) / TAU
    return [dx, dy, dz, speed, dm]


# simplify=False: jitcode's Jacobian generation defaults to simplify=True unconditionally, which
# hangs (confirmed: 7+ minutes) on this system's Jacobian once coefficients are control_pars
# symbols rather than numbers with mostly-zero values to fold away. Skipping simplify doesn't
# change the Jacobian's value, only its unreduced symbolic form - confirmed against real
# discovered hits on a partial run(ended on round 1), agreement at ~1e-16 against the baked-in path.
def _compileOwn():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fAug = _buildGenericAugmentedSystem()
    j = jitcode(fAug, control_pars=PAR_SYMBOLS)
    j.generate_jac_sym(simplify=False)
    # do_cse: measured ~6% faster per-integration once compiled - worth it since compiling is a
    # one-time cost regardless.
    j.generate_f_C(do_cse=True)
    j.generate_jac_C(do_cse=True)
    # Compiling under our own destination name before set_integrator avoids a second, differently
    # named compile that set_integrator's own compile would otherwise trigger - the integrator
    # ends up wired to that first compile regardless, so a later mismatched one silently leaves
    # set_parameters with nothing connected to it.
    j.save_compiled(destination=f'{CACHE_BASENAME}_{os.getpid()}', overwrite=True)
    j.set_integrator('vode', method='bdf', atol=1e-6, rtol=1e-6)
    return j  # already a ready-to-use integrator, no need to reload what we just built


def _tryLoad(path):
    try:
        j = jitcode(n=5, control_pars=PAR_SYMBOLS, module_location=path)
        j.set_integrator('vode', method='bdf', atol=1e-6, rtol=1e-6)
        return j
    except Exception:
        return None  # e.g. another process's compile still in progress - fall back below


def _newIntegrator():
    for path in glob.glob(CACHE_BASENAME + '_*.so'):
        j = _tryLoad(path)
        if j is not None:
            return j
    return _compileOwn()


# One continuous walk from t=0 to tStop, with the fine step adapted each substep toward the
# target arc-length-per-point rate - tracks each candidate's own speed instead of a fixed time
# increment, and keeps the BDF solver's step-order history intact for the whole walk. Validated
# against known attractors and real discovered hits: no measurable change in descriptor distances.
# Point count per replica varies with the candidate's dynamics instead of landing on exactly
# nPoints - callers must use the actual returned length, not nPoints (see getAttractorPointsBatch).
def _runOneReplica(j, ode, initial, targetArcLength, nPoints, maxTimeCap):
    initialSpeed = np.linalg.norm(ode.eval(0, np.array(initial)))
    j.set_initial_value(initial + [0.0, initialSpeed], 0.0)

    targetDs = targetArcLength / nPoints
    fineStep = min(max(targetDs / max(initialSpeed, 1e-3), 1e-4), CHECK_INTERVAL)

    t = 0.0
    prevS = 0.0
    m = initialSpeed
    points = [initial]
    while t < maxTimeCap:
        checkEnd = min(t + CHECK_INTERVAL, maxTimeCap)
        while t < checkEnd:
            tNext = min(t + fineStep, checkEnd)
            try:
                state = j.integrate(tNext)
            except UnsuccessfulIntegration:
                return None
            # Unpacked once and checked with plain Python/math instead of np.all/np.isfinite/
            # np.max/np.clip: this runs on every one of ~180k substeps across a descriptor's 8
            # replicas, and profiling showed numpy's per-call dispatch overhead on a 5-element
            # array was itself ~15% of total time - bigger than the comparisons actually cost.
            # Same checks, same values, just without going through numpy's array machinery for
            # work that's really just 5 floats.
            x, y, z, s, m = state
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
                    and math.isfinite(s) and math.isfinite(m)):
                return None
            if max(abs(x), abs(y), abs(z)) > BLOWUP_BOUND:
                return None
            points.append([x, y, z])
            ds = s - prevS
            if ds > 1e-10:
                fineStep = min(max(fineStep * (targetDs / ds), 1e-4), CHECK_INTERVAL)
            prevS = s
            t = tNext
            if s >= targetArcLength:
                return points
        # Stall check stays at the same CHECK_INTERVAL cadence as the original two-phase version
        # (not every fine substep) - it's meant to catch a genuinely settled trajectory, not react
        # to normal short-timescale speed variation within one check window.
        if checkEnd >= WARMUP and m < MIN_SPEED_THRESHOLD:
            return None

    return None


def main():
    payload = json.loads(sys.stdin.read())
    W = np.array(payload["W"], dtype=np.float64).reshape(3, 27)
    initials = payload["initial_positions"]  # list of [x, y, z], one per requested replica
    targetArcLength = payload["targetArcLength"]
    nPoints = payload["nPoints"]
    maxTimeCap = payload["maxTimeCap"]
    failsafePos = payload.get("failsafePos")  # optional [x,y,z], only used as a last-resort retry

    ode = ODE(W=W, doFitToCube=False)

    # A fresh integrator per replica, not one reused across all of them: an unsimplified
    # control_pars Jacobian can leave the solver's internal state bad after a difficult
    # integration, in a way set_initial_value alone doesn't clear. Reloading from the cached
    # compiled module is still cheap - recompiling would be the expensive part, and that's
    # avoided either way.
    pointsByReplica = []
    usedFailsafeByReplica = []
    for initial in initials:
        j = _newIntegrator()
        j.set_parameters(*W.flatten().tolist())
        points = _runOneReplica(j, ode, initial, targetArcLength, nPoints, maxTimeCap)
        usedFailsafe = False
        # Only a replica whose own random IC failed falls back to the known-good point - one that
        # already succeeded keeps its own (possibly different-basin) result, so this doesn't
        # preempt multistability exploration, it only rescues replicas that would otherwise be
        # lost outright. Mirrors integrateWithFailsafe's own last-resort pattern.
        if points is None and failsafePos is not None:
            j = _newIntegrator()
            j.set_parameters(*W.flatten().tolist())
            points = _runOneReplica(j, ode, failsafePos, targetArcLength, nPoints, maxTimeCap)
            usedFailsafe = points is not None
        pointsByReplica.append(points)
        usedFailsafeByReplica.append(usedFailsafe)

    _writeResult({"pointsByReplica": pointsByReplica, "usedFailsafeByReplica": usedFailsafeByReplica})


if __name__ == "__main__":
    main()
