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
import random
import numpy as np
import sympy as sp
from jitcode import jitcode, y as jy, UnsuccessfulIntegration

# ODEClass.py lives at the project root, one level up from this file (unlike
# integrateTrialWorkerJitcode.py, which sits at the root itself and gets it for free) - Python
# only auto-adds a subprocess script's own directory to sys.path, not its cwd, so this needs to
# be explicit.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ODEClass import ODE

def _writeResult(obj):
    os.write(_realStdoutFd, json.dumps(obj).encode())

BLOWUP_BOUND = 500
CHECK_INTERVAL = 2.0
TAIL_WINDOW = 2.0
TAIL_SAMPLES = 50
STABLE_TOL = 1e-8
MAX_IC_RETRIES = 4

# The 81 coefficients are control_pars (set_parameters()) instead of baked into the compiled
# code, so compiling is a one-time project-wide cost, cached to disk, not a per-candidate one.
# Each attempt still gets a fresh (cache-reloaded, not recompiled) integrator rather than reusing
# one across retries - reuse was verified safe here, but reloading is nearly free and avoids the
# reuse-corruption risk found in arcLengthTrialWorkerJitcode.py's augmented system.
PAR_SYMBOLS = sp.symbols('p0:81')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jitcodeCompiledCache')
CACHE_BASENAME = os.path.join(CACHE_DIR, 'integrateGenericCubic')


def _buildGenericSystem():
    monomials = [1]
    for i in range(1, 27):
        d1, d2, d3 = i // 9, (i // 3) % 3, i % 3
        monomials.append((jy(0) ** d1) * (jy(1) ** d2) * (jy(2) ** d3))
    dx = sum(PAR_SYMBOLS[i] * monomials[i] for i in range(27))
    dy = sum(PAR_SYMBOLS[27 + i] * monomials[i] for i in range(27))
    dz = sum(PAR_SYMBOLS[54 + i] * monomials[i] for i in range(27))
    return [dx, dy, dz]


def _compileOwn():
    os.makedirs(CACHE_DIR, exist_ok=True)
    j = jitcode(_buildGenericSystem(), control_pars=PAR_SYMBOLS)
    # do_cse: measured ~15% faster per-integration once compiled - worth it since compiling is a
    # one-time cost regardless.
    j.generate_f_C(do_cse=True)
    j.generate_jac_C(do_cse=True)
    # Compiling under our own destination name before set_integrator avoids a second, differently
    # named compile that set_integrator's own compile would otherwise trigger - the integrator
    # ends up wired to that first compile regardless, so a later mismatched one silently leaves
    # set_parameters with nothing connected to it.
    j.save_compiled(destination=f'{CACHE_BASENAME}_{os.getpid()}', overwrite=True)
    j.set_integrator('vode', method='bdf', atol=1e-6, rtol=1e-6)
    return j


def _tryLoad(path):
    try:
        j = jitcode(n=3, control_pars=PAR_SYMBOLS, module_location=path)
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


# One attempt from a given initial condition, on its own freshly-reloaded integrator. Returns
# (success, status, minDist, lastPoint) - lastPoint is the tail's final [x,y,z], useful on its
# own as a "known good" point for whatever seeded this candidate, not just for the stabilization
# check.
def _attemptIntegration(WScaled, endTime, initial):
    j = _newIntegrator()
    j.set_parameters(*WScaled.flatten().tolist())
    j.set_initial_value(initial, 0.0)

    t = 0.0
    tailStart = max(endTime - TAIL_WINDOW, 0.0)
    while t < tailStart:
        tNext = min(t + CHECK_INTERVAL, tailStart)
        try:
            state = j.integrate(tNext)
        except UnsuccessfulIntegration:
            return False, "solve_ivp failed", "N/A", None
        # Unpacked once and checked with plain Python/math instead of np.all/np.isfinite/np.max:
        # this system is always 3-dimensional (x,y,z), so numpy's per-call array dispatch on a
        # 3-element array is pure overhead here - same checks, same values, just without it.
        # See arcLengthTrialWorkerJitcode.py's _runOneReplica for where this was first measured.
        x, y, z = state
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False, "Infinity in Result", "N/A", None
        if max(abs(x), abs(y), abs(z)) > BLOWUP_BOUND:
            return False, "Blowup", "N/A", None
        t = tNext

    tailTimes = np.linspace(tailStart, endTime, TAIL_SAMPLES)
    tailPoints = []
    for tOut in tailTimes:
        try:
            state = j.integrate(tOut)
        except UnsuccessfulIntegration:
            return False, "solve_ivp failed", "N/A", None
        x, y, z = state
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False, "Infinity in Result", "N/A", None
        if max(abs(x), abs(y), abs(z)) > BLOWUP_BOUND:
            return False, "Blowup", "N/A", None
        tailPoints.append(state.copy())

    tailPoints = np.vstack(tailPoints)
    consecutiveDists = np.linalg.norm(np.diff(tailPoints, axis=0), axis=1)
    minDist = float(consecutiveDists.min())
    status = "Stablized" if minDist < STABLE_TOL else "Success"
    return True, status, minDist, tailPoints[-1].tolist()


def main():
    payload = json.loads(sys.stdin.read())
    W = np.array(payload["W"], dtype=np.float64).reshape(3, 27)
    scaleBy = payload.get("scaleBy", 1)
    failsafePos = payload.get("failsafePos")  # optional [x,y,z], only used on the last attempt
    WScaled = W * scaleBy
    ode = ODE(W=WScaled, doFitToCube=False)
    endTime = ode.maxIntegrationTime / scaleBy

    # Random ICs for every attempt except the last - preserves full basin-exploration diversity
    # for the common case, and only falls back to a known-good point (the parent's own settled
    # trajectory, if supplied) as a last resort, so it never preempts a random attempt that
    # might have found different, equally-valid behavior. usedFailsafe marks whether that
    # last-resort attempt is what actually succeeded.
    success, status, minDist, lastPoint, usedFailsafe, initialPos = False, None, None, None, False, None
    for attempt in range(MAX_IC_RETRIES + 1):
        isLastAttempt = attempt == MAX_IC_RETRIES
        if isLastAttempt and failsafePos is not None:
            initial = failsafePos
            attemptIsFailsafe = True
        else:
            initial = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
            attemptIsFailsafe = False
        success, status, minDist, lastPoint = _attemptIntegration(WScaled, endTime, initial)
        if success:
            usedFailsafe = attemptIsFailsafe
            # Despite the name, initialPos is NOT always "a freshly chosen IC for this
            # candidate" - when usedFailsafe is True, initial (and so initialPos) is just
            # failsafePos, the parent's own settled point, passed through unchanged. See
            # engine.py's ChildFailsafePos assignment for why that distinction matters
            # to the caller.
            initialPos = initial
            break

    _writeResult({"success": success, "status": status, "minDist": minDist,
                  "lastPoint": lastPoint, "usedFailsafe": usedFailsafe, "initialPos": initialPos})


if __name__ == "__main__":
    main()
