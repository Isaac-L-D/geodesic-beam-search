import os
import sys
import json
import numpy as np

# Runs as a subprocess (see _saveOneTrajectoryTask in engine.py) rather than in-process
# inside the Pool worker, for the same reason integrateWithFailsafe/classifyChaosWithTimeout do:
# scipy's LSODA/BDF stepping happens inside compiled Fortran/C code that never hands control back
# to the Python interpreter, so a Python-level timeout (threading, signal.alarm, even
# Pool.apply_async's own get(timeout=...)) can only give up on WAITING for a hung solver - it can't
# actually stop one, and the Pool worker underneath stays wedged for the rest of that Pool's life.
# subprocess.run(timeout=...) kills the child at the OS level instead, which works regardless of
# what the solver is doing internally.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import _refineHighSpeedSegments, _resampleByArcLength, invertFlatten
from ODEClass import ODE


def main():
    payload = json.loads(sys.stdin.read())
    flatW = payload["flatW"]
    initialPos = payload["initialPos"]
    scaleBy = payload["scaleBy"]
    endTime = payload["endTime"]
    label = payload["label"]
    lleMean = payload["lleMean"]
    trajectoryDir = payload["trajectoryDir"]

    W = invertFlatten(np.asarray(flatW))
    ode = ODE(W=W * scaleBy, doFitToCube=False)
    result = ode.integrate(initial_pos=initialPos, endTime=endTime,
                            rtol=1e-8, atol=1e-10, denseOutput=True)
    if result is None:
        print(json.dumps({"success": False}))
        return

    t, y, denseSol = result
    originalPointCount = len(t)
    step = np.sqrt((np.diff(y, axis=1) ** 2).sum(axis=0))
    maxSpatialStep = 3 * np.median(step)
    t, y = _refineHighSpeedSegments(t, y, denseSol, maxSpatialStep)
    t, y = _resampleByArcLength(t, y, nPoints=originalPointCount)
    np.savez(os.path.join(trajectoryDir, f'{label}.npz'),
             t=t, x=y[0], y=y[1], z=y[2], W=np.asarray(flatW), scaleBy=scaleBy, lleMean=lleMean)
    print(json.dumps({"success": True}))


if __name__ == "__main__":
    main()