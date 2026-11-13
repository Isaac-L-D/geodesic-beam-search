import os
import sys
_realStdoutFd = os.dup(1)
_devnullFd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnullFd, 1)
os.close(_devnullFd)

import json
import numpy as np
sys.path.insert(0, "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors")
from ODEClass import ODE, logDetails

payload = json.loads(sys.stdin.read())
W = np.array(payload["W"], dtype=np.float64).reshape(3, 27)
initial_pos = payload["initial_pos"]
ode = ODE(W=W, doFitToCube=False)
log = logDetails()
result = ode.integrate(initial_pos=initial_pos, log=log)
t, y = result
out = {
    "status": log.status,
    "minDist": log.minDist,
    "n_points": len(t),
    "last_t": float(t[-1]),
    "last_y": [float(y[0][-1]), float(y[1][-1]), float(y[2][-1])],
}
os.write(_realStdoutFd, json.dumps(out).encode())
