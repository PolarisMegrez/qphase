from __future__ import annotations

import numpy as np

from QPhaseSDE.core.protocols import BackendBase, StateBase, TrajectorySetBase
from QPhaseSDE.backends.numpy_backend import NumpyBackend
from QPhaseSDE.states.factory import make_state


def test_backend_minimal_contract():
    be = NumpyBackend()
    assert isinstance(be.backend_name(), str)
    x = be.asarray([1.0, 2.0])
    y = be.zeros((2, 3), dtype=float)
    z = be.empty((1,), dtype=float)
    assert x is not None and y.shape == (2, 3) and z.shape == (1,)


def test_state_view_and_copy_semantics():
    be = NumpyBackend()
    y0 = be.asarray(np.ones((3, 2), dtype=np.complex128))
    st: StateBase = make_state(be, y=y0, t=0.0, attrs={})
    v = st.view(modes=slice(None), trajectories=slice(None))
    c = st.copy()
    # Modify original and check view changes but copy does not
    st.y[0, 0] = 2 + 0j
    assert v.y[0, 0] == 2 + 0j
    assert c.y[0, 0] == 1 + 0j
