"""Tests for the runtime handle/lease implementations."""

import numpy as np
import pytest
from qphase.data import (
    AxisRole,
    AxisSchema,
    BackendArrayHandle,
    DataHandleProtocol,
    DataKind,
    DataLease,
    DataLeaseProtocol,
    HostArrayHandle,
    ProductSchema,
    ReadOnlyArrayView,
    RuntimeProductBacking,
    VariableSchema,
    validate_backing,
)


def _variable(name="alpha", dtype="complex128", dims=("time",)):
    return VariableSchema(
        name=name, dtype=dtype,
        value_domain="complex" if dtype.startswith("complex") else "real",
        dims=dims,
    )


def test_host_handle_protocol_and_materialize():
    """Host handles satisfy the protocol and materialize on cpu only."""
    array = np.arange(8, dtype=np.float64)
    handle = HostArrayHandle(
        array, _variable("x", "float64"), owner="engine.fake"
    )
    assert isinstance(handle, DataHandleProtocol)
    assert handle.device == "cpu"
    assert handle.dtype == "<f8"
    assert handle.shape == (8,)
    assert handle.nbytes == array.nbytes

    assert handle.materialize() is array
    assert handle.materialize("cpu", copy_policy="never") is array
    with pytest.raises(RuntimeError, match="cannot materialize"):
        handle.materialize("cuda:0")


def test_host_handle_validates_array_against_schema():
    """dtype/shape mismatches are rejected at construction."""
    with pytest.raises(ValueError, match="dtype"):
        HostArrayHandle(
            np.zeros(4, dtype=np.float64),
            _variable("z", "complex128"),
            owner="o",
        )
    with pytest.raises(ValueError, match="dims"):
        HostArrayHandle(
            np.zeros((4, 2), dtype=np.complex128),
            _variable("z", "complex128"),
            owner="o",
        )


def test_lease_lifecycle_and_close_semantics():
    """Leases are idempotent; owner close invalidates outstanding leases."""
    handle = HostArrayHandle(
        np.zeros(4), _variable("x", "float64"), owner="engine.fake"
    )
    lease = handle.acquire("analyser.spectrum")
    assert isinstance(lease, DataLeaseProtocol)
    assert lease.scope == "execution"
    assert handle.lease_count == 1

    lease.release()
    lease.release()  # idempotent
    assert handle.lease_count == 0

    outstanding = handle.acquire("analyser.other", scope="session")
    assert outstanding.scope == "session"
    handle.close()
    with pytest.raises(RuntimeError, match="closed"):
        handle.acquire("late.consumer")
    with pytest.raises(RuntimeError, match="closed"):
        handle.materialize()
    outstanding.release()  # release after close is still legal


def test_read_only_view_never_exposes_writable_array():
    """Read-only views return non-writeable arrays and hide the inner handle."""
    array = np.arange(4, dtype=np.float64)
    inner = HostArrayHandle(array, _variable("x", "float64"), owner="o")
    view = ReadOnlyArrayView(inner)
    assert view.read_only
    assert isinstance(view, DataHandleProtocol)

    lease = view.acquire("consumer")
    assert lease.handle is view  # the writable inner handle is not exposed

    materialized = view.materialize()
    assert not materialized.flags.writeable
    with pytest.raises(ValueError):
        materialized[0] = 1.0

    inner.close()
    with pytest.raises(RuntimeError, match="closed"):
        view.materialize()


def test_backend_handle_explicit_copy_policy():
    """Backend handles never copy D2H implicitly; 'never' raises."""
    array = np.arange(4, dtype=np.float64)
    handle = BackendArrayHandle(
        array, _variable("x", "float64"), owner="o", device="cuda:0"
    )
    assert handle.device == "cuda:0"
    assert handle.materialize() is array
    with pytest.raises(RuntimeError, match="copy policy"):
        handle.materialize("cpu", copy_policy="never")
    host = handle.materialize("cpu", copy_policy="allow")
    assert isinstance(host, np.ndarray)
    np.testing.assert_array_equal(host, array)
    with pytest.raises(RuntimeError, match="cross-device"):
        handle.materialize("cuda:1")


def test_handles_validate_against_product_backing():
    """Concrete handles plug into RuntimeProductBacking validation."""

    class _Backing:
        def __init__(self, variables):
            self._variables = variables

        @property
        def variables(self):
            return self._variables

    variable = _variable("x", "float64")
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[AxisSchema(name="time", role=AxisRole.COORDINATE, size=4)],
        variables=[variable],
    )
    handle = HostArrayHandle(np.zeros(4), variable, owner="engine.fake")
    backing = _Backing({"x": handle})
    assert isinstance(backing, RuntimeProductBacking)
    validate_backing(schema, backing)

    other = HostArrayHandle(
        np.zeros(4),
        VariableSchema(
            name="x", dtype="float64", value_domain="real",
            dims=("time",), units="Hz",
        ),
        owner="o",
    )
    with pytest.raises(ValueError, match="variable_schema"):
        validate_backing(schema, _Backing({"x": other}))


def test_data_lease_is_context_manager():
    """Leases work as context managers."""
    handle = HostArrayHandle(
        np.zeros(2), _variable("x", "float64"), owner="o"
    )
    with handle.acquire("consumer") as lease:
        assert isinstance(lease, DataLease)
        assert handle.lease_count == 1
    assert handle.lease_count == 0
