"""Connection retirement via a lease registry (card t_15fcb86f, issue #1163).

CONTRACT
========
A facade (``Mnemosyne`` or ``BeamMemory``) or a substore that receives a
shared ``_BeamConnection`` takes a *lease* on that connection. ``close()`` on
the facade releases its leases. A connection is physically closed exactly
when its lease count drops to zero. Everything is keyed by CONNECTION
IDENTITY (``id(conn)``), never by db path: a newer generation connection on
the same path is never affected by the retirement of an older object.

State (module-level, protected by ``_Registry.cond``)::

    _Registry.leases:        dict[int, int]    # id(conn) -> live lease count
    _Registry.connections:   dict[int, weakref.ref[_BeamConnection]]
    _Registry.retired:       dict[int, weakref.ref[_BeamConnection]]
        # id -> weakref of the EXACT object that was retired. A plain id set
        # would false-positive after CPython recycles the id for a NEW
        # connection object (long-running gateway churn); the weakref keeps
        # retirement keyed to the object, and dead entries are prunable.
    _Registry.cond:          threading.Condition

Expected lease counts for one ``Mnemosyne`` graph on a shared connection:
1 (``Mnemosyne`` wrapper) + 1 (``BeamMemory`` facade) + 1 per substore that
was constructed with the shared conn (annotations, canonical, media,
episodic_graph, veracity_consolidator) = 7 leases on a fully wired graph.
A standalone ``BeamMemory`` holds 1 + 5 = 6.

Semantics guarantees implemented here:

- Registration is idempotent per connection (double-register is a no-op).
- ``acquire``/``release`` are idempotent up to their natural bounds; release
  below zero is a bug: it is logged and clamped at zero.
- Lease release NEVER commits an open transaction. Callers must check
  ``conn.in_transaction`` before releasing; the deferred path (connection
  stays open, warning logged, ``close()`` returns) lives in the facades'
  ``close()``. An optional ``force=True`` rolls back and proceeds; the
  default path must never commit.
- On final release (count reaches zero) the facades run the physical close
  sequence: cancel pending state, ``conn.close()`` (sqlite close rolls back
  residue, never commits), then ``retire_connection`` which marks the id
  retired and scrubs every TLS cache entry holding the SAME OBJECT
  (identity-scoped, C3-safe). If the same object still sits in a TLS slot
  that is a bug: it is logged and the entry is still scrubbed (defensive).
- A weakref finalizer attached at registration is DEREGISTER-ONLY: it drops
  registry entries when the connection object is garbage collected. It never
  closes the connection (closing from ``__del__``-time callbacks can run on
  any thread; keep it predictable).

This module deliberately imports nothing from the rest of ``mnemosyne.core``
(TLS scrubbing looks modules up lazily via ``sys.modules``) so it stays
import-cycle free.
"""

from __future__ import annotations

import logging
import sys
import threading
import weakref

__all__ = [
    "acquire_lease",
    "connection_retirement_state",
    "is_retired",
    "register_connection",
    "release_lease",
    "retire_connection",
    "scrub_tls_by_identity",
]

logger = logging.getLogger(__name__)


class _Registry:
    """Module-level registry of connection leases (see module docstring)."""

    __slots__ = ("leases", "connections", "retired", "cond")

    def __init__(self) -> None:
        self.leases: dict[int, int] = {}
        self.connections: dict[int, "weakref.ref"] = {}
        self.retired: dict[int, "weakref.ref"] = {}
        self.cond = threading.Condition()


_Registry_instance = _Registry()


def _finalize(cid: int, registry: _Registry = _Registry_instance) -> None:
    """weakref.finalize hook: DEREGISTER-ONLY.

    Never closes the connection (callbacks may run on any thread and at
    unpredictable times); just drop the registry entries so a dead handle
    cannot pin state forever.
    """
    with registry.cond:
        registry.leases.pop(cid, None)
        registry.connections.pop(cid, None)
    logger.debug("retirement registry: deregistered dead connection id=%d", cid)


def register_connection(conn: object) -> int:
    """First sight of a handle: record it in the registry with leases=0.

    Idempotent: re-registering the same object is a no-op (the existing lease
    count is preserved). Attaches a deregister-only weakref finalizer.
    Returns the current lease count (0 for a fresh registration).
    """
    cid = id(conn)
    with _Registry_instance.cond:
        prior = _Registry_instance.retired.get(cid)
        if cid in _Registry_instance.leases and (
            prior is None or prior() is conn
        ):
            # Idempotent double-register of the SAME live object: no-op.
            return _Registry_instance.leases[cid]
        if prior is not None and prior() is None:
            # Recycled id(): the retired object is gone; the new object at
            # this address is a DIFFERENT connection and must register fresh.
            del _Registry_instance.retired[cid]
        _Registry_instance.leases[cid] = 0
        try:
            _Registry_instance.connections[cid] = weakref.ref(conn)
        except TypeError:
            # Object without weakref support: record the id anyway so lease
            # bookkeeping still works; no finalizer possible.
            pass
        # Deregister-only finalizer. Keep a strong ref on the finalize object
        # so it is never collected while the connection is alive.
        try:
            _FINALIZERS[cid] = weakref.finalize(conn, _finalize, cid)
        except TypeError:
            pass
        return 0


# Strong references to finalize objects (a finalize object with no strong
# ref dies with the referent's own lifetime rules; we pin ours explicitly).
_FINALIZERS: "dict[int, weakref.finalize]" = {}


def acquire_lease(conn: object) -> int:
    """Take one lease on ``conn``; returns the new lease count."""
    cid = id(conn)
    with _Registry_instance.cond:
        if cid not in _Registry_instance.leases:
            # Lease on an unregistered handle: register implicitly (connect
            # paths normally register first; this keeps bookkeeping total).
            _Registry_instance.leases[cid] = 0
            try:
                _Registry_instance.connections[cid] = weakref.ref(conn)
            except TypeError:
                pass
            try:
                _FINALIZERS[cid] = weakref.finalize(conn, _finalize, cid)
            except TypeError:
                pass
        _Registry_instance.leases[cid] += 1
        return _Registry_instance.leases[cid]


def release_lease(conn: object) -> bool:
    """Release one lease on ``conn``.

    Returns True if the lease count reached 0 (i.e. this was the final
    release and the caller should run the physical close sequence for the
    exact connection object). Release below zero is a bug: logged, clamped
    at 0; a would-be-negative release on an already-zero count still
    returns True so callers converge to closed.
    """
    cid = id(conn)
    with _Registry_instance.cond:
        count = _Registry_instance.leases.get(cid)
        if count is None:
            # Unknown id: nothing tracked (e.g. private-conn owners). Treat
            # as final release so defensive callers still clean up.
            logger.debug(
                "retirement registry: release on unregistered conn id=%d", cid
            )
            return True
        if count <= 0:
            # Underflow guard (mnemosyne#1163 review finding): a double
            # release MUST NOT be reported as the final release. Returning
            # True here could let a live peer's connection be physically
            # closed by an over-release (C1 hazard under "both release
            # orders"). Clamp at zero and report NOT final; the id stays
            # zero-leased until retire_connection() removes it.
            logger.warning(
                "retirement registry: lease underflow for conn id=%d "
                "(release below 0 is a bug; clamped at 0, not final)",
                cid,
            )
            _Registry_instance.leases[cid] = 0
            return False
        count -= 1
        _Registry_instance.leases[cid] = count
        return count == 0


def is_retired(conn: object) -> bool:
    """True if the exact connection object has been retired (final close).

    Identity is verified through the stored weakref, not the bare id: CPython
    recycles ids once an object is freed, and a fresh connection re-hitting a
    recycled id must NOT be mistaken for the retired object.
    """
    with _Registry_instance.cond:
        ref = _Registry_instance.retired.get(id(conn))
        return ref is not None and ref() is conn


def retire_connection(conn: object) -> bool:
    """Remove ``conn`` from the registry and mark it retired (object-keyed).

    Runs the identity-scoped TLS scrub (see ``scrub_tls_by_identity``) under
    the registry lock. Returns True if this call performed the retirement,
    False if the exact object was already retired.
    """
    cid = id(conn)
    with _Registry_instance.cond:
        prior = _Registry_instance.retired.get(cid)
        already = prior is not None and prior() is conn
        _Registry_instance.leases.pop(cid, None)
        _Registry_instance.connections.pop(cid, None)
        fin = _FINALIZERS.pop(cid, None)
        if fin is not None:
            try:
                fin.detach()
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            _Registry_instance.retired[cid] = weakref.ref(conn)
        except TypeError:  # pragma: no cover - defensive
            pass
        _scrub_tls_by_identity_locked(conn)
    if already:
        logger.debug(
            "retirement registry: conn id=%d was already retired", cid
        )
    return not already


def connection_retirement_state() -> dict:
    """Introspection snapshot for tests and control scripts."""
    with _Registry_instance.cond:
        return {
            "leases": dict(_Registry_instance.leases),
            "registered": tuple(sorted(_Registry_instance.connections.keys())),
            "retired": tuple(sorted(_Registry_instance.retired)),
        }


def _purge_dead_entries() -> int:
    """Drop registry entries whose weakref is dead; returns the count purged.

    Runs under the registry lock. Detaches the deregister-only finalizers
    of the purged entries.
    """
    purged = 0
    with _Registry_instance.cond:
        dead = [
            cid
            for cid, ref in _Registry_instance.connections.items()
            if ref() is None
        ]
        for cid in dead:
            _Registry_instance.leases.pop(cid, None)
            _Registry_instance.connections.pop(cid, None)
            fin = _FINALIZERS.pop(cid, None)
            if fin is not None:
                try:
                    fin.detach()
                except Exception:  # pragma: no cover - defensive
                    pass
            purged += 1
        # NOTE: retired entries hold only a weakref; pruning dead ones here
        # keeps the dict bounded under long-running churn.
        for cid in [
            cid
            for cid, ref in _Registry_instance.retired.items()
            if ref() is None
        ]:
            del _Registry_instance.retired[cid]
            purged += 1
        return purged


_TLS_MODULES = ("mnemosyne.core.beam", "mnemosyne.core.memory")


def _scrub_tls_by_identity_locked(conn: object) -> int:
    """Identity-scoped TLS scrub; MUST be called with the registry lock held.

    For the thread-local caches published by ``mnemosyne.core.beam`` and
    ``mnemosyne.core.memory``, clear any slot whose ``conn`` IS the same
    object (identity, not path). A newer generation on the same path (a
    different conn object) is never scrubbed. The calling thread's own cache
    commonly still holds the retired object at final release -- scrubbing it
    is the designed behavior (the next ``_get_connection`` on that path
    reconnects); it is logged at debug, not treated as an error.
    """
    scrubbed = 0
    for module_name in _TLS_MODULES:
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        tl = getattr(mod, "_thread_local", None)
        if tl is None:
            continue
        if getattr(tl, "conn", None) is conn:
            logger.debug(
                "retirement registry: scrubbing retired conn id=%d from "
                "%s._thread_local (identity match; next lookup reconnects)",
                id(conn),
                module_name,
            )
            tl.conn = None
            tl.db_path = None
            scrubbed += 1
    return scrubbed


def scrub_tls_by_identity(conn: object) -> int:
    """Public wrapper: scrub TLS caches holding the exact ``conn`` object.

    Runs under the registry lock. Returns the number of TLS entries
    scrubbed (at most one per module cache).
    """
    with _Registry_instance.cond:
        return _scrub_tls_by_identity_locked(conn)
