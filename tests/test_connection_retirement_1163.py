"""Connection retirement contract tests (mnemosyne#1163, card t_15fcb86f).

Covers the lease registry surfaced by ``mnemosyne.core.retirement`` and the
``close()`` implementations on ``BeamMemory`` / ``Mnemosyne``:

- C1  close() on one facade never closes a connection a peer facade leases
- C2  close() NEVER commits uncommitted caller work
- C3  retirement is keyed by connection IDENTITY, not db path
- C4  provider-side close leaves an external facade usable
- displaced-handle lifecycle, cross-thread handoff, fd regression,
  the #1228 main-thread refusal pin, registry introspection, idempotent
  close + reuse, full-graph release, and the in_transaction fence.

Hygiene rules honored here:
- pytest.skip is NEVER called inside a worker thread; every /proc-dependent
  decision is computed on the main thread (module import) before threads
  start.
- fd-count assertions are gated on ``HAVE_PROC_FD``.
- portable ownership/identity/lease assertions run everywhere.
- the conftest autouse fixture externally closes TLS caches around every
  test; the close() paths tolerate that (dead-conn handling).
"""

import gc
import os
import queue
import sqlite3
import threading
from pathlib import Path

import pytest

from mnemosyne.core import retirement
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.beam import close_thread_connection as beam_close_thread
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.memory import close_thread_connection as memory_close_thread

# Computed ON THE MAIN THREAD at import time; worker threads only read this
# boolean and never call pytest.skip themselves.
HAVE_PROC_FD = os.path.isdir("/proc/self/fd")

# Expected lease counts for one fully wired graph (see retirement.py).
BEAM_GRAPH_LEASES = 6      # 1 facade + annotations/canonical/media/
                           # episodic_graph/veracity_consolidator
MNEMOSYNE_GRAPH_LEASES = 7  # + the Mnemosyne wrapper lease


def _database_dir_fd_count(directory: Path) -> int:
    """Count fds whose target lives under ``directory`` (0 off /proc hosts)."""
    if not HAVE_PROC_FD:
        return 0
    prefix = f"{directory}{os.sep}"
    count = 0
    for fd_path in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if target.startswith(prefix):
            count += 1
    return count


def _conn_is_open(conn) -> bool:
    try:
        conn.execute("SELECT 1")
        return True
    except sqlite3.ProgrammingError:
        return False
    except sqlite3.Error:
        return False


def _leases_for(conn) -> int:
    return retirement.connection_retirement_state()["leases"].get(id(conn), 0)


def _reader_count(db_path: Path, table: str = "working_memory") -> int:
    """Fresh reader connection: committed row count for ``table``."""
    reader = sqlite3.connect(str(db_path))
    try:
        try:
            return reader.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.Error:
            return -1
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# C1: peer facade sharing
# ---------------------------------------------------------------------------

def test_close_is_noop_when_peer_facade_shares_connection(tmp_path):
    db = tmp_path / "c1.db"
    a = BeamMemory(session_id="a", db_path=db)
    b = BeamMemory(session_id="b", db_path=db)
    assert a.conn is b.conn  # same thread, same path: one shared handle
    total = _leases_for(a.conn)
    assert total == BEAM_GRAPH_LEASES * 2

    a.close()

    # The shared connection is still leased by b's graph: still open and
    # usable, no ProgrammingError.
    assert not retirement.is_retired(a.conn)
    assert _conn_is_open(b.conn)
    assert b.remember("peer still alive", source="test") is not None
    assert _leases_for(b.conn) == BEAM_GRAPH_LEASES

    # Only once the final lease holder closes does the conn retire.
    b.close()
    assert retirement.is_retired(a.conn)
    assert not _conn_is_open(b.conn)
    assert _leases_for(b.conn) == 0


# ---------------------------------------------------------------------------
# C2: close() never commits
# ---------------------------------------------------------------------------

def test_close_never_commits_uncommitted_caller_work(tmp_path):
    db = tmp_path / "c2.db"
    a = BeamMemory(session_id="a", db_path=db)
    a.remember("committed seed", source="test")  # commits normally
    committed_before = _reader_count(db)

    # Caller starts an uncommitted transaction directly on the handle.
    a.conn.execute(
        "INSERT INTO working_memory (id, content) VALUES ('uncommitted', 'no')"
    )
    assert a.conn.in_transaction
    assert _reader_count(db) == committed_before  # reader sees nothing yet

    a.close()  # defers: transaction open, never commits, never rolls back

    assert _reader_count(db) == committed_before
    assert _conn_is_open(a.conn)  # deferred, handle stays open for retry
    assert _leases_for(a.conn) == BEAM_GRAPH_LEASES

    # Force path rolls back -- still nothing committed.
    a.close(force=True)
    assert _reader_count(db) == committed_before
    assert retirement.is_retired(a.conn)
    assert not _conn_is_open(a.conn)


# ---------------------------------------------------------------------------
# C3: identity, not path
# ---------------------------------------------------------------------------

def test_close_is_identity_scoped_not_path_scoped(tmp_path):
    db = tmp_path / "c3.db"

    # Generation 1 retires fully.
    a = BeamMemory(session_id="old", db_path=db)
    old_conn = a.conn
    a.close()
    assert retirement.is_retired(old_conn)

    # Generation 2 on the SAME path gets a fresh connection object.
    b = BeamMemory(session_id="new", db_path=db)
    assert b.conn is not old_conn
    assert not retirement.is_retired(b.conn)
    assert b.remember("gen2 writes", source="test") is not None

    # Retiring the old generation again (idempotent close) must not touch
    # the newer object on the same path.
    a.close()
    assert _conn_is_open(b.conn)
    assert not retirement.is_retired(b.conn)
    assert b.remember("gen2 still fine", source="test") is not None
    b.close()

    # Literal C3 variant: the old object's sqlite handle is closed
    # EXTERNALLY (not via close()); a new generation reconnects through
    # _get_connection; the old facade's close() must not close the new one.
    db2 = tmp_path / "c3b.db"
    c = BeamMemory(session_id="old2", db_path=db2)
    c.conn.close()  # external close of the exact object only
    d = BeamMemory(session_id="new2", db_path=db2)
    assert d.conn is not c.conn  # cache-hit probe failed -> reconnect
    assert _conn_is_open(d.conn)

    c.close()  # tolerates the externally-closed handle; identity-scoped

    assert _conn_is_open(d.conn)
    assert not retirement.is_retired(d.conn)
    assert d.remember("variant survivor", source="test") is not None
    d.close()
    assert retirement.is_retired(d.conn)


# ---------------------------------------------------------------------------
# C4: provider shutdown vs external facade
# ---------------------------------------------------------------------------

def test_provider_shutdown_keeps_external_facade_usable(tmp_path):
    db = tmp_path / "c4.db"
    # Emulate the Hermes provider graph exactly: a Mnemosyne wrapper (which
    # builds its own BeamMemory + substores) plus an "external" BeamMemory
    # facade opened on the same database by someone else.
    provider = Mnemosyne(session_id="provider", db_path=db)
    external = BeamMemory(session_id="external", db_path=db)
    assert provider.conn is external.conn
    assert _leases_for(provider.conn) == (
        MNEMOSYNE_GRAPH_LEASES + BEAM_GRAPH_LEASES
    )

    provider.close()  # provider shutdown path

    # The external facade keeps a lease: handle alive and usable.
    assert not retirement.is_retired(external.conn)
    assert _conn_is_open(external.conn)
    assert external.remember("external survives shutdown", source="test")
    assert _leases_for(external.conn) == BEAM_GRAPH_LEASES

    external.close()
    assert retirement.is_retired(external.conn)
    assert _leases_for(external.conn) == 0


# ---------------------------------------------------------------------------
# Displaced banks, both release orders
# ---------------------------------------------------------------------------

def _alternating_worker(out_queue, paths):
    """Create one facade per bank switch on a LIVE worker thread."""
    for i, path in enumerate(paths):
        try:
            facade = BeamMemory(session_id=f"s{i}", db_path=path)
            out_queue.put(("facade", facade))
        except Exception as exc:  # pragma: no cover - surfaced to main
            out_queue.put(("error", exc))
    out_queue.put(("done", None))


def test_displaced_bank_release_order_a(tmp_path):
    paths = [tmp_path / "bankA.db", tmp_path / "bankB.db"]
    # Alternation displaces a fresh handle each switch: old -> new and
    # new -> old release orders must BOTH end fully retired at +0 fds.
    for order_name, order in (("old_to_new", False), ("new_to_old", True)):
        facades = []
        q = queue.Queue()
        worker = threading.Thread(
            target=_alternating_worker,
            args=(q, [paths[i % 2] for i in range(4)]),
        )
        worker.start()
        while True:
            kind, payload = q.get()
            if kind == "error":
                worker.join()
                pytest.fail(f"worker failed ({order_name}): {payload!r}")
            if kind == "done":
                break
            facades.append(payload)
        worker.join()

        conns = [f.conn for f in facades]
        assert len({id(c) for c in conns}) == len(conns)  # all displaced
        baseline = _database_dir_fd_count(tmp_path)
        assert baseline > 0  # displaced handles are open until released

        for facade in (reversed(facades) if order else facades):
            facade.close()

        for conn in conns:
            assert retirement.is_retired(conn)
            assert _leases_for(conn) == 0
            assert not _conn_is_open(conn)
        if HAVE_PROC_FD:
            assert _database_dir_fd_count(tmp_path) <= 2


# ---------------------------------------------------------------------------
# Cross-thread handoff
# ---------------------------------------------------------------------------

def test_cross_thread_handoff_retirement(tmp_path):
    db = tmp_path / "handoff.db"
    made = queue.Queue()

    def worker():
        try:
            made.put(("facade", BeamMemory(session_id="w-a", db_path=db)))
            made.put(("facade", BeamMemory(session_id="w-b", db_path=db)))
            made.put(("done", None))
        except Exception as exc:  # pragma: no cover - surfaced to main
            made.put(("error", exc))

    t = threading.Thread(target=worker)
    t.start()
    facades = []
    while True:
        kind, payload = made.get()
        if kind == "error":
            t.join()
            pytest.fail(f"worker failed: {payload!r}")
        if kind == "done":
            break
        facades.append(payload)
    t.join()

    a, b = facades  # both lease the same handle created on the worker
    assert a.conn is b.conn
    assert _leases_for(a.conn) == BEAM_GRAPH_LEASES * 2

    # Closed from the MAIN thread while the creator is gone.
    a.close()
    assert not retirement.is_retired(a.conn)
    assert b.remember("peer after cross-thread close", source="test")
    b.close()
    assert retirement.is_retired(b.conn)
    assert _leases_for(b.conn) == 0


# ---------------------------------------------------------------------------
# Timed-out worker: leases released from the main thread
# ---------------------------------------------------------------------------

def test_timed_out_worker_leases_released(tmp_path):
    db = tmp_path / "timeout.db"
    box = {}
    finish = threading.Event()
    started = threading.Event()
    worker_error = []

    def worker():
        try:
            facade = BeamMemory(session_id="slow", db_path=db)
            box["facade"] = facade
            # Leave an open transaction so the main thread's close() must
            # observe the deferral rule (never commits, never closes).
            facade.conn.execute(
                "INSERT INTO working_memory (id, content) "
                "VALUES ('pending', 'txn')"
            )
            started.set()
            finish.wait(timeout=30)
            facade.conn.rollback()
        except Exception as exc:  # pragma: no cover - recorded for main
            worker_error.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(timeout=30)
    facade = box["facade"]

    # "Timed out": join with a short timeout; the worker still holds refs.
    t.join(timeout=0.2)
    assert t.is_alive()

    facade.close()  # deferred: transaction open on the worker

    assert facade.conn.in_transaction
    assert _conn_is_open(facade.conn)
    assert _leases_for(facade.conn) == BEAM_GRAPH_LEASES
    assert _reader_count(db) == 0  # nothing committed by the deferred close

    finish.set()  # let the worker roll back and exit
    t.join(timeout=30)
    assert not t.is_alive()
    assert not worker_error

    facade.close()  # retry now lands the final release
    assert retirement.is_retired(facade.conn)
    assert _leases_for(facade.conn) == 0
    assert _reader_count(db) == 0  # rollback, never a commit
    gc.collect()


# ---------------------------------------------------------------------------
# Production fd regression: alternating sessions on a live worker
# ---------------------------------------------------------------------------

def test_fd_regression_alternating_sessions(tmp_path):
    paths = [tmp_path / "bank0.db", tmp_path / "bank1.db"]
    q = queue.Queue()
    commands = queue.Queue()
    worker_error = []

    def worker():
        while True:
            try:
                cmd = commands.get(timeout=30)
            except queue.Empty:  # pragma: no cover - harness break
                return
            if cmd[0] == "open":
                try:
                    q.put(BeamMemory(session_id=cmd[1], db_path=cmd[2]))
                except Exception as exc:  # pragma: no cover - surfaced
                    worker_error.append(exc)
                    return
            else:
                return

    t = threading.Thread(target=worker)
    t.start()
    facades = []
    for i in range(12):
        commands.put(("open", f"s{i}", paths[i % 2]))
        facades.append(q.get(timeout=30))
    commands.put(("stop", None, None))
    t.join(timeout=30)
    assert not worker_error
    assert not t.is_alive()

    conns = [f.conn for f in facades]
    assert len({id(c) for c in conns}) == 12  # every switch displaced one

    baseline = _database_dir_fd_count(tmp_path)
    assert baseline >= 12  # the leak this card fixes

    for facade in facades:  # explicit retirement by the retaining owner
        facade.close()

    for conn in conns:
        assert retirement.is_retired(conn)
        assert not _conn_is_open(conn)
    if HAVE_PROC_FD:
        assert _database_dir_fd_count(tmp_path) <= baseline and (
            _database_dir_fd_count(tmp_path) <= 2
        )


# ---------------------------------------------------------------------------
# #1228 surface pin
# ---------------------------------------------------------------------------

def test_close_thread_connection_still_refused_on_main(tmp_path):
    # The thread-bound release surface must keep refusing the main thread.
    assert beam_close_thread() is False
    assert memory_close_thread() is False
    assert beam_close_thread(str(tmp_path / "x.db")) is False
    assert memory_close_thread(str(tmp_path / "x.db")) is False


# ---------------------------------------------------------------------------
# Registry introspection
# ---------------------------------------------------------------------------

def test_lease_registry_state_introspection(tmp_path):
    db = tmp_path / "introspect.db"
    # Weakref-able handle like the production ones (_BeamConnection factory);
    # plain sqlite3.Connection rejects weakrefs and the registry then tracks
    # lease counts by id only, without a "registered" entry.
    from mnemosyne.core.beam import _BeamConnection

    conn = sqlite3.connect(str(db), factory=_BeamConnection)

    cid = id(conn)
    assert retirement.register_connection(conn) == 0
    assert retirement.register_connection(conn) == 0  # idempotent
    state = retirement.connection_retirement_state()
    assert cid in state["registered"]
    assert state["leases"][cid] == 0
    assert cid not in state["retired"]

    assert retirement.acquire_lease(conn) == 1
    assert retirement.acquire_lease(conn) == 2
    assert retirement.connection_retirement_state()["leases"][cid] == 2

    assert retirement.release_lease(conn) is False  # 2 -> 1, not final
    assert retirement.connection_retirement_state()["leases"][cid] == 1
    assert retirement.release_lease(conn) is True  # 1 -> 0, final
    assert retirement.connection_retirement_state()["leases"][cid] == 0
    # Underflow release reports NOT final (a live peer must not be closed).
    assert retirement.release_lease(conn) is False

    assert retirement.is_retired(conn) is False
    assert retirement.retire_connection(conn) is True
    assert retirement.retire_connection(conn) is False  # already retired
    state = retirement.connection_retirement_state()
    assert cid in state["retired"]
    assert cid not in state["registered"]
    assert cid not in state["leases"]
    assert retirement.is_retired(conn) is True

    conn.close()
    del conn
    gc.collect()
    purged = retirement._purge_dead_entries()
    assert purged >= 1
    state = retirement.connection_retirement_state()
    assert cid not in state["retired"]  # dead weakref pruned

    # TLS scrub is identity-scoped: no live cache holds this object.
    other = sqlite3.connect(str(db), factory=_BeamConnection)
    try:
        assert retirement.scrub_tls_by_identity(other) == 0
    finally:
        other.close()


# ---------------------------------------------------------------------------
# Idempotent close + reuse after close
# ---------------------------------------------------------------------------

def test_beam_close_idempotent_and_reuse_after_close(tmp_path):
    db = tmp_path / "reuse.db"
    a = BeamMemory(session_id="a", db_path=db)
    a.remember("before close", source="test")
    a.close()
    a.close()  # second close: no raise, no lease drift

    assert retirement.is_retired(a.conn)
    assert _leases_for(a.conn) == 0

    b = BeamMemory(session_id="b", db_path=db)  # fresh conn after scrub
    assert b.conn is not a.conn
    assert b.remember("after close", source="test") is not None
    hits = b.recall("after close", top_k=5)
    assert any("after close" in str(h) for h in hits)
    b.close()
    assert retirement.is_retired(b.conn)


# ---------------------------------------------------------------------------
# Full Mnemosyne graph release
# ---------------------------------------------------------------------------

def test_mnemosyne_close_releases_full_graph(tmp_path):
    db = tmp_path / "graph.db"
    m = Mnemosyne(session_id="m", db_path=db)
    conn = m.conn
    assert _leases_for(conn) == MNEMOSYNE_GRAPH_LEASES  # 1+1+5

    m.close()

    assert retirement.is_retired(conn)
    assert _leases_for(conn) == 0
    assert not _conn_is_open(conn)
    m.close()  # idempotent
    if HAVE_PROC_FD:
        assert _database_dir_fd_count(tmp_path) <= 2


# ---------------------------------------------------------------------------
# In-transaction fence: defer, then force rollback
# ---------------------------------------------------------------------------

def test_in_transaction_fence_defers_then_force_rolls_back(tmp_path):
    db = tmp_path / "fence.db"
    a = BeamMemory(session_id="a", db_path=db)
    a.remember("seed", source="test")
    seed = _reader_count(db)

    a.conn.execute(
        "INSERT INTO working_memory (id, content) VALUES ('tx', 'open')"
    )
    assert a.conn.in_transaction

    a.close()  # fence: defers

    assert a.conn.in_transaction          # untouched, no commit/rollback
    assert _conn_is_open(a.conn)
    assert _leases_for(a.conn) == BEAM_GRAPH_LEASES
    assert _reader_count(db) == seed      # reader saw nothing, no commit
    a.conn.rollback()                     # caller ends the transaction
    a.close()                             # retry now proceeds
    assert retirement.is_retired(a.conn)
    assert not _conn_is_open(a.conn)

    # Force variant on a second facade.
    b = BeamMemory(session_id="b", db_path=db)
    b.conn.execute(
        "INSERT INTO working_memory (id, content) VALUES ('tx2', 'open')"
    )
    b.close(force=True)  # rolls back, then closes
    assert retirement.is_retired(b.conn)
    assert not _conn_is_open(b.conn)
    assert _reader_count(db) == seed      # rollback, never a commit

    c = BeamMemory(session_id="c", db_path=db)  # reopen works
    assert c.remember("post-fence", source="test") is not None
    c.close()
