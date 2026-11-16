"""Connection-retirement contract tests (mnemosyne#1163).

Pins the maintainer control matrix C1-C4 plus displaced aliases, cross-thread
handoff, timed-out workers (fenced and idle), gc parity and /proc-gated fd
regressions. fd-dependent skips are decided at COLLECTION time on the main
thread; no test ever calls pytest.skip() from inside a spawned worker.
"""

import gc
import os
import sqlite3
import threading

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core import memory as memory_module
from mnemosyne.core.beam import BeamMemory, MemoryTransactionStateError
from mnemosyne.core.memory import Mnemosyne

# fd assertions read /proc/self/fd; decide availability once, on the main
# thread, at collection time (a worker calling pytest.skip() would error).
_PROC_FD_AVAILABLE = os.path.isdir("/proc/self/fd")


def _usable(conn) -> bool:
    try:
        conn.execute("SELECT 1").fetchone()
        return True
    except sqlite3.ProgrammingError:
        return False


def _scrub_tls():
    """Reset both modules' TLS caches the way conftest does between tests."""
    for mod in (beam_module, memory_module):
        tl = getattr(mod, "_thread_local", None)
        if tl is not None and getattr(tl, "conn", None) is not None:
            try:
                tl.conn.close()
            except Exception:
                pass
            tl.conn = None
            tl.db_path = None


@pytest.fixture(autouse=True)
def _tls_hygiene():
    yield
    _scrub_tls()


# --------------------------------------------------------------- C1 ----
class TestC1SameBankPeers:
    def test_peers_share_handle_and_each_close_is_peer_safe(self, tmp_path):
        db = tmp_path / "c1.db"
        first = BeamMemory(session_id="a", db_path=db)
        peer = BeamMemory(session_id="b", db_path=db)
        assert first.conn is peer.conn
        handle = first.conn
        first.close()
        assert first.is_closed
        assert not peer.is_closed
        assert _usable(peer.conn)
        peer.close()
        assert not _usable(handle)

    def test_close_is_idempotent(self, tmp_path):
        f = BeamMemory(session_id="i", db_path=tmp_path / "c1i.db")
        f.close()
        f.close()  # never raises
        f.close(force=True)  # never raises either

    def test_mnemosyne_close_delegates_one_lease(self, tmp_path):
        db = tmp_path / "c1m.db"
        m = Mnemosyne(session_id="m", db_path=db)
        peer = BeamMemory(session_id="mp", db_path=db)
        assert m.beam.conn is peer.conn
        m.close()
        assert m.is_closed
        assert _usable(peer.conn)
        peer.close()


# --------------------------------------------------------------- C2 ----
class TestC2NoImplicitCommit:
    def test_close_refuses_on_open_transaction(self, tmp_path):
        f = BeamMemory(session_id="c2a", db_path=tmp_path / "a.db")
        f.conn.execute(
            "INSERT INTO scratchpad (id, content) VALUES ('c2','x')"
        )
        with pytest.raises(MemoryTransactionStateError):
            f.close()
        assert not f.is_closed
        assert _usable(f.conn)

    def test_refusal_changes_nothing_and_preserves_authority(self, tmp_path):
        db = tmp_path / "c2b.db"
        f = BeamMemory(session_id="c2b", db_path=db)
        f.conn.execute(
            "INSERT INTO scratchpad (id, content) VALUES ('c2','x')"
        )
        reader = sqlite3.connect(str(db), check_same_thread=False)
        try:
            with pytest.raises(MemoryTransactionStateError):
                f.close()
            assert reader.execute(
                "SELECT COUNT(*) FROM scratchpad WHERE id='c2'"
            ).fetchone()[0] == 0
            f.conn.commit()  # caller keeps commit authority after refusal
            assert reader.execute(
                "SELECT COUNT(*) FROM scratchpad WHERE id='c2'"
            ).fetchone()[0] == 1
        finally:
            reader.close()
            f.close(force=True)

    def test_force_discards_open_transaction(self, tmp_path):
        db = tmp_path / "c2c.db"
        f = BeamMemory(session_id="c2c", db_path=db)
        f.conn.execute(
            "INSERT INTO scratchpad (id, content) VALUES ('c2','x')"
        )
        f.close(force=True)
        assert f.is_closed
        reader = sqlite3.connect(str(db))
        try:
            assert reader.execute(
                "SELECT COUNT(*) FROM scratchpad WHERE id='c2'"
            ).fetchone()[0] == 0  # rolled back, not committed
        finally:
            reader.close()


# --------------------------------------------------------------- C3 ----
class TestC3IdentityScopedScrub:
    def test_older_close_does_not_hit_newer_generation(self, tmp_path):
        db = tmp_path / "c3.db"
        old = BeamMemory(session_id="old", db_path=db)
        handle_old = old.conn
        BeamMemory(session_id="sw", db_path=tmp_path / "sw.db")
        new = BeamMemory(session_id="new", db_path=db)
        handle_new = new.conn
        assert handle_new is not handle_old
        old.close()
        assert _usable(handle_new)
        new.close()
        assert not _usable(handle_new)

    def test_newer_close_does_not_hit_older_displaced_handle(self, tmp_path):
        db = tmp_path / "c3b.db"
        old = BeamMemory(session_id="old", db_path=db)
        handle_old = old.conn
        switch = BeamMemory(session_id="sw", db_path=tmp_path / "sw.db")
        new = BeamMemory(session_id="new", db_path=db)
        new.close()
        assert _usable(handle_old)
        old.close()
        switch.close()

    def test_tls_scrub_is_identity_scoped_both_modules(self, tmp_path):
        db = tmp_path / "c3t.db"
        old = BeamMemory(session_id="old", db_path=db)
        handle_old = old.conn
        BeamMemory(session_id="sw", db_path=tmp_path / "sw.db")
        new = BeamMemory(session_id="new", db_path=db)
        new.close()  # scrubs TLS only if TLS holds the new object
        # old generation's handle still cached... it was displaced, so the
        # TLS slot now holds the NEW connection; after new.close() the slot
        # is scrubbed; verify neither module pins the dead object.
        for mod in (beam_module, memory_module):
            tl = getattr(mod, "_thread_local", None)
            if tl is None:
                continue
            assert getattr(tl, "conn", None) is not handle_old
        old.close()


# --------------------------------------------------------------- C4 ----
class TestC4ExternalAndSubstores:
    def test_external_facade_survives_provider_close(self, tmp_path):
        db = tmp_path / "c4.db"
        provider = BeamMemory(session_id="prov", db_path=db)
        external = BeamMemory(session_id="ext", db_path=db)
        assert provider.conn is external.conn
        provider.close()
        assert _usable(external.conn)
        external.close()

    def test_substore_aliases_ride_facade_lease(self, tmp_path):
        f = BeamMemory(session_id="s", db_path=tmp_path / "s.db")
        assert f.episodic_graph is None or f.episodic_graph.conn is f.conn
        assert (
            f.veracity_consolidator is None
            or f.veracity_consolidator.conn is f.conn
        )
        peer = BeamMemory(session_id="sp", db_path=tmp_path / "s.db")
        f.close()
        assert _usable(peer.conn)  # substore aliases did not over-release
        peer.close()

    def test_episodic_close_guards_shared_handle(self, tmp_path):
        f = BeamMemory(session_id="e", db_path=tmp_path / "e.db")
        peer = BeamMemory(session_id="ep", db_path=tmp_path / "e.db")
        if f.episodic_graph is not None:
            f.episodic_graph.close()  # must be a no-op on a shared handle
            assert _usable(f.conn)
            assert _usable(peer.conn)
        peer.close()
        f.close()


# ----------------------------------------------------- displaced / gc ----
class TestDisplacedAliasesAndGcParity:
    def test_displaced_facade_close_releases_fd(self, tmp_path):
        if not _PROC_FD_AVAILABLE:
            pytest.skip("open-descriptor assertions require /proc")
        gc.collect()
        base = len(os.listdir("/proc/self/fd"))
        for i in range(12):
            f = BeamMemory(
                session_id=f"L{i}",
                db_path=(
                    tmp_path / "alt-a.db" if i % 2 == 0 else tmp_path / "alt-b.db"
                ),
            )
            f.remember("n" + str(i))
            f.close()
        gc.collect()
        after = len(os.listdir("/proc/self/fd"))
        assert after <= base + 4, f"+{after - base} fds after retirement"

    def test_abandoned_facade_releases_by_gc(self, tmp_path):
        # Weak lease: an abandoned facade must not pin its handle's entry.
        # After gc the entry for THIS handle holds no live lease (gc parity
        # with unmodified main, where collect_connection_cycles() is the
        # backstop), and the drained entry is pruned on the next
        # registration so the registry stays bounded by live handles.
        # Scoped to this handle's entry: the registry is process-wide and
        # other suites may hold live facades.
        from mnemosyne.core.beam import _CONNECTION_LEASES

        f = BeamMemory(session_id="g", db_path=tmp_path / "gc.db")
        key = id(f.conn)
        assert _CONNECTION_LEASES.get(key)
        del f
        gc.collect()
        assert not _CONNECTION_LEASES.get(key)

    def test_abandoned_facade_gc_parity_no_stale_leases(self, tmp_path):
        # Repeated same-path create/retire must not accumulate registry
        # entries for the handle (each close is the last lease on it).
        from mnemosyne.core.beam import _CONNECTION_LEASES

        first = BeamMemory(session_id="g0", db_path=tmp_path / "gc2.db")
        key = id(first.conn)
        first.close()
        for i in range(1, 5):
            BeamMemory(session_id=f"g{i}", db_path=tmp_path / "gc2.db").close()
        gc.collect()
        assert not _CONNECTION_LEASES.get(key)


class TestRegistryCoalescing:
    def test_peers_coalesce_to_one_entry(self, tmp_path):
        from mnemosyne.core.beam import _CONNECTION_LEASES

        db = tmp_path / "coal.db"
        facades = [BeamMemory(session_id=f"p{i}", db_path=db) for i in range(4)]
        key = id(facades[0].conn)
        leases = _CONNECTION_LEASES.get(key)
        assert leases is not None
        assert len(leases) == 4  # one lease per facade, coalesced to one entry
        for f in facades[:-1]:
            f.close()
            assert _CONNECTION_LEASES.get(key)  # entry lives until the last
        facades[-1].close()
        assert not _CONNECTION_LEASES.get(key)

    def test_displaced_facade_substore_alias_keeps_working(self, tmp_path):
        # The displaced facade (and its sub-store aliases) keep operating on
        # their own handle after a newer generation displaced the TLS slot.
        db = tmp_path / "disp.db"
        old = BeamMemory(session_id="old", db_path=db)
        BeamMemory(session_id="sw", db_path=tmp_path / "sw.db")
        new = BeamMemory(session_id="new", db_path=db)
        assert old.conn is not new.conn
        old.remember("from-the-displaced-facade")  # still usable, own handle
        new.remember("from-the-new-generation")
        assert old.recall("displaced-facade"), "displaced facade lost its write"
        new.close()
        old.remember("still-alive-after-peer-retired")
        old.close()

    def test_mnemosyne_close_idempotent_and_force_delegates(self, tmp_path):
        db = tmp_path / "mi.db"
        m = Mnemosyne(session_id="mi", db_path=db)
        peer = BeamMemory(session_id="mip", db_path=db)
        m.close()
        m.close()  # idempotent through the delegation too
        assert m.is_closed
        assert _usable(peer.conn)
        m2 = Mnemosyne(session_id="mi2", db_path=tmp_path / "mi2.db")
        m2.conn.execute(
            "INSERT INTO scratchpad (id, content) VALUES ('mi2','x')"
        )
        m2.close(force=True)  # force reaches the beam facade
        assert m2.is_closed


# --------------------------------------------------------- cross-thread ----
class TestCrossThread:
    def test_close_from_another_thread(self, tmp_path):
        db = tmp_path / "x.db"
        f = BeamMemory(session_id="main", db_path=db)
        handle = f.conn
        errors = []

        def retire():
            try:
                f.close()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t = threading.Thread(target=retire)
        t.start()
        t.join()
        assert not errors
        assert f.is_closed
        assert not _usable(handle)

    def test_timed_out_worker_leaves_fenced_transaction(self, tmp_path):
        db = tmp_path / "w.db"
        owner = BeamMemory(session_id="w-owner", db_path=db)
        survivor = BeamMemory(session_id="w-surv", db_path=db)
        done = threading.Event()

        def worker():
            owner.conn.execute(
                "INSERT INTO scratchpad (id, content) VALUES ('w','x')"
            )
            done.set()
            # "dies" mid-transaction: never commits, never rolls back

        t = threading.Thread(target=worker)
        t.start()
        done.wait(10)
        t.join()
        # Supervisor close() must refuse: the worker's transaction is open.
        with pytest.raises(MemoryTransactionStateError):
            owner.close()
        assert not owner.is_closed
        assert _usable(survivor.conn)
        owner.close(force=True)  # explicit discard of the open transaction
        assert owner.is_closed
        assert _usable(survivor.conn)  # peer lease keeps the shared handle
        survivor.close()
        assert not _usable(survivor.conn)

    def test_timed_out_worker_idle_then_close(self, tmp_path):
        db = tmp_path / "w2.db"
        owner = BeamMemory(session_id="w2", db_path=db)
        owner.conn.execute(
            "INSERT INTO scratchpad (id, content) VALUES ('w2','x')"
        )
        with pytest.raises(MemoryTransactionStateError):
            owner.close()
        owner.conn.rollback()
        owner.close()  # now retires cleanly
        assert owner.is_closed


# --------------------------------------------------------------- fd gate ----
@pytest.mark.skipif(
    not _PROC_FD_AVAILABLE, reason="open-descriptor assertions require /proc"
)
class TestFdRegression:
    def test_alternating_sessions_stay_flat(self, tmp_path):
        bank_a = tmp_path / "flat-a.db"
        bank_b = tmp_path / "flat-b.db"
        gc.collect()
        base = len(os.listdir("/proc/self/fd"))
        for i in range(12):
            f = BeamMemory(
                session_id=f"F{i}",
                db_path=bank_a if i % 2 == 0 else bank_b,
            )
            f.remember("n" + str(i))
            f.close()
        gc.collect()
        after = len(os.listdir("/proc/self/fd"))
        assert after <= base + 4


# ------------------------------------------------------------- controls ----
def test_controls_script_matrix():
    """The standalone controls script passes its whole matrix."""
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "scripts", "retirement_controls_1163.py"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "MNEMOSYNE_NO_EMBEDDINGS": "1"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for line in (
        "C1 same-bank peers",
        "C2 no implicit commit",
        "C3 newer-generation scoping",
        "C4 external facade + sub-store close",
        "OVERALL: PASS",
    ):
        assert line in proc.stdout, f"missing: {line}"
