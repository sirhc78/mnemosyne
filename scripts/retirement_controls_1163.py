"""Standalone retirement controls for mnemosyne#1163 (card t_15fcb86f).

Plain ``python scripts/retirement_controls_1163.py`` - no pytest. Runs the
C1-C4 controls plus an fd-regression loop against the patched tree and
prints one ``CONTROL <name>: PASS/FAIL`` line per control. Exit code 0 iff
all pass. /proc-dependent fd assertions degrade to a PASS-with-note on
hosts without ``/proc/self/fd``. ASCII only; all databases live under
tempfile.mkdtemp() directories that are removed at exit.
"""

import atexit
import gc
import os
import queue
import shutil
import sqlite3
import sys
import tempfile
import threading

from mnemosyne.core import retirement
from mnemosyne.core.beam import BeamMemory

HAVE_PROC_FD = os.path.isdir("/proc/self/fd")
BEAM_GRAPH_LEASES = 6  # 1 facade + annotations/canonical/media/episodic/veracity

_TMP_DIRS = []


def _new_tmp_dir():
    d = tempfile.mkdtemp(prefix="retire_ctl_1163_")
    _TMP_DIRS.append(d)
    return d


@atexit.register
def _cleanup_tmp_dirs():
    for d in _TMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


def _db(name, base):
    return os.path.join(base, name)


def _conn_is_open(conn):
    try:
        conn.execute("SELECT 1")
        return True
    except sqlite3.Error:
        return False


def _leases_for(conn):
    return retirement.connection_retirement_state()["leases"].get(id(conn), 0)


def _reader_count(db_path, table="working_memory"):
    reader = sqlite3.connect(db_path)
    try:
        try:
            return reader.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
        except sqlite3.Error:
            return -1
    finally:
        reader.close()


def _db_dir_fd_count(directory):
    """Count fds whose target lives under directory (0 off /proc hosts)."""
    if not HAVE_PROC_FD:
        return 0
    prefix = directory + os.sep
    count = 0
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return 0
    for name in names:
        try:
            target = os.readlink(os.path.join("/proc/self/fd", name))
        except OSError:
            continue
        if target.startswith(prefix):
            count += 1
    return count


class Control:
    def __init__(self, name):
        self.name = name
        self.notes = []

    def note(self, msg):
        self.notes.append(msg)

    def ok(self):
        print("CONTROL %s: PASS" % self.name)
        return True

    def fail(self, reason):
        print("CONTROL %s: FAIL - %s" % (self.name, reason))
        return False


def control_c1(ctl):
    """C1: two same-path BeamMemory facades share conn; close one; peer
    stays usable."""
    base = _new_tmp_dir()
    db = _db("c1.db", base)
    a = BeamMemory(session_id="a", db_path=db)
    b = BeamMemory(session_id="b", db_path=db)
    if a.conn is not b.conn:
        return ctl.fail("facades did not share one connection")
    if _leases_for(a.conn) != BEAM_GRAPH_LEASES * 2:
        return ctl.fail(
            "expected %d leases for two graphs, got %d"
            % (BEAM_GRAPH_LEASES * 2, _leases_for(a.conn))
        )

    a.close()

    if retirement.is_retired(a.conn):
        return ctl.fail("conn retired while peer facade still leases it")
    if not _conn_is_open(b.conn):
        return ctl.fail("peer connection closed by a.close()")
    if b.remember("peer still alive", source="control") is None:
        return ctl.fail("peer facade unusable after a.close()")
    if _leases_for(b.conn) != BEAM_GRAPH_LEASES:
        return ctl.fail("lease count wrong after a.close()")

    b.close()
    if not retirement.is_retired(a.conn):
        return ctl.fail("conn not retired after final lease holder closed")
    if _conn_is_open(b.conn):
        return ctl.fail("conn still open after final close()")
    if _leases_for(b.conn) != 0:
        return ctl.fail("leases not zero after full retirement")
    return ctl.ok()


def control_c2(ctl):
    """C2: uncommitted INSERT invisible to a separate reader before AND
    after close(), including the close(force=True) path."""
    base = _new_tmp_dir()
    db = _db("c2.db", base)
    a = BeamMemory(session_id="a", db_path=db)
    a.remember("committed seed", source="control")
    committed = _reader_count(db)

    a.conn.execute(
        "INSERT INTO working_memory (id, content) VALUES ('unc', 'no')"
    )
    if not a.conn.in_transaction:
        return ctl.fail("INSERT did not open a transaction")
    if _reader_count(db) != committed:
        return ctl.fail("reader saw uncommitted work BEFORE close")

    a.close()  # fence: defers, never commits

    if _reader_count(db) != committed:
        return ctl.fail("reader saw uncommitted work AFTER deferred close")
    if not _conn_is_open(a.conn):
        return ctl.fail("deferred close physically closed the handle")
    if _leases_for(a.conn) != BEAM_GRAPH_LEASES:
        return ctl.fail("deferred close released leases")

    a.close(force=True)  # rolls back, still never commits

    if _reader_count(db) != committed:
        return ctl.fail("reader saw uncommitted work after force close")
    if not retirement.is_retired(a.conn):
        return ctl.fail("force close did not retire the connection")
    if _conn_is_open(a.conn):
        return ctl.fail("force close left the handle open")
    return ctl.ok()


def control_c3(ctl):
    """C3: closing an old facade after a NEWER facade reconnected to the
    same path must not close the newer conn. Covers the retire-then-
    reconnect variant and the literal external-close variant."""
    base = _new_tmp_dir()

    # Variant 1: retire-then-reconnect.
    db = _db("c3.db", base)
    a = BeamMemory(session_id="old", db_path=db)
    old_conn = a.conn
    a.close()
    if not retirement.is_retired(old_conn):
        return ctl.fail("variant1: old conn not retired")
    b = BeamMemory(session_id="new", db_path=db)
    if b.conn is old_conn:
        return ctl.fail("variant1: new facade reused the retired object")
    a.close()  # idempotent retry must not touch the newer object
    if not _conn_is_open(b.conn):
        return ctl.fail("variant1: old close() closed the newer conn")
    if retirement.is_retired(b.conn):
        return ctl.fail("variant1: newer conn marked retired by old close")
    if b.remember("gen2 fine", source="control") is None:
        return ctl.fail("variant1: newer facade unusable")
    b.close()

    # Variant 2: literal external close of the old object.
    db2 = _db("c3b.db", base)
    c = BeamMemory(session_id="old2", db_path=db2)
    c.conn.close()  # external sqlite close of the exact object
    d = BeamMemory(session_id="new2", db_path=db2)
    if d.conn is c.conn:
        return ctl.fail("variant2: new facade reused the externally closed conn")
    if not _conn_is_open(d.conn):
        return ctl.fail("variant2: new facade did not reconnect")
    c.close()  # must tolerate dead handle, stay identity-scoped
    if not _conn_is_open(d.conn):
        return ctl.fail("variant2: old close() closed the newer conn")
    if retirement.is_retired(d.conn):
        return ctl.fail("variant2: newer conn marked retired")
    if d.remember("survivor", source="control") is None:
        return ctl.fail("variant2: newer facade unusable")
    d.close()
    if not retirement.is_retired(d.conn):
        return ctl.fail("variant2: final close did not retire")
    return ctl.ok()


def control_c4(ctl):
    """C4: Mnemosyne wrapper as provider + retained BeamMemory external
    facade on the same path; wrapper.close() leaves the external facade
    usable."""
    from mnemosyne.core.memory import Mnemosyne

    base = _new_tmp_dir()
    db = _db("c4.db", base)
    provider = Mnemosyne(session_id="provider", db_path=db)
    external = BeamMemory(session_id="external", db_path=db)
    if provider.conn is not external.conn:
        return ctl.fail("provider and external did not share the connection")
    expected = BEAM_GRAPH_LEASES * 2 + 1  # beam graphs + wrapper lease
    if _leases_for(provider.conn) != expected:
        return ctl.fail(
            "expected %d leases, got %d" % (expected, _leases_for(provider.conn))
        )

    provider.close()  # provider shutdown path

    if retirement.is_retired(external.conn):
        return ctl.fail("conn retired while external facade still leases it")
    if not _conn_is_open(external.conn):
        return ctl.fail("provider.close() closed the external conn")
    if external.remember("external survives", source="control") is None:
        return ctl.fail("external facade unusable after provider shutdown")
    if _leases_for(external.conn) != BEAM_GRAPH_LEASES:
        return ctl.fail("external lease count wrong after provider close")
    external.close()
    if not retirement.is_retired(external.conn):
        return ctl.fail("conn not retired after final holder closed")
    return ctl.ok()


def control_fd_regression(ctl):
    """12 alternating-path BeamMemory sessions on one LIVE worker thread;
    facades retained, explicit close() each; /proc fd delta <= 2."""
    base = _new_tmp_dir()
    paths = [_db("bank0.db", base), _db("bank1.db", base)]
    q = queue.Queue()
    commands = queue.Queue()
    worker_error = []

    def worker():
        while True:
            try:
                cmd = commands.get(timeout=60)
            except queue.Empty:
                return
            if cmd[0] == "open":
                try:
                    q.put(BeamMemory(session_id=cmd[1], db_path=cmd[2]))
                except Exception as exc:  # pragma: no cover
                    worker_error.append(exc)
                    return
            else:
                return

    t = threading.Thread(target=worker)
    t.start()
    facades = []
    try:
        for i in range(12):
            commands.put(("open", "s%d" % i, paths[i % 2]))
            facades.append(q.get(timeout=60))
        commands.put(("stop", None, None))
        t.join(timeout=60)
        if worker_error:
            return ctl.fail("worker failed: %r" % worker_error[0])
        if t.is_alive():
            return ctl.fail("worker thread did not exit")
        if len({id(f.conn) for f in facades}) != 12:
            return ctl.fail("path switching did not displace connections")

        if not HAVE_PROC_FD:
            ctl.note("fd check skipped (/proc/self/fd unavailable)")
        else:
            baseline = _db_dir_fd_count(base)
            if baseline < 12:
                return ctl.fail(
                    "expected >= 12 leaked fds pre-fix, saw %d" % baseline
                )

        for facade in facades:  # explicit retirement by retaining owner
            facade.close()
        for facade in facades:
            if not retirement.is_retired(facade.conn):
                return ctl.fail("conn not retired after explicit close()")
            if _conn_is_open(facade.conn):
                return ctl.fail("conn still open after close()")
        gc.collect()

        if HAVE_PROC_FD:
            after = _db_dir_fd_count(base)
            if after > 2:
                return ctl.fail(
                    "fd leak: %d fds remain under db dir (limit 2)" % after
                )
            ctl.note("fd delta after close: %d -> %d" % (baseline, after))
        return ctl.ok()
    finally:
        commands.put(("stop", None, None))
        t.join(timeout=30)


CONTROLS = [
    ("C1", control_c1),
    ("C2", control_c2),
    ("C3", control_c3),
    ("C4", control_c4),
    ("FD-REGRESSION", control_fd_regression),
]


def main():
    all_ok = True
    for name, fn in CONTROLS:
        ctl = Control(name)
        try:
            passed = fn(ctl)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            import traceback

            traceback.print_exc()
            passed = ctl.fail("raised %r" % (exc,))
        for line in ctl.notes:
            print("  note: %s" % line)
        all_ok = all_ok and passed
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
