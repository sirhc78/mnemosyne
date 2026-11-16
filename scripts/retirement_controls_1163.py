#!/usr/bin/env python3
"""Standalone controls C1-C4 for the connection-retirement contract (issue #1163).

Runs the four maintainer controls against the patched tree and prints
PASS/FAIL per control, plus an fd-gated leak regression where /proc is
available. Exit code 0 iff every control passes.

    python scripts/retirement_controls_1163.py
"""
import gc
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mnemosyne.core.beam import (  # noqa: E402
    BeamMemory,
    MemoryTransactionStateError,
)
from mnemosyne.core.memory import Mnemosyne  # noqa: E402


def _usable(conn) -> bool:
    try:
        conn.execute("SELECT 1").fetchone()
        return True
    except sqlite3.ProgrammingError:
        return False


def _control_c1(root: Path) -> str:
    """Retained same-bank peers share one handle; each close is peer-safe."""
    db = root / "c1.db"
    first = BeamMemory(session_id="c1-a", db_path=db)
    peer = BeamMemory(session_id="c1-b", db_path=db)
    if first.conn is not peer.conn:
        return "FAIL: same-path facades do not share the handle"
    handle = first.conn
    first.close()
    if not _usable(peer.conn):
        return "FAIL: C1 peer broke after first close()"
    peer.close()
    if _usable(handle):
        return "FAIL: last lease did not release the handle"
    # Mnemosyne facade delegates to the same single lease.
    m = Mnemosyne(session_id="c1-m", db_path=db)
    m_peer = BeamMemory(session_id="c1-mp", db_path=db)
    m.close()
    if not _usable(m_peer.conn):
        return "FAIL: Mnemosyne.close() broke a same-handle peer"
    m_peer.close()
    return "PASS"


def _control_c2(root: Path) -> str:
    """close() never commits; open transactions refuse and stay invisible."""
    db = root / "c2.db"
    f = BeamMemory(session_id="c2", db_path=db)
    f.conn.execute("INSERT INTO scratchpad (id, content) VALUES ('c2','x')")
    reader = sqlite3.connect(str(db), check_same_thread=False)
    try:
        if reader.execute(
            "SELECT COUNT(*) FROM scratchpad WHERE id='c2'"
        ).fetchone()[0] != 0:
            return "FAIL: setup visible before close (journal leak?)"
        try:
            f.close()
        except MemoryTransactionStateError:
            pass
        else:
            return "FAIL: close() did not refuse on an open transaction"
        if reader.execute(
            "SELECT COUNT(*) FROM scratchpad WHERE id='c2'"
        ).fetchone()[0] != 0:
            return "FAIL: C2 uncommitted INSERT became visible after close()"
        if f.is_closed:
            return "FAIL: refused close() still retired the facade"
        # Caller keeps authority: deliberate commit then clean retire.
        f.conn.commit()
        if reader.execute(
            "SELECT COUNT(*) FROM scratchpad WHERE id='c2'"
        ).fetchone()[0] != 1:
            return "FAIL: deliberate commit after refusal did not persist"
    finally:
        reader.close()
        f.close()
    return "PASS"


def _control_c3(root: Path) -> str:
    """Identity-scoped cleanup: newer same-path generations untouched."""
    db = root / "c3.db"
    old = BeamMemory(session_id="c3-old", db_path=db)
    handle_old = old.conn
    BeamMemory(session_id="c3-switch", db_path=root / "c3-switch.db")
    new = BeamMemory(session_id="c3-new", db_path=db)  # newer generation
    handle_new = new.conn
    if handle_new is handle_old:
        return "FAIL: setup did not produce two generations"
    old.close()
    if not _usable(handle_new):
        return "FAIL: C3 newer generation hit by old facade's close()"
    new.close()
    return "PASS"


def _control_c3b(root: Path) -> str:
    """Release order: new-first must not hit the older displaced handle."""
    db = root / "c3b.db"
    old = BeamMemory(session_id="old", db_path=db)
    handle_old = old.conn
    switch = BeamMemory(session_id="sw", db_path=root / "c3b-sw.db")
    new = BeamMemory(session_id="new", db_path=db)
    new.close()
    if not _usable(handle_old):
        return "FAIL: older displaced handle died with new-first order"
    old.close()
    switch.close()
    return "PASS"


def _control_c4(root: Path) -> str:
    """External facade on the same handle survives provider close()."""
    db = root / "c4.db"
    provider = BeamMemory(session_id="prov", db_path=db)
    external = BeamMemory(session_id="ext", db_path=db)
    if provider.conn is not external.conn:
        return "FAIL: provider/external did not share one handle"
    provider.close()
    if not _usable(external.conn):
        return "FAIL: C4 external facade died with provider close()"
    external.close()
    # Sub-store close() must not kill the shared handle either.
    f = BeamMemory(session_id="s", db_path=root / "c4b.db")
    if f.episodic_graph is not None:
        f.episodic_graph.close()
        if not _usable(f.conn):
            return "FAIL: EpisodicGraph.close() killed the shared handle"
    f.close()
    return "PASS"


def _leak_probe(root: Path) -> str:
    """fd regression: alternating banks + retirement stays flat (needs /proc)."""
    if not Path("/proc/self/fd").exists():
        return "SKIP (no /proc)"
    gc.collect()
    bank_a = root / "alt-a.db"
    bank_b = root / "alt-b.db"
    base = len(os.listdir("/proc/self/fd"))
    for i in range(12):
        f = BeamMemory(session_id=f"L{i}", db_path=bank_a if i % 2 == 0 else bank_b)
        f.remember("n" + str(i))
        f.close()
    gc.collect()
    after = len(os.listdir("/proc/self/fd"))
    if after > base + 4:
        return "FAIL: +%d fds after 12 retired alternating sessions" % (after - base)
    return "PASS (+%d fds)" % (after - base)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="retirement-controls-"))
    os.environ.setdefault("MNEMOSYNE_NO_EMBEDDINGS", "1")
    results = [
        ("C1 same-bank peers", _control_c1(root)),
        ("C2 no implicit commit", _control_c2(root)),
        ("C3 newer-generation scoping", _control_c3(root)),
        ("C3b release order (new-first)", _control_c3b(root)),
        ("C4 external facade + sub-store close", _control_c4(root)),
        ("fd leak regression", _leak_probe(root)),
    ]
    failed = 0
    for name, outcome in results:
        if not (outcome == "PASS" or outcome.startswith(("PASS ", "SKIP"))):
            failed += 1
        print("%-34s %s" % (name, outcome))
    print("OVERALL:", "PASS" if failed == 0 else "FAIL (%d)" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
