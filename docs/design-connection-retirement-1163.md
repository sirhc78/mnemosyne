# Connection retirement contract (mnemosyne#1163)

Status: proposal for review under MAINTAINERS.md (core architecture - AJ).
Branch: `fix/1163-connection-retirement`.

## Problem

A long-running gateway accumulates SQLite fds monotonically: the single-slot
TLS cache displaces handles on path switches without closing them, facades
(`Mnemosyne`/`BeamMemory` + sub-stores) pin the displaced handles forever,
and no `close()` exists in the core object model. `close_thread_connection`
(#1228) is thread-bound and path-matched; it cannot release a handle from
another thread and cannot distinguish two generations on one path.

## Contract

**Who retires each instance.** Each `BeamMemory`/`Mnemosyne` facade owns one
lease on the exact connection object it was built on; leases live in a
process-wide weak registry keyed by `id(conn)` (weakrefs to facades). The
facade's owner - the application, provider wrapper, or worker loop - calls
`close()` when it is done. `Mnemosyne.close()` delegates to its beam facade
(one lease per facade pair). Sub-store aliases ride their facade's lease.
External facades on the same handle register their own lease.

**What fences in-flight work.** `close()` refuses with
`MemoryTransactionStateError` while the connection reports
`in_transaction` - including a transaction left mid-flight by a dead or
timed-out worker. Disposal never commits (would persist caller-owned work)
and never silently discards it. `close(force=True)` is the explicit
discard: rollback, then retire. After a refusal nothing has changed; the
caller keeps full commit/rollback authority and may retry close().

**When the final owner releases.** `close()` drops the retiring facade's
lease; if another live lease remains (same-path peer, external facade on
the same object), the handle stays open and usable for them. The last lease
out closes the handle - never before.

**How external aliases participate.** They are lease holders like anyone
else. An external facade/sub-store that shares the handle registers
(`_lease_register`) and thereby blocks retirement until it retires or
becomes unreachable (weak). No central authority needs to know every alias:
whoever releases checks the registry, not the path.

**Identity scoping.** Cleanup is strictly object-identity-scoped: TLS caches
are scrubbed only where they hold this exact object; a newer generation
reconnected to the same path is a different object and is untouched. This
is what C3 pins, and why no path-based release (the C3 failure mode) exists
in the contract.

**Leak fix mechanics.** Alternating banks still displace handles (unchanged
semantics; pools and close-on-displacement are explicitly out of scope), but
the displaced handle's facade holds the only live lease on it, so its
`close()` - or the facade becoming unreachable (weak lease; gc parity with
unmodified main, `collect_connection_cycles()` remains the backstop) -
releases the fd. Same-path sessions stay flat (TLS reuse untouched).

**Out of scope (maintainer-set boundaries).** No bounded pool, no automatic
close-on-displacement of live displaced handles, no changes to
`connection_release.py` (#1228 surface).

## Controls

`scripts/retirement_controls_1163.py` runs C1-C4 against the patched tree:

```
C1 same-bank peers                 PASS
C2 no implicit commit              PASS
C3 newer-generation scoping        PASS
C3b release order (new-first)      PASS
C4 external facade + sub-store close PASS
fd leak regression                 PASS (+0 fds)
```

`tests/test_connection_retirement_1163.py` pins the same matrix plus
displaced-bank aliases, cross-thread handoff, timed-out worker (fenced and
idle variants), gc parity, and /proc-gated fd counts (skips decided at
collection time on the main thread, never inside a worker).
