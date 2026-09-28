"""Bounded pool of keep-alive TCP connections keyed by (host, port).

Design decisions
----------------
- ``acquire`` is synchronous and blocks up to ``timeout`` seconds when the
  per-(host, port) cap is reached. Blocking is implemented with a ``threading.Condition``
  so that ``release`` can wake exactly one waiter without thundering-herd wakeups.
- Idle eviction is driven by an explicit ``sweep_idle`` call rather than a background
  thread. A background thread would own a daemon that outlives any reasonable unit test
  and makes determinism impossible; callers that want automatic sweeping can drive it
  from their own timer.
- Time is injected via ``clock`` so tests never touch the wall clock. All comparisons
  are against integers returned by that callable; no floats are stored or compared.
- Connections are raw ``socket.socket`` objects. We do not wrap them in a custom type
  because callers need the real socket API (``recv``, ``sendall``, ``fileno``, …) and
  a wrapper would either leak the socket anyway or grow into a reimplementation of
  ``socket``.
- A connection that is found to be closed by the OS (``shutdown`` raises) is dropped
  during ``acquire`` rather than handed to the caller.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple


Addr = Tuple[str, int]


class PoolError(Exception):
    """Raised for pool-level failures: exhaustion, shutdown, bad arguments."""


class PooledConnection:
    """A socket checked out of the pool.

    Wraps the raw ``socket.socket`` so the pool can track whether it has been
    returned. Use ``.socket`` to access the underlying socket, or treat the
    instance as a context manager to release on exit.
    """

    __slots__ = ("socket", "host", "port", "pool", "_released")

    def __init__(self, sock: socket.socket, host: str, port: int, pool: "ConnectionPool") -> None:
        self.socket = sock
        self.host = host
        self.port = port
        self.pool = pool
        self._released = False

    def release(self, *, broken: bool = False) -> None:
        """Return the connection to the pool.

        Pass ``broken=True`` when the socket is in an unknown state (e.g. a
        previous ``recv`` raised); the pool will close it instead of recycling.
        Calling ``release`` twice is a no-op on the second call.
        """
        if self._released:
            return
        self._released = True
        self.pool._release(self, broken=broken)

    def close(self) -> None:
        """Release and close the underlying socket unconditionally."""
        self.release(broken=True)

    def __enter__(self) -> "PooledConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If the body raised, assume the connection may be in a bad state.
        self.release(broken=exc is not None)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PooledConnection(host={self.host!r}, port={self.port}, released={self._released})"


class _Bucket:
    """Per-(host, port) state: idle connections and a count of in-use ones."""

    __slots__ = ("idle", "in_use", "cond")

    def __init__(self) -> None:
        # Idle entries are (last_used_time, socket). Newest at the end so we
        # can pop() the most-recently-used and evict from index 0.
        self.idle: List[Tuple[int, socket.socket]] = []
        self.in_use: int = 0
        self.cond = threading.Condition()


class ConnectionPool:
    """A bounded pool of keep-alive TCP connections.

    Parameters
    ----------
    max_per_host:
        Hard cap on live (idle + in-use) connections to a single (host, port).
    max_total:
        Hard cap on live connections across all hosts. ``acquire`` blocks if
        hitting this cap would be required to open a new connection and no
        idle connection can be evicted to make room.
    idle_timeout:
        Connections idle for longer than this many seconds are eligible for
        eviction by ``sweep_idle``.
    connect_timeout:
        Per-call ``socket.connect`` timeout in seconds.
    clock:
        Callable returning the current time as a number. Defaults to
        ``time.monotonic``. Inject a fake in tests.
    """

    def __init__(
        self,
        *,
        max_per_host: int = 8,
        max_total: int = 64,
        idle_timeout: float = 60.0,
        connect_timeout: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_per_host < 1:
            raise ValueError("max_per_host must be >= 1")
        if max_total < 1:
            raise ValueError("max_total must be >= 1")
        if max_total < max_per_host:
            raise ValueError("max_total must be >= max_per_host")
        if idle_timeout < 0:
            raise ValueError("idle_timeout must be >= 0")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be > 0")

        self._max_per_host = max_per_host
        self._max_total = max_total
        self._idle_timeout = idle_timeout
        self._connect_timeout = connect_timeout
        self._clock = clock

        self._buckets: Dict[Addr, _Bucket] = {}
        self._global_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def acquire(self, host: str, port: int, *, timeout: Optional[float] = None) -> PooledConnection:
        """Get a connection to ``host:port``, blocking up to ``timeout`` seconds.

        Raises ``PoolError`` if the pool is shut down, the timeout elapses, or
        the arguments are invalid. Raises ``OSError`` if a new connection
        attempt fails.
        """
        if not isinstance(host, str) or not host:
            raise PoolError("host must be a non-empty string")
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise PoolError("port must be an int in 1..65535")
        if self._closed:
            raise PoolError("pool is closed")

        deadline = None if timeout is None else self._clock() + timeout
        bucket = self._get_bucket((host, port))

        with bucket.cond:
            while True:
                if self._closed:
                    raise PoolError("pool is closed")

                # Reuse an idle connection if one is healthy.
                sock = self._take_idle(bucket)
                if sock is not None:
                    bucket.in_use += 1
                    return PooledConnection(sock, host, port, self)

                # Open a new one if we have headroom under both caps.
                if self._can_open(bucket):
                    bucket.in_use += 1
                    # Drop the lock while connecting so we don't block other
                    # hosts. We've already reserved the slot by bumping in_use.
                    break

                # Otherwise wait for a release.
                if deadline is None:
                    bucket.cond.wait()
                else:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise PoolError("timed out waiting for a connection")
                    bucket.cond.wait(timeout=remaining)

        # Outside the lock: actually connect. On failure, undo the reservation.
        try:
            sock = self._open(host, port)
        except BaseException:
            with bucket.cond:
                bucket.in_use -= 1
                bucket.cond.notify_all()
            raise
        return PooledConnection(sock, host, port, self)

    def release(self, conn: PooledConnection, *, broken: bool = False) -> None:
        """Return ``conn`` to the pool.

        Prefer ``conn.release()`` or the context manager; this method exists
        for callers holding a bare ``PooledConnection`` reference.
        """
        conn.release(broken=broken)

    def sweep_idle(self) -> int:
        """Close idle connections that have exceeded ``idle_timeout``.

        Returns the number of connections closed. Safe to call on a closed
        pool (no-op, returns 0).
        """
        if self._idle_timeout == 0:
            # Every idle connection is immediately eligible. We still iterate
            # so the return count is correct.
            pass
        now = self._clock()
        closed = 0
        with self._global_lock:
            if self._closed:
                return 0
            for bucket in self._buckets.values():
                with bucket.cond:
                    kept: List[Tuple[int, socket.socket]] = []
                    for ts, sock in bucket.idle:
                        if (now - ts) >= self._idle_timeout:
                            self._safe_close(sock)
                            closed += 1
                        else:
                            kept.append((ts, sock))
                    bucket.idle = kept
                    # Eviction may have freed slots; wake waiters.
                    bucket.cond.notify_all()
        return closed

    def close(self) -> None:
        """Close all idle connections and mark the pool as shut down.

        Connections currently checked out are not closed here; their
        ``release``/``__exit__`` will close them since the pool refuses to
        recycle after shutdown.
        """
        with self._global_lock:
            if self._closed:
                return
            self._closed = True
            for bucket in self._buckets.values():
                with bucket.cond:
                    for _, sock in bucket.idle:
                        self._safe_close(sock)
                    bucket.idle.clear()
                    bucket.cond.notify_all()

    def stats(self) -> Dict[str, object]:
        """Return a snapshot of pool occupancy.

        The returned dict maps ``"host:port"`` to a dict with ``idle`` and
        ``in_use`` counts, plus a ``"total"`` entry summing both. Intended
        for tests and monitoring; do not rely on the exact shape across
        versions.
        """
        out: Dict[str, object] = {}
        total_idle = 0
        total_in_use = 0
        with self._global_lock:
            for (host, port), bucket in self._buckets.items():
                with bucket.cond:
                    idle = len(bucket.idle)
                    in_use = bucket.in_use
                out[f"{host}:{port}"] = {"idle": idle, "in_use": in_use}
                total_idle += idle
                total_in_use += in_use
        out["total"] = {"idle": total_idle, "in_use": total_in_use}
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _get_bucket(self, addr: Addr) -> _Bucket:
        with self._global_lock:
            b = self._buckets.get(addr)
            if b is None:
                b = _Bucket()
                self._buckets[addr] = b
            return b

    def _take_idle(self, bucket: _Bucket) -> Optional[socket.socket]:
        # Must be called with bucket.cond held. Returns a healthy socket or
        # None. Closes any socket the OS reports as closed.
        while bucket.idle:
            _, sock = bucket.idle.pop()
            if self._is_alive(sock):
                return sock
            self._safe_close(sock)
        return None

    def _can_open(self, bucket: _Bucket) -> bool:
        # Must be called with bucket.cond held.
        if bucket.in_use + len(bucket.idle) >= self._max_per_host:
            return False
        with self._global_lock:
            total = sum(len(b.idle) + b.in_use for b in self._buckets.values())
        if total >= self._max_total:
            # Try to evict an idle connection from another bucket to make room.
            if self._evict_one_idle(exclude=bucket):
                return True
            return False
        return True

    def _evict_one_idle(self, *, exclude: _Bucket) -> bool:
        # Must be called WITHOUT bucket.cond held for ``exclude`` (we hold the
        # global lock). Evicts the oldest idle connection in any bucket other
        # than ``exclude``. Returns True if one was evicted.
        candidate: Optional[Tuple[_Bucket, int, int]] = None  # (bucket, idx, ts)
        for b in self._buckets.values():
            if b is exclude or not b.idle:
                continue
            # idle[0] is the oldest by construction (we append on release).
            ts = b.idle[0][0]
            if candidate is None or ts < candidate[2]:
                candidate = (b, 0, ts)
        if candidate is None:
            return False
        b, idx, _ = candidate
        _, sock = b.idle.pop(idx)
        self._safe_close(sock)
        b.cond.notify_all()
        return True

    def _release(self, conn: PooledConnection, *, broken: bool) -> None:
        bucket = self._get_bucket((conn.host, conn.port))
        with bucket.cond:
            bucket.in_use -= 1
            if bucket.in_use < 0:
                # Defensive: should never happen unless release is called more
                # times than acquire. Reset rather than corrupt state.
                bucket.in_use = 0
            if not broken and not self._closed and self._is_alive(conn.socket):
                bucket.idle.append((int(self._clock()), conn.socket))
            else:
                self._safe_close(conn.socket)
            bucket.cond.notify()

    @staticmethod
    def _is_alive(sock: socket.socket) -> bool:
        # A non-blocking peek is the cheapest portable liveness check. If the
        # peer closed cleanly we get b""; if the connection is reset we get an
        # error; if there's real data we put it back and let the caller read.
        try:
            sock.setblocking(False)
            try:
                data = sock.recv(1, socket.MSG_PEEK)
            finally:
                sock.setblocking(True)
        except BlockingIOError:
            return True
        except OSError:
            return False
        if data == b"":
            return False
        return True

    @staticmethod
    def _safe_close(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _open(self, host: str, port: int) -> socket.socket:
        # ``create_connection`` handles getaddrinfo + connect with a timeout.
        # We disable Nagle by default because keep-alive pools are typically
        # used for request/response protocols where delayed ACKs hurt.
        sock = socket.create_connection((host, port), timeout=self._connect_timeout)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            # Not all platforms support TCP_NODELAY (e.g. some Unix domain
            # sockets routed through the same API). Non-fatal.
            pass
        return sock
