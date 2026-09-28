import socket
import threading
import unittest

from socket_connection_pool import ConnectionPool, PoolError, PooledConnection


class _FakeClock:
    """Monotonic-ish clock that only advances when told to."""

    def __init__(self, start: int = 0):
        self._t = start

    def __call__(self) -> float:
        return float(self._t)

    def advance(self, seconds: int) -> None:
        self._t += seconds


class _Listener:
    """A minimal TCP listener for round-trip tests.

    Accepts one connection per ``accept()``, echoing back any bytes received.
    Each accepted socket is keep-alive: it stays open until the client closes
    it or the listener is stopped.
    """

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.host, self.port = self.sock.getsockname()
        self._accepted: list[socket.socket] = []
        self._lock = threading.Lock()

    def accept(self) -> socket.socket:
        conn, _ = self.sock.accept()
        with self._lock:
            self._accepted.append(conn)
        return conn

    def stop(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
        with self._lock:
            for c in self._accepted:
                try:
                    c.close()
                except OSError:
                    pass
            self._accepted.clear()


class _EchoServer:
    """Background thread that accepts and echoes on each connection."""

    def __init__(self) -> None:
        self.listener = _Listener()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def host(self) -> str:
        return self.listener.host

    @property
    def port(self) -> int:
        return self.listener.port

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn = self.listener.sock.accept()[0]
            except OSError:
                break
            t = threading.Thread(target=self._echo, args=(conn,), daemon=True)
            t.start()

    def _echo(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                data = conn.recv(4096)
                if not data:
                    break
                conn.sendall(data)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        self.listener.stop()
        self._thread.join(timeout=1)


class TestConnectionPool(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _EchoServer()
        self.clock = _FakeClock()

    def tearDown(self) -> None:
        self.server.stop()

    def test_acquire_release_reuses_connection(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        c1 = pool.acquire(self.server.host, self.server.port)
        c1.socket.sendall(b"hello")
        self.assertEqual(c1.socket.recv(5), b"hello")
        c1.release()

        c2 = pool.acquire(self.server.host, self.server.port)
        # Same fd => same underlying socket was recycled.
        self.assertEqual(c1.socket.fileno(), c2.socket.fileno())
        c2.release()
        pool.close()

    def test_context_manager_releases_on_normal_exit(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        with pool.acquire(self.server.host, self.server.port) as conn:
            self.assertIsInstance(conn, PooledConnection)
        # Pool should now have one idle connection.
        stats = pool.stats()
        key = f"{self.server.host}:{self.server.port}"
        self.assertEqual(stats[key]["idle"], 1)
        self.assertEqual(stats[key]["in_use"], 0)
        pool.close()

    def test_context_manager_releases_as_broken_on_exception(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        with self.assertRaises(RuntimeError):
            with pool.acquire(self.server.host, self.server.port):
                raise RuntimeError("boom")
        # Broken connections are closed, not recycled.
        stats = pool.stats()
        key = f"{self.server.host}:{self.server.port}"
        self.assertEqual(stats[key]["idle"], 0)
        pool.close()

    def test_max_per_host_blocks_then_succeeds(self) -> None:
        pool = ConnectionPool(max_per_host=2, clock=self.clock)
        a = pool.acquire(self.server.host, self.server.port)
        b = pool.acquire(self.server.host, self.server.port)

        result: dict = {}

        def third() -> None:
            try:
                c = pool.acquire(self.server.host, self.server.port, timeout=5)
                result["conn"] = c
            except Exception as exc:  # pragma: no cover - we want to surface failures
                result["err"] = exc

        t = threading.Thread(target=third)
        t.start()
        # Give the waiter a moment to park on the condition.
        t.join(timeout=0.2)
        self.assertNotIn("conn", result)

        b.release()
        t.join(timeout=1)
        self.assertIn("conn", result)
        self.assertIsInstance(result["conn"], PooledConnection)
        result["conn"].release()
        a.release()
        pool.close()

    def test_acquire_timeout_raises_pool_error(self) -> None:
        pool = ConnectionPool(max_per_host=1, clock=self.clock)
        a = pool.acquire(self.server.host, self.server.port)
        with self.assertRaises(PoolError) as ctx:
            pool.acquire(self.server.host, self.server.port, timeout=0)
        self.assertIn("timed out", str(ctx.exception))
        a.release()
        pool.close()

    def test_sweep_idle_evicts_expired(self) -> None:
        pool = ConnectionPool(idle_timeout=10, clock=self.clock)
        c = pool.acquire(self.server.host, self.server.port)
        c.release()
        self.clock.advance(9)
        self.assertEqual(pool.sweep_idle(), 0)
        self.clock.advance(2)
        self.assertEqual(pool.sweep_idle(), 1)
        self.assertEqual(pool.sweep_idle(), 0)
        pool.close()

    def test_sweep_idle_keeps_recent(self) -> None:
        pool = ConnectionPool(idle_timeout=10, clock=self.clock)
        pool.acquire(self.server.host, self.server.port).release()
        self.clock.advance(5)
        self.assertEqual(pool.sweep_idle(), 0)
        stats = pool.stats()
        key = f"{self.server.host}:{self.server.port}"
        self.assertEqual(stats[key]["idle"], 1)
        pool.close()

    def test_broken_release_closes_socket(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        c = pool.acquire(self.server.host, self.server.port)
        fd = c.socket.fileno()
        c.release(broken=True)
        # The socket should be closed; fileno() returns -1 on a closed socket.
        self.assertEqual(c.socket.fileno(), -1)
        # And not recycled.
        stats = pool.stats()
        key = f"{self.server.host}:{self.server.port}"
        self.assertEqual(stats[key]["idle"], 0)
        pool.close()

    def test_close_marks_pool_shut_down(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        pool.close()
        with self.assertRaises(PoolError):
            pool.acquire(self.server.host, self.server.port)

    def test_close_is_idempotent(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        pool.close()
        pool.close()  # must not raise

    def test_double_release_is_noop(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        c = pool.acquire(self.server.host, self.server.port)
        c.release()
        c.release()  # second release must not corrupt in_use count
        stats = pool.stats()
        key = f"{self.server.host}:{self.server.port}"
        self.assertEqual(stats[key]["in_use"], 0)
        self.assertEqual(stats[key]["idle"], 1)
        pool.close()

    def test_invalid_host_raises_pool_error(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        with self.assertRaises(PoolError):
            pool.acquire("", 80)
        with self.assertRaises(PoolError):
            pool.acquire("localhost", 0)
        with self.assertRaises(PoolError):
            pool.acquire("localhost", 70000)
        pool.close()

    def test_constructor_rejects_bad_args(self) -> None:
        with self.assertRaises(ValueError):
            ConnectionPool(max_per_host=0)
        with self.assertRaises(ValueError):
            ConnectionPool(max_total=0)
        with self.assertRaises(ValueError):
            ConnectionPool(max_per_host=10, max_total=5)
        with self.assertRaises(ValueError):
            ConnectionPool(idle_timeout=-1)
        with self.assertRaises(ValueError):
            ConnectionPool(connect_timeout=0)

    def test_max_total_evicts_idle_from_other_host(self) -> None:
        # Two listeners so we have two distinct (host, port) keys.
        server2 = _EchoServer()
        try:
            pool = ConnectionPool(
                max_per_host=2, max_total=2, clock=self.clock
            )
            # Fill host1 with one idle + one in-use.
            a = pool.acquire(self.server.host, self.server.port)
            a.release()  # idle for host1
            b = pool.acquire(self.server.host, self.server.port)  # in-use for host1

            # Acquiring from host2 should evict host1's idle connection to stay
            # under max_total.
            c = pool.acquire(server2.host, server2.port)
            stats = pool.stats()
            key1 = f"{self.server.host}:{self.server.port}"
            self.assertEqual(stats[key1]["idle"], 0)
            self.assertEqual(stats[key1]["in_use"], 1)
            b.release()
            c.release()
            pool.close()
        finally:
            server2.stop()

    def test_dead_idle_connection_is_dropped_on_acquire(self) -> None:
        pool = ConnectionPool(clock=self.clock)
        c = pool.acquire(self.server.host, self.server.port)
        # Simulate the peer closing the connection while it sits idle.
        # We close the server-side accepted socket by stopping the server; the
        # client socket will then read EOF.
        self.server.stop()
        c.release()
        # Now acquire again; the pool should detect the dead socket and open a
        # fresh one. We need a live server for that, so restart one.
        self.server = _EchoServer()
        c2 = pool.acquire(self.server.host, self.server.port)
        c2.socket.sendall(b"ping")
        self.assertEqual(c2.socket.recv(4), b"ping")
        c2.release()
        pool.close()

    def test_stats_reports_totals(self) -> None:
        pool = ConnectionPool(max_per_host=4, clock=self.clock)
        a = pool.acquire(self.server.host, self.server.port)
        b = pool.acquire(self.server.host, self.server.port)
        a.release()
        stats = pool.stats()
        key = f"{self.server.host}:{self.server.port}"
        self.assertEqual(stats[key], {"idle": 1, "in_use": 1})
        self.assertEqual(stats["total"], {"idle": 1, "in_use": 1})
        b.release()
        pool.close()


if __name__ == "__main__":
    unittest.main()
