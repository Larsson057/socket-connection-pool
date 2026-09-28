# socket-connection-pool

A small, dependency-free pool of keep-alive TCP connections keyed by `(host, port)`, with per-host and global caps and idle eviction.

```python
from socket_connection_pool import ConnectionPool

pool = ConnectionPool(max_per_host=8, max_total=64, idle_timeout=60.0)
try:
    with pool.acquire("127.0.0.1", 5432) as conn:
        conn.socket.sendall(b"PING\r\n")
        reply = conn.socket.recv(64)
except OSError:
    # connect or I/O failed; the pool drops the broken connection
    pass

# Later, when traffic has died down:
pool.sweep_idle()
pool.close()
```

## Why

Repeatedly opening TCP connections for short request/response exchanges wastes a round-trip on the handshake and, under TLS, another one or two. Reusing a warm socket collapses that to a single send/recv. The trade-off this library makes is simplicity over sophistication: there is no background reaper thread, no health-check pinging, and no integration with any event loop. Eviction is explicit (`sweep_idle`), and liveness is checked opportunistically when a connection is taken off the idle queue.

## The awkward edge

A socket that the *peer* has closed quietly (half-open) is not always detectable without writing to it. This pool does a non-blocking `recv(MSG_PEEK)` before handing out an idle connection, which catches clean closes and resets but not a silently dropped path. If a `sendall` on a checked-out connection raises, pass `broken=True` to `release` (or let the context manager do it by raising from the body) so the pool drops the socket instead of recycling it.

## Exports

- `ConnectionPool` — the pool itself. Constructor keyword args: `max_per_host`, `max_total`, `idle_timeout`, `connect_timeout`, `clock`.
- `PooledConnection` — what `acquire` returns. Use `.socket` for the raw `socket.socket`, `.release(broken=...)` to return it, or `with ... as conn:` for automatic release.
- `PoolError` — raised for exhaustion, timeouts, shutdown, and bad arguments.
