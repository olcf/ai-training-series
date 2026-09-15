"""RedisDataBackend: launches and owns the lifecycle of one or more
independent `redis-server` instances."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shlex
import shutil
import socket
import time
import uuid
from collections.abc import Mapping
from collections.abc import Sequence

from rhapsody.backends.data.base import DataBackend
from rhapsody.backends.data.base import DataBackendStartupError
from rhapsody.backends.data.base import Endpoint

_PING = b"*1\r\n$4\r\nPING\r\n"


def _get_logger() -> logging.Logger:
    """Get logger for the redis data backend module.

    This function provides lazy logger evaluation, ensuring the logger is created after the user has
    configured logging, not at module import time.
    """
    return logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RedisEndpoint(Endpoint):
    """Connection information for one independent Redis node.

    `serialize()` returns a `host:port` string. `RedisEndpoint` never
    constructs a client itself -- build whichever client you need directly,
    e.g. `redis.Redis(host=endpoint.host, port=endpoint.port)`.
    """

    host: str
    port: int

    def serialize(self) -> str:
        return f"{self.host}:{self.port}"


class RedisDataBackend(DataBackend):
    """A DataBackend backed by N independent per-node `redis-server` instances.

    This models independent, unrelated keyspaces -- not a Redis Cluster.
    `backend.endpoints` is a list with one `RedisEndpoint` per host in
    `hosts`.

    `RedisDataBackend()` with no arguments spawns a single local
    `redis-server` on an auto-picked free port -- it works out of the box
    locally. Multi-node (HPC) usage requires an explicit `cmd`
    launch-command template (e.g.
    `"srun --nodelist={host} redis-server --port {port}"`, formatted per
    host/port and executed directly, never via a shell) plus an explicit
    `port`, since a free port picked on the launching host says nothing
    about availability on a remote target host.

    Launching is done via a plain `asyncio` subprocess for now.
    """

    def __init__(
        self,
        *,
        name: str = "redis",
        hosts: Sequence[str] | None = None,
        port: int | None = None,
        cmd: str | None = None,
        redis_server_path: str = "redis-server",
        extra_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        work_dir: str | None = None,
        connect_timeout: float = 5.0,
        startup_timeout: float = 30.0,
        poll_interval: float = 0.2,
        shutdown_grace_period: float = 5.0,
    ) -> None:
        """Initialize a RedisDataBackend.

        Args:
            name: Name this backend is registered under when attached to a
                Session.
            hosts: Hostnames to launch a `redis-server` on, one per entry.
                Defaults to a single `"localhost"` node.
            port: Port to bind on every host. Required when `cmd` is given
                (a free port picked on the launching host says nothing
                about a remote target host); auto-picked per host
                otherwise.
            cmd: Launch-command template, formatted with `host`/`port` and
                executed directly (never via a shell), e.g.
                `"srun --nodelist={host} redis-server --port {port}"`.
            redis_server_path: Path to the `redis-server` executable, used
                when `cmd` is not given.
            extra_args: Extra CLI arguments appended when `cmd` is not
                given.
            env: Environment variables to add/override for the launched
                process(es); the parent's own environment is preserved
                underneath (PATH included), not replaced.
            work_dir: Directory for per-node `redis.node{index}.log` files
                (redis-server's stdout/stderr, redirected there instead of
                left as an undrained pipe). Defaults to a fresh
                `rhapsody.data.<hash>` directory under the cwd.
            connect_timeout: Per-attempt timeout for the readiness PING.
            startup_timeout: Overall timeout to wait for each node to
                become ready.
            poll_interval: Delay between readiness poll attempts.
            shutdown_grace_period: Time to wait after SIGTERM before
                escalating to SIGKILL.

        Raises:
            ValueError: If `hosts` is empty, or `cmd` is given without an
                explicit `port`.
        """
        super().__init__(name=name)
        self.logger = _get_logger()
        self._hosts = list(hosts) if hosts is not None else ["localhost"]
        if not self._hosts:
            raise ValueError("`hosts` must be non-empty if provided")
        if cmd is not None and port is None:
            raise ValueError(
                "`port` must be given explicitly when `cmd` is provided: "
                "RedisDataBackend cannot safely auto-pick a free port on a "
                "remote host from the launching process."
            )
        self._cmd_template = cmd
        self._explicit_port = port
        self._redis_server_path = redis_server_path
        self._extra_args = list(extra_args)
        self._env_overrides = dict(env) if env is not None else None
        # Merge (not replace) the parent environment: passing `env=` should
        # add/override variables, not strip PATH and everything else out
        # from under the child (which would break resolving a bare
        # `redis-server` via PATH even though shutil.which() just found it).
        self._env = (
            {**os.environ, **self._env_overrides} if self._env_overrides is not None else None
        )
        self._work_dir = work_dir or os.path.join(
            os.getcwd(), f"rhapsody.data.{uuid.uuid4().hex[:8]}"
        )
        self._connect_timeout = connect_timeout
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._shutdown_grace_period = shutdown_grace_period
        self._processes: dict[int, asyncio.subprocess.Process] = {}
        self._planned: list[tuple[str, int]] = []
        self._log_paths: dict[int, str] = {}

    def _resolve_port(self) -> int:
        if self._explicit_port is not None:
            return self._explicit_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def _build_argv(self, host: str, port: int) -> list[str]:
        if self._cmd_template is not None:
            try:
                formatted = self._cmd_template.format(host=host, port=port)
            except (KeyError, IndexError) as exc:
                raise ValueError(f"Invalid `cmd` template {self._cmd_template!r}: {exc}") from exc
            return shlex.split(formatted)
        return [self._redis_server_path, "--port", str(port), *self._extra_args]

    async def _do_start(self, wait: bool) -> list[Endpoint]:
        if self._cmd_template is None and shutil.which(self._redis_server_path) is None:
            raise DataBackendStartupError(
                f"'{self._redis_server_path}' not found on PATH; install "
                "redis-server, or pass redis_server_path=/cmd= explicitly."
            )

        os.makedirs(self._work_dir, exist_ok=True)
        self.logger.info(
            "Starting %d redis-server node(s) (logs under %s)...",
            len(self._hosts),
            self._work_dir,
        )
        self._processes = {}
        self._planned = [(host, self._resolve_port()) for host in self._hosts]

        launch_results = await asyncio.gather(
            *(self._launch_one(i, h, p) for i, (h, p) in enumerate(self._planned)),
            return_exceptions=True,
        )
        launch_errors = [r for r in launch_results if isinstance(r, BaseException)]
        if launch_errors:
            await self._terminate_all()
            self.logger.error(
                "Failed to launch %d/%d redis node(s): %s",
                len(launch_errors),
                len(self._planned),
                launch_errors,
            )
            raise DataBackendStartupError(
                f"Failed to launch {len(launch_errors)}/{len(self._planned)} "
                f"redis node(s): {launch_errors}"
            )

        if not wait:
            self.logger.info("RedisDataBackend launched (not waiting for ready)")
            return [RedisEndpoint(host=h, port=p) for h, p in self._planned]

        ready_results = await asyncio.gather(
            *(self._wait_ready_one(i, h, p) for i, (h, p) in enumerate(self._planned)),
            return_exceptions=True,
        )
        ready_errors = [r for r in ready_results if isinstance(r, BaseException)]
        if ready_errors:
            await self._terminate_all()
            self.logger.error(
                "%d/%d redis node(s) failed to become ready: %s",
                len(ready_errors),
                len(self._planned),
                ready_errors,
            )
            raise DataBackendStartupError(
                f"{len(ready_errors)}/{len(self._planned)} redis node(s) "
                f"failed to become ready: {ready_errors}"
            )
        self.logger.info("RedisDataBackend ready: %d node(s)", len(ready_results))
        return list(ready_results)

    async def _launch_one(self, index: int, host: str, port: int) -> None:
        argv = self._build_argv(host, port)
        log_path = os.path.join(self._work_dir, f"redis.node{index}.log")
        self._log_paths[index] = log_path
        log_file = open(log_path, "wb")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=log_file,
                stderr=log_file,
                env=self._env,
            )
        finally:
            # The child has its own fd copy after fork/exec; safe to close
            # the parent's handle immediately.
            log_file.close()
        self._processes[index] = proc

    async def _tail_log(self, index: int, n: int = 4000) -> str:
        path = self._log_paths.get(index)
        if not path:
            return "<no log file>"

        def _read() -> str:
            try:
                with open(path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(max(0, size - n))
                    return f.read().decode(errors="replace")
            except OSError:
                return "<log file unavailable>"

        return await asyncio.to_thread(_read)

    async def _wait_ready_one(self, index: int, host: str, port: int) -> RedisEndpoint:
        deadline = time.monotonic() + self._startup_timeout
        proc = self._processes[index]
        while True:
            if proc.returncode is not None:
                tail = await self._tail_log(index)
                raise DataBackendStartupError(
                    f"redis-server for node {index} ({host}:{port}) exited "
                    f"early with code {proc.returncode}; see "
                    f"{self._log_paths.get(index)}:\n{tail}"
                )
            if await self._ping(host, port):
                self.logger.debug("redis node %d (%s:%d) ready", index, host, port)
                return RedisEndpoint(host=host, port=port)
            if time.monotonic() >= deadline:
                tail = await self._tail_log(index)
                raise DataBackendStartupError(
                    f"Timed out after {self._startup_timeout}s waiting for "
                    f"redis node {index} ({host}:{port}) to become ready; "
                    f"see {self._log_paths.get(index)}:\n{tail}"
                )
            await asyncio.sleep(self._poll_interval)

    async def _ping(self, host: str, port: int) -> bool:
        def _do_ping() -> bool:
            try:
                with socket.create_connection((host, port), timeout=self._connect_timeout) as sock:
                    sock.sendall(_PING)
                    reply = sock.recv(64)
                    return reply.startswith(b"+PONG")
            except OSError:
                return False

        return await asyncio.to_thread(_do_ping)

    async def _do_shutdown(self) -> None:
        self.logger.info("Shutting down RedisDataBackend (%d node(s))...", len(self._processes))
        await self._terminate_all()
        self.logger.info("RedisDataBackend shutdown complete")

    async def _terminate_all(self) -> None:
        await asyncio.gather(
            *(self._terminate_one(p) for p in self._processes.values()),
            return_exceptions=True,
        )
        self._processes = {}

    async def _terminate_one(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._shutdown_grace_period)
            self.logger.debug("redis process (pid=%s) terminated", proc.pid)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            self.logger.debug("redis process (pid=%s) killed after grace period", proc.pid)

    async def _do_ready(self) -> bool:
        if not self._endpoints:
            return False
        results = await asyncio.gather(*(self._ping(ep.host, ep.port) for ep in self._endpoints))
        return all(results)
