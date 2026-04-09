"""
MCP-over-SSH client for CryoEMAgent.

Talks to the Cryosparc_mcp_Server over SSH stdio using MCP JSON-RPC 2.0 protocol.
This is the production client — runs from any laptop, connects to a remote GPU server.
"""

import json
import logging
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-RPC / MCP stdio framing (newline-delimited JSON / JSONL)
# ---------------------------------------------------------------------------

def _encode_message(msg: dict) -> bytes:
    """Encode a JSON-RPC message as a newline-terminated JSON line."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def _read_message(stream) -> dict:
    """
    Read one newline-delimited JSON message from a byte stream.

    Skips empty lines and non-JSON lines (e.g. SSH banners, Python
    warnings).  Blocks until a valid JSON message is available.
    Raises EOFError when the stream closes.
    """
    while True:
        line = stream.readline()
        if not line:
            raise EOFError("MCP server closed the connection")
        text = line.decode("utf-8").strip()
        if not text:
            continue
        if not text.startswith("{"):
            # Skip non-JSON output (SSH banners, warnings, etc.)
            logger.debug("Skipping non-JSON line: %s", text[:120])
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed JSON line: %s", text[:120])
            continue


# ---------------------------------------------------------------------------
# Low-level MCP stdio client
# ---------------------------------------------------------------------------

class MCPStdioClient:
    """
    Minimal MCP client that communicates with a server over subprocess stdio.

    The subprocess is typically an SSH command that launches the MCP server
    on a remote GPU machine — identical to how Cursor / Claude Desktop
    connects to MCP servers.
    """

    def __init__(self, command: str, args: List[str], timeout: float = 600):
        """
        Parameters
        ----------
        command : str
            Executable to run (e.g. "ssh").
        args : list of str
            Arguments to the command (SSH flags, remote command, etc.).
        timeout : float
            Max seconds to wait for a single tool call (default 600 = 10 min,
            enough for long GPU jobs).
        """
        self._timeout = timeout
        self._request_id = 0
        self._lock = threading.Lock()

        logger.info("Spawning MCP server: %s %s", command, " ".join(args[:4]) + "...")
        self._proc = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=-1,   # default buffering — unbuffered (bufsize=0) breaks Windows pipes
        )

        # Drain stderr in background so it doesn't block.
        self._stderr_lines: List[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for SSH connection + Python startup on remote server.
        startup_wait = 5
        logger.info("Waiting %ds for SSH + MCP server startup...", startup_wait)
        time.sleep(startup_wait)

        if self._proc.poll() is not None:
            stderr_out = "\n".join(self._stderr_lines[-10:])
            raise RuntimeError(
                f"MCP server process exited during startup (code {self._proc.returncode}). "
                f"Stderr: {stderr_out}"
            )

        self._initialize()

    def _drain_stderr(self):
        """Read stderr in background, store lines for debugging."""
        try:
            for line in self._proc.stderr:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self._stderr_lines.append(decoded)
                if len(self._stderr_lines) > 200:
                    self._stderr_lines = self._stderr_lines[-100:]
                logger.debug("MCP stderr: %s", decoded)
        except Exception:
            pass

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, msg: dict) -> None:
        """Send a JSON-RPC message to the server."""
        data = _encode_message(msg)
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        """Read the next JSON-RPC message from the server."""
        return _read_message(self._proc.stdout)

    def _initialize(self) -> dict:
        """Perform MCP initialize handshake."""
        # 1. Send initialize request
        init_resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cryoemagent", "version": "0.2.0"},
        })
        logger.info(
            "MCP initialized. Server: %s",
            init_resp.get("result", {}).get("serverInfo", {}).get("name", "unknown"),
        )

        # 2. Send initialized notification (no response expected)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        return init_resp

    def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        with self._lock:
            rid = self._next_id()
            msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
            self._send(msg)

            # Read responses until we get one matching our ID.
            # (Server may send notifications in between.)
            deadline = time.time() + self._timeout
            while time.time() < deadline:
                resp = self._recv()
                if resp.get("id") == rid:
                    return resp
                # It's a notification or mismatched id — log and continue.
                logger.debug("MCP received non-matching message: %s", resp.get("method", "?"))
            raise TimeoutError(f"MCP request {method} timed out after {self._timeout}s")

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call an MCP tool and return the parsed result dict.

        Parameters
        ----------
        tool_name : str
            Name of the MCP tool (e.g. "cs_continue_pipeline").
        arguments : dict, optional
            Arguments to pass to the tool.

        Returns
        -------
        dict
            Parsed JSON result from the tool.
        """
        logger.info("MCP call_tool: %s(%s)", tool_name, json.dumps(arguments or {})[:200])

        resp = self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })

        # Check for JSON-RPC error
        if "error" in resp:
            error = resp["error"]
            raise RuntimeError(f"MCP error {error.get('code')}: {error.get('message')}")

        # Extract tool result from MCP response envelope:
        # result.content[0].text contains the JSON-serialized tool return value.
        result = resp.get("result", {})

        if result.get("isError"):
            content = result.get("content", [{}])
            error_text = content[0].get("text", "Unknown MCP tool error") if content else "Unknown"
            raise RuntimeError(f"MCP tool error: {error_text}")

        content = result.get("content", [])
        if not content:
            return {}

        text = content[0].get("text", "{}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some tools return plain text
            return {"text": text}

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        resp = self._request("tools/list", {})
        return resp.get("result", {}).get("tools", [])

    @property
    def alive(self) -> bool:
        """Check if the subprocess is still running."""
        return self._proc.poll() is None

    def close(self):
        """Terminate the MCP server subprocess."""
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        logger.info("MCP client closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RemoteState — wraps MCP summarize() response
# ---------------------------------------------------------------------------

class RemoteState:
    """
    Lightweight wrapper around the MCP server's summarize() response dict.

    Provides attribute-style access compatible with RunState so the agent
    control loop works identically in local and remote mode.
    """

    def __init__(self, data: Dict[str, Any]):
        self.run_id: str = data.get("run_id", "")
        self.status: str = data.get("status", "running")
        self.current_stage: str = data.get("current_stage", "w1")
        self.current_step: str = data.get("current_step", "")
        self.checkpoint_required: bool = data.get("checkpoint_required", False)
        self.checkpoint_message: str = data.get("checkpoint_message", "")
        self.checkpoint_job_uid: str = data.get("checkpoint_job_uid", "")
        self.workspace_w1_uid: str = data.get("workspace_w1_uid", "")
        self.workspace_w2_uid: str = data.get("workspace_w2_uid", "")
        self.jobs: Dict[str, str] = data.get("jobs", {})
        self.errors: Dict[str, str] = data.get("errors", {})
        self.config: Dict[str, Any] = data.get("config", {})
        self.completed_steps: int = data.get("completed_steps", 0)
        # MCP-specific fields
        self.operator_instruction: str = data.get("operator_instruction", "")
        self.next_suggested_tool: str = data.get("next_suggested_tool", "")
        self._raw = data

    def __repr__(self):
        return (
            f"RemoteState(run_id={self.run_id!r}, status={self.status!r}, "
            f"step={self.current_step!r}, checkpoint={self.checkpoint_required})"
        )


# ---------------------------------------------------------------------------
# MCPOrchestratorClient — drop-in replacement for OrchestratorClient
# ---------------------------------------------------------------------------

class MCPOrchestratorClient:
    """
    Orchestrator client backed by MCP-over-SSH.

    Drop-in replacement for OrchestratorClient — same public API — but
    talks to the MCP server on a remote GPU machine via SSH stdio.
    This is how users run the agent from their laptop.
    """

    def __init__(self, ssh_config: Dict[str, Any], profile_config: Dict[str, Any]):
        """
        Parameters
        ----------
        ssh_config : dict
            Must contain "command" (str) and "args" (list of str).
            Identical format to the mcpServers entry in Claude Desktop config.
        profile_config : dict
            The full profile YAML dict (used for reference; the server has its own copy).
        """
        self._profile = profile_config
        self._mcp = MCPStdioClient(
            command=ssh_config["command"],
            args=ssh_config["args"],
            timeout=ssh_config.get("timeout", 600),
        )

        # Verify connection by listing tools.
        tools = self._mcp.list_tools()
        tool_names = [t.get("name", "") for t in tools]
        logger.info("MCP connected. Available tools: %s", tool_names)

        required = {"cs_start_pipeline", "cs_continue_pipeline", "cs_resume_pipeline",
                     "cs_pipeline_status", "cs_pipeline_report", "cs_list_runs"}
        missing = required - set(tool_names)
        if missing:
            raise RuntimeError(f"MCP server missing required tools: {missing}")

    # ------------------------------------------------------------------
    # Public API (matches OrchestratorClient interface)
    # ------------------------------------------------------------------

    def validate_inputs(self, runtime_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate inputs via MCP server."""
        args = {}
        if runtime_overrides:
            movie_path = runtime_overrides.get("data", {}).get("movie_blob_path", "")
            if movie_path:
                args["movie_blob_path"] = movie_path
        try:
            return self._mcp.call_tool("cs_validate_data_source", args)
        except Exception as exc:
            logger.exception("validate_inputs failed")
            return {"ok": False, "issues": [str(exc)]}

    def new_run(self, runtime_overrides: Optional[Dict[str, Any]] = None) -> RemoteState:
        """Start a new pipeline run. Executes import_movies as the first step."""
        args = {}
        if runtime_overrides:
            args["runtime_overrides"] = runtime_overrides
        result = self._mcp.call_tool("cs_start_pipeline", args)
        state = RemoteState(result)
        logger.info("New run started: %s (step=%s)", state.run_id, state.current_step)
        return state

    def load_state(self, run_id: str) -> Optional[RemoteState]:
        """Load run state from the remote server."""
        result = self._mcp.call_tool("cs_pipeline_status", {"run_id": run_id})
        if "error" in result:
            return None
        return RemoteState(result)

    def save_state(self, state) -> None:
        """No-op in MCP mode — state is managed server-side."""
        pass

    def step(self, state) -> RemoteState:
        """Execute one pipeline step via MCP."""
        result = self._mcp.call_tool("cs_continue_pipeline", {
            "run_id": state.run_id,
            "steps": 1,
        })
        new_state = RemoteState(result)
        logger.info(
            "Step result: run=%s stage=%s step=%s checkpoint=%s status=%s",
            new_state.run_id, new_state.current_stage, new_state.current_step,
            new_state.checkpoint_required, new_state.status,
        )
        return new_state

    def resume_checkpoint(self, state) -> RemoteState:
        """Resume after a human checkpoint via MCP."""
        result = self._mcp.call_tool("cs_resume_pipeline", {"run_id": state.run_id})
        new_state = RemoteState(result)
        logger.info("Checkpoint resumed: step=%s status=%s", new_state.current_step, new_state.status)
        return new_state

    def write_report(self, state) -> Dict[str, str]:
        """Generate report on the remote server."""
        return self._mcp.call_tool("cs_pipeline_report", {"run_id": state.run_id})

    def list_runs(self) -> List[str]:
        """List all run IDs from the remote server."""
        result = self._mcp.call_tool("cs_list_runs", {})
        return result.get("run_ids", [])

    def get_cs_client(self):
        """
        Not available in MCP mode — quality critics that need direct
        CryoSPARC access will gracefully degrade.
        """
        return None

    def close(self):
        """Shut down the SSH/MCP connection."""
        self._mcp.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
