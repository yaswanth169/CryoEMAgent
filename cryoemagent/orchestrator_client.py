"""OrchestratorClient - bridges CryoEMAgent to the MCP orchestrator via direct Python import."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge patch into base, returning a new dict."""
    result = dict(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class OrchestratorClient:
    """
    Bridges the CryoEMAgent to the MCP orchestrator via direct Python import.

    This class dynamically adds the Cryosparc_mcp_Server/src directory to sys.path
    and imports the necessary components (RunStore, Orchestrator, CryoSPARCAdapter,
    write_report) at construction time.
    """

    _DEFAULT_MCP_SRC: Path = (
        Path(__file__).parent.parent.parent.parent
        / "Cryosparc_mcp_Server"
        / "src"
    )

    def __init__(self, config: Dict[str, Any], mcp_src_path: Optional[str] = None):
        """
        Parameters
        ----------
        config : dict
            Full MCP server config dict (loaded from profile YAML).
        mcp_src_path : str, optional
            Override path to Cryosparc_mcp_Server/src.  Defaults to the sibling
            directory relative to this package.
        """
        self._config = config
        self._mcp_src = Path(mcp_src_path) if mcp_src_path else self._DEFAULT_MCP_SRC

        self._ensure_mcp_on_path()

        # Import MCP components after path is configured.
        from cryosparc_mcp_server.state_store import RunStore  # noqa: PLC0415
        from cryosparc_mcp_server.orchestrator import Orchestrator  # noqa: PLC0415
        from cryosparc_mcp_server.cryosparc_adapter import CryoSPARCAdapter  # noqa: PLC0415
        from cryosparc_mcp_server.report import write_report  # noqa: PLC0415
        from cryosparc_mcp_server.config import ensure_dirs  # noqa: PLC0415

        self._RunStore = RunStore
        self._Orchestrator = Orchestrator
        self._CryoSPARCAdapter = CryoSPARCAdapter
        self._write_report_fn = write_report
        self._ensure_dirs = ensure_dirs

        root = self._resolve_root()
        self._ensure_dirs(root)

        self.run_store = RunStore(root / "runs")
        self._reports_dir = root / "reports"
        self._adapter = CryoSPARCAdapter(config["cryosparc"])

        logger.info(
            "OrchestratorClient initialised. MCP src=%s root=%s", self._mcp_src, root
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_mcp_on_path(self) -> None:
        """Add the MCP src directory to sys.path if not already present."""
        mcp_src_str = str(self._mcp_src.resolve())
        if mcp_src_str not in sys.path:
            sys.path.insert(0, mcp_src_str)
            logger.debug("Added %s to sys.path", mcp_src_str)

    def _resolve_root(self) -> Path:
        """Return the root directory for runs/ and reports/."""
        root_str = self._config.get("root_dir", "")
        if root_str:
            return Path(root_str)
        # Fall back: sibling of the mcp src
        return self._mcp_src.parent.parent / "agent_runs"

    def _make_orchestrator(self, runtime_overrides: Optional[Dict[str, Any]] = None):
        """Instantiate an Orchestrator with optional runtime overrides applied."""
        cfg = self._config
        if runtime_overrides:
            cfg = _deep_merge(cfg, runtime_overrides)
        return self._Orchestrator(cfg, self._adapter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_run(self, runtime_overrides: Optional[Dict[str, Any]] = None):
        """
        Create and persist a new RunState, pre-seeded with configuration.

        Parameters
        ----------
        runtime_overrides : dict, optional
            Key/value overrides merged into the stored config.

        Returns
        -------
        RunState
        """
        from cryosparc_mcp_server.state_store import RunState  # noqa: PLC0415

        state = self.run_store.new_run()
        state.current_stage = "w1"
        state.current_step = "import_movies"
        state.status = "running"

        cfg = self._config
        if runtime_overrides:
            cfg = _deep_merge(cfg, runtime_overrides)
        state.config = cfg

        self.run_store.save(state)
        logger.info("Created new run %s", state.run_id)
        return state

    def validate_inputs(self, runtime_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate that all required inputs are present and accessible.

        Returns
        -------
        dict with keys "ok" (bool) and "issues" (list of str).
        """
        orch = self._make_orchestrator(runtime_overrides)
        try:
            result = orch.validate_inputs()
        except Exception as exc:
            logger.exception("validate_inputs raised")
            result = {"ok": False, "issues": [str(exc)]}
        return result

    def load_state(self, run_id: str):
        """
        Load and return a RunState by run_id, or None if not found.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        RunState or None
        """
        return self.run_store.load(run_id)

    def save_state(self, state) -> None:
        """Persist the given RunState."""
        self.run_store.save(state)

    def step(self, state):
        """
        Execute one pipeline step via the orchestrator.

        On any exception the state status is set to "failed" and the exception
        message is stored in state.errors before saving.

        Parameters
        ----------
        state : RunState

        Returns
        -------
        RunState  (the updated state, also persisted to disk)
        """
        orch = self._make_orchestrator(state.config if state.config else None)
        try:
            updated = orch.run_until_pause_or_done(state, single_step=True)
        except Exception as exc:
            logger.exception("Orchestrator step raised an exception for run %s", state.run_id)
            state.status = "failed"
            error_key = f"step_error_{state.current_step}"
            state.errors[error_key] = str(exc)
            updated = state
        self.run_store.save(updated)
        return updated

    def resume_checkpoint(self, state, suggested_params=None):
        """
        Resume a paused checkpoint step, optionally applying VLM-suggested params.

        Parameters
        ----------
        state : RunState
        suggested_params : dict, optional
            Parameter key/value pairs recommended by VLMCritic (e.g. ncc_threshold).

        Returns
        -------
        RunState
        """
        orch = self._make_orchestrator(state.config if state.config else None)
        try:
            orch.resume_checkpoint(state, suggested_params=suggested_params or {})
        except Exception as exc:
            logger.exception("resume_checkpoint raised for run %s", state.run_id)
            state.errors[f"resume_error_{state.current_step}"] = str(exc)

        state.checkpoint_required = False
        state.checkpoint_message = ""
        state.checkpoint_job_uid = ""
        self.run_store.save(state)
        return state

    def fetch_checkpoint_image(self, state) -> Optional[bytes]:
        """
        Fetch a thumbnail/plot image relevant to the current checkpoint.

        Interactive checkpoint jobs are in 'waiting' state and have no plots yet.
        We instead fetch from the PRECEDING completed job whose output is being
        reviewed at this checkpoint:
          curate            → patch_ctf   (CTF power spectra)
          inspect_blob      → blob_picker (micrograph + picks)
          inspect_template  → template_picker
        Falls back to the checkpoint job itself, then to None.
        """
        _PRECEDING: Dict[str, str] = {
            "curate":           "patch_ctf",
            "inspect_blob":     "blob_picker",
            "inspect_template": "template_picker",
        }
        step = getattr(state, "current_step", "")
        jobs = getattr(state, "jobs", {})
        project_uid = self._config.get("cryosparc", {}).get("project_uid", "")

        source_uid = jobs.get(_PRECEDING.get(step, ""), "") or getattr(state, "checkpoint_job_uid", "")
        if not source_uid:
            return None
        try:
            return self._adapter.fetch_job_thumbnail(project_uid, source_uid)
        except Exception as exc:
            logger.debug("fetch_checkpoint_image failed for %s: %s", source_uid, exc)
            return None

    def write_report(self, state) -> Dict[str, str]:
        """
        Write markdown and JSON reports for the given run.

        Parameters
        ----------
        state : RunState

        Returns
        -------
        dict with keys "markdown_report" and "json_report".
        """
        return self._write_report_fn(self._reports_dir, state)

    def list_runs(self) -> List[str]:
        """Return a sorted list of all run IDs."""
        return self.run_store.list_runs()

    def get_cs_client(self):
        """
        Return the underlying CryoSPARC client instance (cryosparc.tools.CryoSPARC).

        Useful for quality critic methods that need direct API access.
        """
        return self._adapter.cs
