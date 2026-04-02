"""Configuration management for CryoEMAgent."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


@dataclass
class CryoSPARCConfig:
    """CryoSPARC connection configuration."""

    url: str = ""
    email: str = ""
    password: str = ""

    def __post_init__(self):
        self.url = self.url or os.getenv("CRYOSPARC_URL", "http://localhost:39000")
        self.email = self.email or os.getenv("CRYOSPARC_EMAIL", "")
        self.password = self.password or os.getenv("CRYOSPARC_PASSWORD", "")

    def validate(self) -> bool:
        return bool(self.url and self.email and self.password)


@dataclass
class LLMConfig:
    """LLM configuration for reasoning."""

    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096

    def __post_init__(self):
        # Set the API key based on provider
        provider = (self.provider or "openai").lower()
        if provider == "anthropic":
            self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
            # Default model for Anthropic
            if not self.model:
                self.model = "claude-sonnet-4-6"
        else:
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
            # Default model for OpenAI
            if not self.model:
                self.model = "gpt-4o"

    def validate(self) -> bool:
        return bool(self.api_key)


@dataclass
class GPCRParameters:
    """GPCR-optimized processing parameters."""

    box_size: int = 256
    particle_diameter_min: float = 80.0
    particle_diameter_max: float = 150.0
    num_2d_classes: int = 50
    num_abinit_classes: int = 3
    symmetry: str = "C1"
    resolution_target: float = 3.5
    mask_threshold: float = 0.2


@dataclass
class ProcessingDefaults:
    """Default processing parameters."""

    pixel_size: float = 1.05
    voltage: int = 300
    spherical_aberration: float = 2.7
    amplitude_contrast: float = 0.1
    total_dose: float = 50.0

    gpcr: GPCRParameters = field(default_factory=GPCRParameters)


@dataclass
class AgentConfig:
    """
    Agent-specific configuration.

    Controls paths, LLM settings, and agent behaviour limits.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)

    mcp_server_src_path: str = ""   # path to Cryosparc_mcp_Server/src
    mcp_config_path: str = ""        # path to mcp server config YAML
    root_dir: str = ""               # where to store runs/ and reports/
    max_agent_iterations: int = 200

    def __post_init__(self):
        self.mcp_server_src_path = self.mcp_server_src_path or os.getenv(
            "CRYOEM_AGENT_MCP_SRC_PATH", ""
        )
        self.mcp_config_path = self.mcp_config_path or os.getenv(
            "CRYOEM_AGENT_MCP_CONFIG", ""
        )
        self.root_dir = self.root_dir or os.getenv(
            "CRYOEM_AGENT_ROOT_DIR", ""
        )


@dataclass
class Config:
    """Main configuration container."""

    cryosparc: CryoSPARCConfig = field(default_factory=CryoSPARCConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    processing: ProcessingDefaults = field(default_factory=ProcessingDefaults)
    agent: AgentConfig = field(default_factory=AgentConfig)

    log_level: str = "INFO"
    max_retries: int = 3
    polling_interval: int = 30

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_file and env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            cryosparc=CryoSPARCConfig(),
            llm=LLMConfig(),
            processing=ProcessingDefaults(),
            agent=AgentConfig(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            polling_interval=int(os.getenv("POLLING_INTERVAL", "30")),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        cryosparc_data = data.get("cryosparc", {})
        llm_data = data.get("llm", {})
        processing_data = dict(data.get("processing", {}))
        agent_data = data.get("agent", {})

        gpcr_data = processing_data.pop("gpcr", {})
        gpcr_params = GPCRParameters(**gpcr_data) if gpcr_data else GPCRParameters()

        llm_config = LLMConfig(**llm_data) if llm_data else LLMConfig()
        agent_config = AgentConfig(llm=llm_config, **{
            k: v for k, v in agent_data.items() if k != "llm"
        }) if agent_data else AgentConfig(llm=llm_config)

        return cls(
            cryosparc=CryoSPARCConfig(**cryosparc_data),
            llm=llm_config,
            processing=ProcessingDefaults(**processing_data, gpcr=gpcr_params),
            agent=agent_config,
            log_level=data.get("log_level", "INFO"),
            max_retries=data.get("max_retries", 3),
            polling_interval=data.get("polling_interval", 30),
        )

    @classmethod
    def from_mcp_config(cls, mcp_config_path: str) -> "Config":
        """
        Load configuration from an MCP server YAML profile.

        Extracts the cryosparc section to populate CryoSPARCConfig and
        merges with defaults for the agent config.

        Parameters
        ----------
        mcp_config_path : str
            Path to the MCP server config YAML file.

        Returns
        -------
        Config
        """
        try:
            import yaml  # noqa: PLC0415
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except ImportError:
            # Fallback: try json
            import json  # noqa: PLC0415
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

        if raw is None:
            raw = {}

        cs_raw = raw.get("cryosparc", {})

        # Map MCP cryosparc fields to our CryoSPARCConfig fields
        cryosparc_config = CryoSPARCConfig(
            url=cs_raw.get("base_url", "") or os.getenv("CRYOSPARC_URL", ""),
            email=cs_raw.get("email", "") or os.getenv("CRYOSPARC_EMAIL", ""),
            password=cs_raw.get("password", "") or os.getenv("CRYOSPARC_PASSWORD", ""),
        )

        llm_config = LLMConfig()
        agent_config = AgentConfig(
            llm=llm_config,
            mcp_config_path=mcp_config_path,
        )

        return cls(
            cryosparc=cryosparc_config,
            llm=llm_config,
            processing=ProcessingDefaults(),
            agent=agent_config,
            log_level=raw.get("log_level", "INFO"),
        )

    def validate(self) -> tuple:
        """Validate configuration and return status with errors."""
        errors = []

        if not self.cryosparc.validate():
            errors.append("CryoSPARC credentials not configured")

        if not self.llm.validate():
            errors.append("LLM API key not configured")

        return len(errors) == 0, errors
