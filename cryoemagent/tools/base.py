"""Base tool interface for CryoEMAgent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ToolStatus(str, Enum):
    """Tool execution status."""
    
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


@dataclass
class ToolResult:
    """Result from tool execution."""
    
    status: ToolStatus
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None
    
    @classmethod
    def success(cls, data: Dict[str, Any], message: str = "") -> "ToolResult":
        return cls(status=ToolStatus.SUCCESS, data=data, message=message)
    
    @classmethod
    def failure(cls, error: str, data: Optional[Dict[str, Any]] = None) -> "ToolResult":
        return cls(status=ToolStatus.FAILURE, data=data or {}, error=error)
    
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS


class Tool(ABC):
    """Abstract base class for all tools."""
    
    name: str = "base_tool"
    description: str = "Base tool interface"
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_params(self, **kwargs) -> tuple[bool, str]:
        """Validate input parameters."""
        return True, ""
