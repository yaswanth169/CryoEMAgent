"""Planner module for CryoEMAgent - LLM-based planning and reasoning."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import re

from openai import OpenAI

from cryoemagent.core.memory import Memory
from cryoemagent.config import LLMConfig


class ActionType(str, Enum):
    """Types of actions the agent can take."""
    
    CREATE_JOB = "create_job"
    SET_PARAM = "set_param"
    CONNECT_INPUT = "connect_input"
    QUEUE_JOB = "queue_job"
    WAIT_JOB = "wait_job"
    LOAD_OUTPUT = "load_output"
    ASSESS_QUALITY = "assess_quality"
    FINISH = "finish"


@dataclass
class PlannedAction:
    """A single planned action."""
    
    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
        }


@dataclass
class Plan:
    """Execution plan with multiple actions."""
    
    goal: str
    actions: List[PlannedAction]
    contingency: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "actions": [a.to_dict() for a in self.actions],
            "contingency": self.contingency,
        }


SYSTEM_PROMPT = """You are an expert cryo-EM data processing agent specialized in GPCR structure determination.

Your task is to plan the next processing step based on the current state and available tools.

WORKFLOW ORDER:
1. import_movies - Import raw movie data
2. patch_motion_correction - Correct beam-induced motion
3. patch_ctf_estimation - Estimate contrast transfer function
4. blob_picker OR template_picker - Pick particle locations
5. extract_micrographs - Extract particle images
6. class_2d + select_2d - 2D classification and selection
7. homo_abinit - Ab-initio 3D reconstruction
8. hetero_refine OR homo_refine - Heterogeneous or homogeneous refinement
9. nonuniform_refine - Non-uniform refinement for final map

OUTPUT FORMAT:
You must respond with a JSON object containing:
{
    "goal": "Brief description of what we're trying to achieve",
    "reasoning": "Chain-of-thought reasoning for this decision",
    "actions": [
        {
            "action_type": "create_job|set_param|connect_input|queue_job|wait_job|load_output|assess_quality|finish",
            "parameters": {...},
            "reasoning": "Why this specific action"
        }
    ],
    "contingency": "What to do if this step fails"
}

IMPORTANT:
- Always check current state before recommending next step
- Use GPCR-optimized parameters (box_size=256, particle diameter 80-150Å)
- Monitor quality metrics and adjust if needed
- Stop if resolution target is reached
"""


class Planner:
    """LLM-based planning engine."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.conversation_history: List[Dict[str, str]] = []
    
    def plan_next_step(self, memory: Memory) -> Plan:
        """Generate the next execution plan based on current memory state."""
        context = memory.get_full_context()
        
        user_message = f"""Current context:
{context}

Based on the above state, what is the next processing step?
Respond with a JSON plan."""
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history[-10:],
        ]
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })
        
        return self._parse_plan(assistant_message)
    
    def reflect_on_result(self, memory: Memory, result: Dict[str, Any]) -> str:
        """Reflect on a result and provide analysis."""
        context = memory.get_full_context()
        
        user_message = f"""Result from last action:
{json.dumps(result, indent=2)}

Current context:
{context}

Analyze this result. Is it satisfactory? What should we do next?
Provide brief analysis and recommendation."""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content
    
    def _parse_plan(self, response: str) -> Plan:
        """Parse LLM response into a structured Plan."""
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return Plan(
                goal="Continue processing",
                actions=[PlannedAction(
                    action_type=ActionType.FINISH,
                    parameters={"reason": "Could not parse LLM response"},
                    reasoning=response,
                )],
            )
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return Plan(
                goal="Continue processing",
                actions=[PlannedAction(
                    action_type=ActionType.FINISH,
                    parameters={"reason": "Invalid JSON in response"},
                    reasoning=response,
                )],
            )
        
        actions = []
        for action_data in data.get("actions", []):
            try:
                action_type = ActionType(action_data.get("action_type", "finish"))
            except ValueError:
                action_type = ActionType.FINISH
            
            actions.append(PlannedAction(
                action_type=action_type,
                parameters=action_data.get("parameters", {}),
                reasoning=action_data.get("reasoning", ""),
            ))
        
        if not actions:
            actions.append(PlannedAction(
                action_type=ActionType.FINISH,
                parameters={"reason": "No actions specified"},
                reasoning=data.get("reasoning", ""),
            ))
        
        return Plan(
            goal=data.get("goal", "Continue processing"),
            actions=actions,
            contingency=data.get("contingency"),
        )
    
    def get_initial_plan(self, movies_path: str, params: Dict[str, Any]) -> Plan:
        """Generate initial plan for starting a new processing session."""
        return Plan(
            goal="Import movies and begin GPCR processing pipeline",
            actions=[
                PlannedAction(
                    action_type=ActionType.CREATE_JOB,
                    parameters={
                        "job_type": "import_movies",
                        "params": {
                            "blob_paths": movies_path,
                            "psize_A": params.get("pixel_size", 1.05),
                            "accel_kv": params.get("voltage", 300),
                            "cs_mm": params.get("spherical_aberration", 2.7),
                            "total_dose": params.get("total_dose", 50.0),
                        },
                    },
                    reasoning="Starting pipeline with movie import",
                ),
                PlannedAction(
                    action_type=ActionType.QUEUE_JOB,
                    parameters={},
                    reasoning="Queue the import job for execution",
                ),
                PlannedAction(
                    action_type=ActionType.WAIT_JOB,
                    parameters={"error_on_incomplete": True},
                    reasoning="Wait for import to complete before proceeding",
                ),
            ],
            contingency="If import fails, check movie paths and parameters",
        )
