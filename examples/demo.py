"""Example script demonstrating CryoEMAgent usage."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cryoemagent import CryoEMAgent, Config
from cryoemagent.playbooks.gpcr import GPCRPlaybook, GPCRParameters


def run_example():
    """Run example GPCR workflow."""
    print("=" * 60)
    print("CryoEMAgent - GPCR Structure Determination Example")
    print("=" * 60)
    
    try:
        config = Config.from_env()
        valid, errors = config.validate()
        
        if not valid:
            print("\nConfiguration errors:")
            for error in errors:
                print(f"  - {error}")
            print("\nPlease set the following environment variables:")
            print("  CRYOSPARC_URL, CRYOSPARC_EMAIL, CRYOSPARC_PASSWORD")
            print("  OPENAI_API_KEY")
            print("\nOr copy .env.example to .env and fill in values")
            return
        
        print("\n✓ Configuration validated")
        print(f"  CryoSPARC: {config.cryosparc.url}")
        print(f"  LLM Model: {config.llm.model}")
    except Exception as e:
        print(f"\nConfiguration error: {e}")
        return
    
    gpcr_params = GPCRParameters(
        min_particle_diameter=80.0,
        max_particle_diameter=150.0,
        box_size=256,
        num_2d_classes=50,
        num_abinit_classes=3,
        resolution_target=3.5,
    )
    
    playbook = GPCRPlaybook(gpcr_params)
    
    print("\n" + "-" * 40)
    print("GPCR Workflow Steps:")
    print("-" * 40)
    for i, step in enumerate(playbook.steps, 1):
        print(f"  {i:2d}. {step.name} ({step.job_type})")
    
    print("\n" + "-" * 40)
    print("Example Usage:")
    print("-" * 40)
    print("""
from cryoemagent import CryoEMAgent

agent = CryoEMAgent()
result = agent.run(
    project_uid="P3",
    workspace_uid="W1",
    movies_path="/path/to/movies/*.mrc",
    pixel_size=1.05,
    voltage=300,
    target_resolution=3.5,
)

if result.success:
    print(f"Resolution: {result.resolution} Å")
    print(f"Particles: {result.particle_count}")
else:
    print(f"Error: {result.error}")
""")
    
    print("\n" + "-" * 40)
    print("CLI Usage:")
    print("-" * 40)
    print("""
# Run full workflow
cryoem-agent run --project P3 --workspace W1 --movies /data/*.mrc

# Check CryoSPARC connection
cryoem-agent status

# List jobs in project
cryoem-agent jobs --project P3
""")


def demo_memory_system():
    """Demonstrate the memory system."""
    from cryoemagent.core.memory import Memory, ProcessingState
    
    print("\n" + "=" * 60)
    print("Memory System Demo")
    print("=" * 60)
    
    memory = Memory()
    memory.initialize("P1", "W1", "Determine GPCR structure to 3.5Å resolution")
    
    memory.episodic.add_observation("Movies imported successfully", {"count": 1000})
    memory.episodic.add_decision("Proceed with motion correction", "All movies valid")
    
    print("\nSemantic Memory (Domain Knowledge):")
    print(f"  Available tools: {len(memory.semantic.available_tools)}")
    print(f"  Workflow stages: {len(memory.semantic.domain_knowledge['workflow_stages'])}")
    
    print("\nEpisodic Memory (Session State):")
    print(f"  Project: {memory.episodic.state.project_uid}")
    print(f"  Current stage: {memory.episodic.state.current_stage}")
    print(f"  Observations: {len(memory.episodic.observations)}")
    print(f"  Decisions: {len(memory.episodic.decisions)}")
    
    print("\nFull Context for LLM:")
    print("-" * 40)
    context = memory.get_full_context()
    print(context[:500] + "..." if len(context) > 500 else context)


def demo_planner():
    """Demonstrate the planning system."""
    from cryoemagent.core.planner import Planner
    from cryoemagent.config import LLMConfig
    
    print("\n" + "=" * 60)
    print("Planner Demo")
    print("=" * 60)
    
    config = LLMConfig()
    if not config.api_key:
        print("\nNote: OpenAI API key not set, showing initial plan only")
    
    planner = Planner(config)
    
    plan = planner.get_initial_plan(
        movies_path="/data/gpcr_movies/*.mrc",
        params={"pixel_size": 1.05, "voltage": 300}
    )
    
    print(f"\nInitial Plan: {plan.goal}")
    print(f"Actions: {len(plan.actions)}")
    for i, action in enumerate(plan.actions, 1):
        print(f"  {i}. {action.action_type.value}: {action.reasoning}")


if __name__ == "__main__":
    run_example()
    demo_memory_system()
    demo_planner()
