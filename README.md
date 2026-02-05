# CryoEMAgent

**Autonomous AI Agent for Cryo-EM Structure Determination**

CryoEMAgent is an intelligent orchestration system that autonomously determines protein structures from cryo-EM data. It combines LLM-based planning with CryoSPARC's GPU-accelerated algorithms to automate the complete structure determination pipeline.

---

## 🎯 What This Agent Does

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CryoEMAgent Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │
│   │    USER      │    │  CryoEMAgent │    │      CryoSPARC           │ │
│   │              │    │              │    │                          │ │
│   │  "Process    │───►│  Memory      │───►│  Motion Correction       │ │
│   │   my GPCR    │    │  + Planning  │    │  CTF Estimation          │ │
│   │   dataset"   │    │  + Tools     │    │  Particle Picking        │ │
│   │              │    │              │    │  2D/3D Classification    │ │
│   └──────────────┘    └──────────────┘    │  3D Refinement           │ │
│                              │            │                          │ │
│                              ▼            │  (GPU Processing)        │ │
│                       ┌──────────────┐    └──────────────────────────┘ │
│                       │   OpenAI     │                                 │
│                       │   GPT-4      │                                 │
│                       │  (Reasoning) │                                 │
│                       └──────────────┘                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Problem
- Cryo-EM structure determination requires **10-15 processing steps**
- Each step has **dozens of parameters** that need tuning
- Experts spend hours manually optimizing workflows
- CryoSPARC provides the algorithms but not autonomous decision-making

### The Solution
CryoEMAgent provides an **AI brain** that:
1. **Observes** the current processing state
2. **Plans** the next actions using LLM reasoning
3. **Acts** by calling CryoSPARC APIs
4. **Reflects** on results and adapts

---

## 📖 Research Foundation

This agent was designed by studying two key sources:

### 1. SpatialAgent Paper (Memory-Planning-Action Architecture)

We studied the SpatialAgent paper to understand how to build an autonomous AI agent:

| Concept | What We Learned | How We Applied It |
|---------|-----------------|-------------------|
| **Memory Module** | Agents need both domain knowledge (semantic) and session state (episodic) | Created `memory.py` with `SemanticMemory` (GPCR knowledge, quality thresholds) and `EpisodicMemory` (job history, observations) |
| **Planning Module** | LLMs can do chain-of-thought reasoning to decide next steps | Created `planner.py` with structured prompts that output JSON actions |
| **Action Module** | Agents need tools to interact with external systems | Created `tools/cryosparc.py` to wrap CryoSPARC API, `tools/databases.py` for EMPIAR/PDB/UniProt |
| **Playbooks** | Pre-defined workflow templates guide the agent | Created `playbooks/gpcr.py` with GPCR-specific workflow steps |

### 2. CryoSPARC Paper (Automated Cryo-EM Workflows)

We studied CryoSPARC's automated workflows to understand the processing pipeline:

| Concept | What We Learned | How We Applied It |
|---------|-----------------|-------------------|
| **Workflow Pipeline** | Import → Motion → CTF → Denoise → Pick → 2D → 3D → Refine | Implemented 11-step GPCR pipeline in `GPCRPlaybook` class |
| **Quality Assessment** | Junk Detector, CTF metrics, resolution tracking | Created `tools/quality.py` with CTF, particle, and resolution assessment |
| **Config Hierarchy** | Dataset-level → Class-level → Workflow-level parameters | Implemented in `config.py` with `ProcessingDefaults` and `GPCRParameters` |
| **Key Tools** | Blob Picker, RBAS 2D/3D, Heterogeneous Refinement | Mapped to CryoSPARC job types in `JOB_TYPE_MAP` |

### How We Combined Them

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH → IMPLEMENTATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SpatialAgent Paper                    CryoSPARC Paper                     │
│   ┌─────────────────────┐              ┌─────────────────────┐             │
│   │ • Memory Module     │              │ • Import Movies     │             │
│   │ • Planning Module   │              │ • Motion Correction │             │
│   │ • Action Module     │              │ • CTF Estimation    │             │
│   │ • Playbook Templates│              │ • Particle Picking  │             │
│   └──────────┬──────────┘              │ • 2D Classification │             │
│              │                         │ • 3D Ab-initio      │             │
│              │                         │ • 3D Refinement     │             │
│              │                         │ • Quality Metrics   │             │
│              │                         └──────────┬──────────┘             │
│              │                                    │                         │
│              └──────────────┬─────────────────────┘                         │
│                             ▼                                               │
│              ┌──────────────────────────────┐                               │
│              │       CryoEMAgent            │                               │
│              ├──────────────────────────────┤                               │
│              │ Memory:                      │                               │
│              │   • SemanticMemory (GPCR)    │  ← Domain knowledge           │
│              │   • EpisodicMemory (jobs)    │  ← Session tracking           │
│              │                              │                               │
│              │ Planner:                     │                               │
│              │   • LLM chain-of-thought     │  ← Decision making            │
│              │   • GPCR playbook templates  │  ← Workflow guidance          │
│              │                              │                               │
│              │ Tools:                       │                               │
│              │   • CryoSPARC API wrapper    │  ← GPU processing             │
│              │   • Quality assessment       │  ← Validation                 │
│              │   • Database connectors      │  ← External data              │
│              └──────────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Research Artifacts

The following research artifacts document our analysis:

| Document | Purpose |
|----------|---------|
| `implementation_plan.md` | Detailed design derived from paper analysis |
| `cryosparc_study_report.md` | Deep-dive into CryoSPARC algorithms |
| `cryosparc_tools_analysis.md` | Analysis of the Python API structure |


## 🏗️ Architecture

### Memory-Planning-Action Framework

```python
# Inspired by SpatialAgent (Yu et al., 2024)

class CryoEMAgent:
    def run(self, movies_path):
        # 1. OBSERVE: Understand current state
        self.memory.initialize(project, workspace, objective)
        
        # 2. PLAN: LLM generates processing steps
        plan = self.planner.plan_next_step(self.memory)
        
        # 3. ACT: Execute CryoSPARC jobs
        for action in plan.actions:
            result = self.tools.execute(action)
            
            # 4. REFLECT: Check quality, adapt if needed
            self.memory.add_observation(result)
            if self.quality.assess(result).score < threshold:
                plan = self.planner.replan(self.memory)
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Agent** | `core/agent.py` | Main orchestration loop |
| **Memory** | `core/memory.py` | Semantic (domain knowledge) + Episodic (session state) |
| **Planner** | `core/planner.py` | LLM-based chain-of-thought reasoning |
| **CryoSPARC Tools** | `tools/cryosparc.py` | API wrapper for job management |
| **Quality** | `tools/quality.py` | CTF, particle, resolution metrics |
| **Playbooks** | `playbooks/gpcr.py` | GPCR-specific workflow templates |

---

## 📋 GPCR Processing Pipeline

The agent executes an **11-step workflow** optimized for GPCR membrane proteins:

```
Step 1: Import Movies           → 1,000 movies
Step 2: Motion Correction       → 1,000 micrographs
Step 3: CTF Estimation          → 980 micrographs (quality filtered)
Step 4: Particle Picking        → 150,000 particles
Step 5: Extract Particles       → 150,000 extracted particles
Step 6: 2D Classification       → 120,000 particles + 50 class averages
Step 7: Select 2D Classes       → 80,000 good particles
Step 8: Ab-initio 3D            → Initial 3D volume at 8.5Å
Step 9: Heterogeneous Refine    → 60,000 particles at 5.2Å
Step 10: Homogeneous Refine     → 55,000 particles at 4.1Å
Step 11: Non-uniform Refine     → 52,000 particles at 3.2Å ✓
```

### GPCR-Specific Parameters
- **Box size**: 256 pixels
- **Particle diameter**: 80-150 Å
- **Symmetry**: C1 (no symmetry)
- **Target resolution**: 3.5 Å

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/CryoEMAgent.git
cd CryoEMAgent

# Install the package
pip install -e .

# Verify installation
cryoem-agent version
```

### Requirements
- Python 3.9+
- CryoSPARC v4.5+ (for real processing)
- OpenAI API key (for LLM planning)

---

## ⚙️ Configuration

Create a `.env` file or set environment variables:

```bash
# CryoSPARC Connection (required for real processing)
export CRYOSPARC_URL="http://localhost:39000"
export CRYOSPARC_EMAIL="your@email.com"
export CRYOSPARC_PASSWORD="your_password"

# LLM Configuration (required)
export OPENAI_API_KEY="sk-..."
```

### Why These Are Needed

| Variable | Purpose |
|----------|---------|
| `CRYOSPARC_URL` | The CryoSPARC server runs the actual GPU algorithms (motion correction, 3D reconstruction, etc.) |
| `OPENAI_API_KEY` | The LLM provides intelligent planning and decision-making |

**CryoEMAgent = LLM Brain + CryoSPARC Muscle**

---

## 💻 Usage

### Command Line

```bash
# Run full GPCR workflow
cryoem-agent run \
    --project P3 \
    --workspace W1 \
    --movies "/data/gpcr_dataset/*.mrc" \
    --pixel-size 1.05 \
    --voltage 300 \
    --target-resolution 3.5

# Check CryoSPARC connection
cryoem-agent status

# List jobs in a project
cryoem-agent jobs --project P3
```

### Python API

```python
from cryoemagent import CryoEMAgent

# Initialize agent
agent = CryoEMAgent()

# Run end-to-end workflow
result = agent.run(
    project_uid="P3",
    workspace_uid="W1",
    movies_path="/data/gpcr_dataset/*.mrc",
    pixel_size=1.05,
    voltage=300,
    target_resolution=3.5,
)

# Check results
if result.success:
    print(f"Resolution: {result.resolution} Å")
    print(f"Particles: {result.particle_count:,}")
else:
    print(f"Error: {result.error}")
```

### Mock Mode (Testing Without CryoSPARC)

```bash
# Run demo with simulated CryoSPARC
python examples/run_mock_demo.py
```

This demonstrates the complete workflow using:
- ✅ Real LLM planning (OpenAI API)
- ✅ Simulated CryoSPARC jobs (no GPU required)

---

## 📁 Project Structure

```
CryoEMAgent/
├── cryoemagent/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   ├── cli.py                # Command-line interface
│   ├── utils.py              # Helper functions
│   │
│   ├── core/
│   │   ├── agent.py          # Main agent (Observe-Plan-Act-Reflect)
│   │   ├── memory.py         # Semantic + Episodic memory
│   │   └── planner.py        # LLM-based planning
│   │
│   ├── tools/
│   │   ├── base.py           # Tool interface
│   │   ├── cryosparc.py      # CryoSPARC API wrapper
│   │   ├── mock_cryosparc.py # Mock for testing
│   │   ├── quality.py        # Quality metrics
│   │   └── databases.py      # EMPIAR/PDB connectors
│   │
│   └── playbooks/
│       └── gpcr.py           # GPCR workflow definitions
│
├── playbooks/
│   └── gpcr_standard.yaml    # YAML workflow template
│
├── examples/
│   ├── demo.py               # Basic usage examples
│   └── run_mock_demo.py      # Full mock workflow demo
│
├── tests/
│   ├── test_agent.py
│   ├── test_memory.py
│   ├── test_planner.py
│   └── test_tools.py
│
├── requirements.txt
├── setup.py
├── pytest.ini
└── .env.example
```

---

## 🧠 How the LLM Planning Works

The agent uses GPT-4 with a specialized system prompt:

```
SYSTEM PROMPT (Summarized):
You are an expert cryo-EM scientist. Given the current processing state,
decide the next action:

- CREATE_JOB: Create a new CryoSPARC job
- SET_PARAM: Configure job parameters
- QUEUE_JOB: Submit job for GPU processing
- WAIT_JOB: Wait for job completion
- ASSESS_QUALITY: Check metrics
- FINISH: Complete workflow

Output JSON with action_type, parameters, and reasoning.
```

### Example LLM Response

```json
{
  "goal": "Continue GPCR structure determination",
  "actions": [
    {
      "action_type": "create_job",
      "parameters": {
        "job_type": "patch_motion_correction",
        "params": {"do_dose_weighting": true}
      },
      "reasoning": "Movies imported successfully. Next step is motion correction to align frames and reduce beam-induced motion."
    }
  ]
}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_memory.py -v

# Run with coverage
pytest tests/ --cov=cryoemagent
```

---

## 📊 Demo Results

Running `python examples/run_mock_demo.py` produces:

```
┌─────────────────────── Pipeline Summary ───────────────────────┐
│ ✓ WORKFLOW COMPLETED SUCCESSFULLY                              │
│                                                                │
│ Final Resolution: 3.2 Å                                        │
│ Final Particles: 52,000                                        │
│ Total Jobs: 11                                                 │
│ Target Resolution: 3.5 Å (EXCEEDED!)                           │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Dependencies

| Package | Purpose |
|---------|---------|
| `cryosparc-tools` | Official CryoSPARC Python API |
| `openai` | LLM integration for planning |
| `rich` | Beautiful CLI output |
| `pyyaml` | Playbook configuration |
| `numpy` | Numerical operations |

---

## 📚 References

- [CryoSPARC Paper](https://www.nature.com/articles/nmeth.4169) - Punjani et al., 2017
- [SpatialAgent Paper](https://arxiv.org/abs/2401.01724) - Memory-Planning-Action architecture
- [cryosparc-tools Documentation](https://tools.cryosparc.com/)

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

**Built for autonomous GPCR structure determination** 🧬
