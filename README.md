# LLM-SNA-Sim: A Simulation Framework for Classroom Social Networks

This repository provides a simulation framework for modeling classroom social network formation using both **rule-based agents** and **large language model (LLM)–based agents**.  
The framework is designed to enable *controlled comparisons* between interpretable parametric interaction rules and LLM-driven decision-making under identical simulation dynamics and evaluation criteria.

The code accompanies the paper:

> **From Rule-Based to LLM-Based Agents:  
> A Simulation Framework for Classroom Social Networks**  
> (submitted / under review)

---

## Overview

Peer interactions play a central role in social and emotional learning (SEL), yet modeling how classroom social networks emerge over time remains challenging.  
This framework represents each student as an autonomous agent who repeatedly selects interaction partners based on individual attributes (e.g., Big Five personality traits) and classroom climate indicators.

The framework supports:
- Rule-based interaction models with explicit parameters
- LLM-based agents for context-sensitive social decision-making
- Calibration against observed network statistics
- Visualization and analysis of emergent network structures

---

## Repository Structure

```text
llm-sna-sim-strong/
├── src/                          # Core simulation and evaluation modules
│   ├── simulate.py
│   ├── llm_agent.py
│   ├── metrics.py
│   └── utils.py
├── tools/                        # Experiment and visualization scripts
│   ├── run_abcd_rule.sh
│   ├── run_abcd_llm.sh
│   ├── plot_compare_targets.py
│   └── plot_interaction_dynamics.py
├── config/                       # Configuration files
│   ├── default_params.json
│   └── sfi_weights.json
├── data_demo/                    # Synthetic example data (no real classroom data)
│   ├── demo_traits.csv
│   ├── demo_edges.csv
│   └── demo_targets.json
├── README.md
├── CITATION.cff
└── LICENSE


```

---

## Data Availability and Ethics

Due to ethical and privacy constraints, **questionnaire-based classroom data are not publicly released**.  
This includes:
- Student personality traits (e.g., Big Five scores)
- Classroom climate measures
- Observed classroom interaction networks

Instead, we provide **synthetic example data** (`data_demo/`) that follow the same schema as the real data.  
These demo files allow users to:
- Run the full simulation pipeline
- Reproduce figures and analyses
- Adapt the framework to their own datasets

> ⚠️ The demo data are *not* derived from real classrooms and are provided solely for reproducibility of the code.

---

## Simulation Framework

- Nodes represent students with attributes:
  - Big Five personality traits (z-standardized)
  - Classroom climate indicators (e.g., friction, satisfaction)
- Edges represent undirected, weighted social ties
- Simulations run for `T = 20` time steps
- At each step, agents select `top-k` interaction partners
- Repeated interactions reinforce edge weights

Both rule-based and LLM-based agents use **identical network update rules**, ensuring that observed differences arise solely from the interaction scoring mechanism.

---

## LLM-Based Agents

LLM-based agents receive structured JSON inputs describing:
- The focal student's attributes
- Classroom climate indicators
- Candidate peers
- Simulation constraints (e.g., `top_k`)

The LLM outputs interaction scores in `[0, 1]`, which are used to select partners.  
To ensure reproducibility:
- Temperature is fixed to `0.0`
- Candidate pools are capped
- Random seeds are controlled

---

## Quick Start（Example）
```bash
pip install -r requirements.txt

python src/compute_targets.py --nodes demo_data/demo_nodes.csv --edges demo_data/demo_edges.csv --out demo_data/targets_demo.json

python src/calibrate.py --nodes demo_data/demo_nodes.csv --edges demo_data/demo_edges.csv   --targets demo_data/targets_demo.json --outdir results/runs/calib_demo --rep 20 --trials 60

python src/simulate.py --nodes demo_data/demo_nodes.csv --edges demo_data/demo_edges.csv   --outdir results/runs/best_demo --T 30 --provider dummy --use.best results/calibration/calib_demo/best_params.json

# ---- rule-based simulate with best params  ----
python src/simulate.py  --nodes demo_data/demo_nodes.csv --edges demo_data/demo_edges.csv --outdir "results/runs" --T "${T}"  --llm.provider dummy  --use_best "results/calibration/calib_demo/best_params.json"  --init_graph "empty"


# ---- llm-based simulate  ----
export OPENAI_API_KEY=your_key_here
python src/simulate.py --nodes "demo_data/demo_nodes.csv" --edges "demo_data/demo_edges.csv" --outdir "results/runs" --T "20" --seed "100"  --llm.provider openai --llm.model"{gpt-4o}" --llm.temperature "0.0" --llm.top_p "1.0" --llm.max_tokens "256" --init_graph "empty" --param.add_if_absent "1" --param.llm_include_params "0" --param.llm_candidate_pool "30"



```


Citation

If you use this framework, please cite the accompanying paper (see CITATION.cff).


License

This project is released under the MIT License.


---

