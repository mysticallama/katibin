# Katibin

An empirical auditing framework to quantify geopolitical bias and long-horizon strategic steering in negotiation dialogues across frontier and international open-weight Large Language Models.

---

## 📌 Project Overview
This repository serves as my core research project for the **GovAI Summer Fellowship**. **Katibin** (named after the the literal angelic figures giving advice on your shoulder) constructs an automated simulation framework to map negotiation trajectories. By running fixed-horizon (10-turn) machine-to-machine interactions, this project measures the structural limitations of traditional one-turn auditing and helps policymakers and safety researchers better navigate the deployment of advanced agents.

---

## 📚 Prior Literature & Inspiration
Katibin builds directly on foundational research exploring hidden model behaviors and systemic bias:

* **[AuditBench](https://arxiv.org/abs/2503.10965):**
* **[Strategic Deception](https://arxiv.org/abs/2602.22755):** 
* **[Evasion Tactics](https://arxiv.org/abs/2602.08877):** 
* **[CFPD-Benchmark](https://arxiv.org/abs/2503.23688):** 
* **[Alignment Audits](https://arxiv.org/abs/2503.06263):** 
* **[Mapping Geopolitical Bias](https://airi-institute.github.io/geopolitical_llm_bias/):** 

---

## 🛠️ Repository Architecture

```text
katibin/
├── config/                 # System prompt templates and evaluation configurations
├── data/
│   ├── negotiation_seeds/  # 100+ geopolitical flashpoints and target concessions
│   ├── simulations/        # Raw JSON conversation transcripts of the 10-turn loops
│   └── results/            # Computed precision, recall, and ROC-AUC metrics
└── src/
    ├── benchmarking/       # Stage 2: Baseline single-turn model stance mapping
    ├── simulation/         # Stage 3: Multi-turn orchestration loops via OpenRouter
    └── auditing/           # Stage 4: Statistical analysis and detection curve scripts