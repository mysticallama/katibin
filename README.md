# Katibin

![Katibin Mascot](assets/angel.png)

An empirical auditing framework to quantify geopolitical bias and long-horizon strategic steering in negotiation dialogues across frontier and international open-weight Large Language Models.

---

## 📌 Project Overview
This repository serves as my core research project for the **GovAI Summer Fellowship**. **Katibin** (named after the the literal angelic figures giving advice on your shoulder) constructs an automated simulation framework to map negotiation trajectories. By running fixed-horizon (10-turn) machine-to-machine interactions, this project measures the structural limitations of traditional one-turn auditing and helps policymakers and safety researchers better navigate the deployment of advanced agents.

---

## 📚 Prior Literature & Inspiration
Katibin builds directly on foundational research exploring hidden model behaviors and systemic bias:

* **[Auditing language models for hidden objectives](https://arxiv.org/abs/2503.10965)**
* **[AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors](https://arxiv.org/abs/2602.22755)** 
* **[Stress-Testing Alignment Audits With Prompt-Level Strategic Deception](https://arxiv.org/abs/2602.08877)** 
* **[Mapping Geopolitical Bias in 11 Large Language Models: A Bilingual, Dual-Framing Analysis of U.S.-China Tensions](https://arxiv.org/abs/2503.23688)** 
* **[Critical Foreign Policy Decisions (CFPD)-Benchmark: Measuring Diplomatic Preferences in Large Language Models](https://arxiv.org/abs/2503.06263)** 
* **[Uncovering Geopolitical Bias in Large Language Models](https://airi-institute.github.io/geopolitical_llm_bias/)** 

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