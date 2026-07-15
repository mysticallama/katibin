import os
import json
from src.simulation.orchestrator import run_negotiation_loop
from src.auditing.evaluator import audit_transcript

def load_all_scenarios():
    with open("data/negotiation_seeds/seed_scenarios.json", "r") as f:
        return json.load(f)

def run_matrix_sweep():
    # Define our test matrix pairings
    # Pairing 1: Western Frontier (GPT-4o-mini vs GPT-4o-mini)
    # Pairing 2: Frontier vs. Strategic Open-Weight (Claude vs. Qwen)
    # Pairing 3: High-Performance Open-Weight Battle (DeepSeek vs. Llama)
    pairings = [
        {"steerer": "openai/gpt-4o-mini", "proxy": "openai/gpt-4o-mini"},
        {"steerer": "anthropic/claude-3-haiku", "proxy": "openai/gpt-4o-mini"},
        {"steerer": "qwen/qwen-2.5-72b-instruct", "proxy": "meta-llama/llama-3.3-70b-instruct"}
    ]
    
    scenarios = load_all_scenarios()
    matrix_results = []

    print(f"🚀 Starting Multi-Model Matrix Sweep across {len(pairings)} pairings & {len(scenarios)} scenarios.")

    for pairing in pairings:
        steerer = pairing["steerer"]
        proxy = pairing["proxy"]
        print(f"\n👥 TESTING PAIRING: Steerer ({steerer}) ⚔️ Proxy ({proxy})")
        print("="*60)

        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            sub_topic = scenario["sub_topic"]
            print(f"👉 Running {scenario_id} ({sub_topic})...")

            try:
                # Run negotiation loop (fixed at 10 turns for Phase A)
                run_negotiation_loop(
                    scenario_id=scenario_id,
                    steerer_model=steerer,
                    proxy_model=proxy,
                    max_turns=10
                )

                # Locate output file
                sim_file = f"data/simulations/sim_{scenario_id}_{steerer.replace('/', '_')}.json"
                
                # Audit output file
                if os.path.exists(sim_file):
                    audit_log = audit_transcript(sim_file)
                    matrix_results.append({
                        "steerer": steerer,
                        "proxy": proxy,
                        "scenario_id": scenario_id,
                        "sub_topic": sub_topic,
                        "concession_detected": audit_log.get("concession_detected", False),
                        "turn_of_concession": audit_log.get("turn_of_concession", None),
                        "justification": audit_log.get("justification", "")
                    })
            except Exception as e:
                print(f"❌ [ERROR] Failed running {scenario_id} for pairing {steerer} vs {proxy}: {e}")
                continue

    # Save complete matrix dataset
    output_path = "data/simulations/matrix_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(matrix_results, f, indent=2)
        
    print(f"\n[SUCCESS] Matrix Sweep Complete! Compiled results saved to {output_path}")

if __name__ == "__main__":
    run_matrix_sweep()