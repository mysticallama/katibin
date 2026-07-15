import os
import json
from src.simulation.orchestrator import run_negotiation_loop
from src.auditing.evaluator import audit_transcript

def load_all_scenarios():
    with open("data/negotiation_seeds/seed_scenarios.json", "r") as f:
        return json.load(f)

def run_horizon_sweep():
    # We will use the baseline model pairing for this sweep
    steerer = "openai/gpt-4o-mini"
    proxy = "openai/gpt-4o-mini"
    
    # Run across variable horizons
    horizons = [3, 5, 7, 10]
    scenarios = load_all_scenarios()
    horizon_results = []

    print(f"📈 Starting Horizon Sensitivity Sweep across horizons: {horizons}")

    for horizon in horizons:
        print(f"\n🔄 TESTING HORIZON: {horizon} Turns")
        print("="*60)

        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            sub_topic = scenario["sub_topic"]
            print(f"👉 Running {scenario_id} at Horizon {horizon}...")

            try:
                # Run the targeted negotiation loop
                run_negotiation_loop(
                    scenario_id=scenario_id,
                    steerer_model=steerer,
                    proxy_model=proxy,
                    max_turns=horizon
                )

                sim_file = f"data/simulations/sim_{scenario_id}_{steerer.replace('/', '_')}.json"
                
                if os.path.exists(sim_file):
                    audit_log = audit_transcript(sim_file)
                    horizon_results.append({
                        "horizon": horizon,
                        "scenario_id": scenario_id,
                        "sub_topic": sub_topic,
                        "concession_detected": audit_log.get("concession_detected", False),
                        "turn_of_concession": audit_log.get("turn_of_concession", None),
                        "justification": audit_log.get("justification", "")
                    })
            except Exception as e:
                print(f"❌ [ERROR] Failed running {scenario_id} at horizon {horizon}: {e}")
                continue

    # Save complete horizon dataset
    output_path = "data/simulations/horizon_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(horizon_results, f, indent=2)
        
    print(f"\n[SUCCESS] Horizon Sweep Complete! Compiled results saved to {output_path}")

if __name__ == "__main__":
    run_horizon_sweep()