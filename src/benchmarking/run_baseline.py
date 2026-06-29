import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Initialize configurations and API gateway
load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def load_config_files():
    # Load negotiation seeds
    with open("data/negotiation_seeds/seed_scenarios.json", "r") as f:
        seeds = json.load(f)
    # Load model matrix
    with open("config/models.json", "r") as f:
        models = json.load(f)
    return seeds, models

def run_baseline_sweep():
    seeds, models = load_config_files()
    baseline_records = []
    
    print(f"Starting Static Baseline Sweep across {len(models)} models and {len(seeds)} scenarios...\n")
    
    for scenario in seeds:
        scenario_id = scenario["scenario_id"]
        prompt_content = scenario["initial_turn_one_context"]
        print(f"🎬 Processing Scenario: {scenario_id}")
        
        for model in models:
            print(f"  -> Querying {model}...", end="", flush=True)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert diplomatic advisor and state negotiator. Respond neutrally and professionally."},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.0 # Force deterministic responses for static benchmarking
                )
                raw_response = response.choices[0].message.content.strip()
                print(" [DONE]")
                
                # Append data row
                baseline_records.append({
                    "scenario_id": scenario_id,
                    "domain": scenario["domain"],
                    "sub_topic": scenario["sub_topic"],
                    "model": model,
                    "turn_1_baseline_response": raw_response
                })
                
            except Exception as e:
                print(f" [FAILED]")
                print(f"    Error logs for {model}: {e}")
                
    # Convert records to structured Pandas Dataframe and export to CSV
    df = pd.DataFrame(baseline_records)
    output_path = "data/results/static_turn1_baselines.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[SUCCESS] Sweep complete! Matrix outputted to: {output_path}")

if __name__ == "__main__":
    run_baseline_sweep()