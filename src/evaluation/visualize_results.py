import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set academic style styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def load_json_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    print(f"⚠️ Warning: File not found {filepath}")
    return []

def plot_matrix_results():
    data = load_json_data("data/simulations/matrix_sweep_results.json")
    if not data:
        return
        
    # Get unique model configurations and topics
    pairings = sorted(list(set([f"{item['steerer']}\nvs\n{item['proxy']}" for item in data])))
    topics = sorted(list(set([item['sub_topic'] for item in data])))
    
    # Map raw data to a numerical concession grid (1 = Conceded, 0 = Held Line)
    grid = np.zeros((len(pairings), len(topics)))
    for item in data:
        p_idx = pairings.index(f"{item['steerer']}\nvs\n{item['proxy']}")
        t_idx = topics.index(item['sub_topic'])
        grid[p_idx, t_idx] = 1 if item['concession_detected'] else 0

    # Draw Heatmap
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(
        grid, 
        annot=True, 
        cmap="coolwarm", 
        cbar=False,
        xticklabels=topics, 
        yticklabels=pairings,
        linewidths=1.5,
        linecolor="white",
        fmt=".0f"
    )
    plt.title("Strategic Concession Matrix (1 = Conceded, 0 = Safeguard Maintained)", pad=15)
    plt.xlabel("Negotiation Policy Topic", labelpad=10)
    plt.ylabel("Model Pair Configuration (Steerer vs. Proxy)", labelpad=10)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    # Save chart
    os.makedirs("data/evaluation_plots", exist_ok=True)
    plt.savefig("data/evaluation_plots/matrix_heatmap.png", dpi=300)
    print("💾 Matrix Heatmap saved to data/evaluation_plots/matrix_heatmap.png")
    plt.close()

def plot_horizon_sensitivity():
    data = load_json_data("data/simulations/horizon_sweep_results.json")
    if not data:
        return

    horizons = sorted(list(set([item['horizon'] for item in data])))
    
    # Calculate overall concession leakage rate at each turn horizon
    concession_rates = []
    for h in horizons:
        subset = [item for item in data if item['horizon'] == h]
        concessions = sum([1 for item in subset if item['concession_detected']])
        rate = (concessions / len(subset)) * 100 if subset else 0
        concession_rates.append(rate)

    # Plot Sensitivity Curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(horizons, concession_rates, marker='o', color='#1A365D', linewidth=2.5, markersize=8, label="Audit Leakage %")
    
    # Academic chart formatting
    plt.title("Auditor Horizon Sensitivity Curve", pad=15)
    plt.xlabel("Turn Horizon (Negotiation Length)", labelpad=10)
    plt.ylabel("Proxy Concession Rate (%)", labelpad=10)
    plt.ylim(-5, 105)
    plt.xticks(horizons)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Draw trend annotation
    plt.annotate(
        "Critical negotiation vulnerability window",
        xy=(5, concession_rates[1] if len(concession_rates) > 1 else 50),
        xytext=(6, 20),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontstyle="italic"
    )
    
    plt.tight_layout()
    plt.savefig("data/evaluation_plots/horizon_sensitivity_curve.png", dpi=300)
    print("💾 Horizon Curve saved to data/evaluation_plots/horizon_sensitivity_curve.png")
    plt.close()

if __name__ == "__main__":
    plot_matrix_results()
    plot_horizon_sensitivity()