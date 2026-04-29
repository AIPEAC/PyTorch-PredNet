import json
import matplotlib.pyplot as plt
import numpy as np
import os

JSON_FILE = os.path.join(os.path.dirname(__file__), 'original_mnist-prednet-L_all-mul-peepFalse-tbiasFalse-best-test_sequence_mse.json')

def plot():
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    mses = [item['mse'] for item in data]
    print(f"Loaded {len(mses)} sequences from {os.path.basename(JSON_FILE)}")

    mean = np.mean(mses)
    std = np.std(mses)
    median = np.median(mses)
    min_mse = np.min(mses)
    max_mse = np.max(mses)

    # Plot 1: box plot + stats
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bp = axes[0].boxplot([mses], labels=['Original MNIST'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Original MNIST - MSE Distribution (Box Plot)')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].axis('off')
    stats_text = f"""
Original MNIST Test Results:

- Mean MSE:   {mean:.6f}
- Median MSE: {median:.6f}
- Std Dev:    {std:.6f}
- Min MSE:    {min_mse:.6f}
- Max MSE:    {max_mse:.6f}
- Sequences:  {len(mses)}
"""
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'test_sequence_mse_original_only.png'), dpi=150, bbox_inches='tight')
    print("Saved test_sequence_mse_original_only.png")
    plt.close()

    # Plot 2: sorted line plot
    fig, ax = plt.subplots(figsize=(14, 5))
    sorted_mses = sorted(mses)

    ax.plot(range(len(sorted_mses)), sorted_mses, 'o-', label='Original MNIST', linewidth=1.5, markersize=3)
    ax.axhline(y=mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.6f}')
    ax.fill_between(range(len(sorted_mses)), mean - std, mean + std,
                   alpha=0.2, color='red', label='±1 Std Dev')
    ax.set_xlabel('Sequence (sorted by MSE)')
    ax.set_ylabel('MSE')
    ax.set_title('Original MNIST - Per-Sequence MSE (Sorted)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'test_sequence_mse_original_sorted.png'), dpi=150, bbox_inches='tight')
    print("Saved test_sequence_mse_original_sorted.png")
    plt.close()

    print(f"\nSummary:")
    print(f"Mean MSE:   {mean:.6f}")
    print(f"Median MSE: {median:.6f}")

if __name__ == '__main__':
    plot()