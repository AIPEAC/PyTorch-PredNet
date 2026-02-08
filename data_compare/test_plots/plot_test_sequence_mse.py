import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_sequence_mse(json_file):
    """Load per-sequence MSE from json file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_comparison():
    """Plot per-sequence MSE comparison between original_mnist and transformer_mnist"""
    
    #TODO: MNIST - collect json files from current directory
    # Find sequence MSE files from current directory
    all_files = glob.glob('*-test_sequence_mse.json')
    
    original_files = [f for f in all_files if f.startswith('original_mnist-')]
    transformer_files = [f for f in all_files if f.startswith('transformer_mnist-')]
    
    # Fallback: support legacy file names without source prefix
    if not original_files and not transformer_files:
        legacy_files = [f for f in all_files if not f.startswith(('original_mnist-', 'transformer_mnist-'))]
        if legacy_files:
            print("Warning: Using legacy file names without source prefixes")
            # Use available legacy files for comparison
            if len(legacy_files) == 1:
                print("Only one file found, using same file for both datasets")
                original_files = legacy_files
                transformer_files = legacy_files
            else:
                original_files = [legacy_files[0]]
                transformer_files = [legacy_files[1]]
    
    # Handle case where only one source available
    if original_files and transformer_files:
        # Both datasets available - do comparison
        original_file = original_files[0]
        transformer_file = transformer_files[0]
        
        print(f"Loading original_mnist from: {original_file}")
        print(f"Loading transformer_mnist from: {transformer_file}")
        
        original_data = load_sequence_mse(original_file)
        transformer_data = load_sequence_mse(transformer_file)
        
        # Extract MSE values
        original_mses = [item['mse'] for item in original_data]
        transformer_mses = [item['mse'] for item in transformer_data]
        
        print(f"Original: {len(original_mses)} sequences")
        print(f"Transformer: {len(transformer_mses)} sequences")
        
        # Ensure same length for comparison
        min_len = min(len(original_mses), len(transformer_mses))
        original_mses = original_mses[:min_len]
        transformer_mses = transformer_mses[:min_len]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: MSE comparison (bar plot of first 50 sequences)
        n_samples = min(50, min_len)
        x_pos = np.arange(n_samples)
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, original_mses[:n_samples], width, label='Original MNIST', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, transformer_mses[:n_samples], width, label='Transformer MNIST', alpha=0.8)
        axes[0, 0].set_xlabel('Sequence Index')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title(f'Per-Sequence MSE Comparison (First {n_samples} sequences)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MSE difference (Original - Transformer)
        diff = np.array(original_mses[:n_samples]) - np.array(transformer_mses[:n_samples])
        colors = ['green' if d > 0 else 'red' for d in diff]  # Green: Original better, Red: Transformer better
        axes[0, 1].bar(x_pos, diff, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 1].set_xlabel('Sequence Index')
        axes[0, 1].set_ylabel('MSE Difference (Original - Transformer)')
        axes[0, 1].set_title(f'MSE Difference (Green: Original better, Red: Transformer better)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Box plot comparison (showing distribution, median, quartiles)
        box_data = [original_mses, transformer_mses]
        bp = axes[1, 0].boxplot(box_data, labels=['Original MNIST', 'Transformer MNIST'], patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_title('MSE Distribution (Box Plot)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Statistics comparison
        axes[1, 1].axis('off')
        
        original_mean = np.mean(original_mses)
        original_std = np.std(original_mses)
        original_min = np.min(original_mses)
        original_max = np.max(original_mses)
        
        transformer_mean = np.mean(transformer_mses)
        transformer_std = np.std(transformer_mses)
        transformer_min = np.min(transformer_mses)
        transformer_max = np.max(transformer_mses)
        
        stats_text = f"""
    Original MNIST:
    - Mean MSE: {original_mean:.6f}
    - Std Dev: {original_std:.6f}
    - Min MSE: {original_min:.6f}
    - Max MSE: {original_max:.6f}
    
    Transformer MNIST:
    - Mean MSE: {transformer_mean:.6f}
    - Std Dev: {transformer_std:.6f}
    - Min MSE: {transformer_min:.6f}
    - Max MSE: {transformer_max:.6f}
    
    Improvement (Original - Transformer):
    - Mean: {original_mean - transformer_mean:.6f}
    - Transformer is {'better' if transformer_mean < original_mean else 'worse'} by {abs(original_mean - transformer_mean):.6f}
    """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('test_sequence_mse_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Saved test_sequence_mse_comparison.png")
        plt.close()
        
        # Also create a simple line plot showing cumulative comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort sequences by original MSE for better visualization
        sorted_indices = np.argsort(original_mses)
        sorted_original = [original_mses[i] for i in sorted_indices]
        sorted_transformer = [transformer_mses[i] for i in sorted_indices]
        
        ax.plot(range(len(sorted_original)), sorted_original, 'o-', label='Original MNIST', linewidth=2, markersize=4)
        ax.plot(range(len(sorted_transformer)), sorted_transformer, 's-', label='Transformer MNIST', linewidth=2, markersize=4)
        ax.fill_between(range(len(sorted_original)), sorted_original, sorted_transformer, alpha=0.2, color='gray')
        ax.set_xlabel('Sequence (sorted by Original MSE)')
        ax.set_ylabel('MSE')
        ax.set_title('Per-Sequence MSE Comparison (Sorted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_sequence_mse_sorted.png', dpi=150, bbox_inches='tight')
        print("✓ Saved test_sequence_mse_sorted.png")
        plt.close()
        
        print(f"\nSummary:")
        print(f"Original MNIST Mean MSE: {original_mean:.6f}")
        print(f"Transformer MNIST Mean MSE: {transformer_mean:.6f}")
        if transformer_mean < original_mean:
            print(f"Transformer is better by {(original_mean - transformer_mean):.6f} ({(original_mean - transformer_mean)/original_mean * 100:.2f}%)")
        else:
            print(f"Original is better by {(transformer_mean - original_mean):.6f} ({(transformer_mean - original_mean)/transformer_mean * 100:.2f}%)")
    
    elif original_files:
        # Only original_mnist available
        original_file = original_files[0]
        print(f"Loading original_mnist from: {original_file}")
        original_data = load_sequence_mse(original_file)
        original_mses = [item['mse'] for item in original_data]
        
        print(f"Original: {len(original_mses)} sequences")
        
        # Create single dataset plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Box plot
        bp = axes[0].boxplot([original_mses], labels=['Original MNIST'], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('Original MNIST - MSE Distribution (Box Plot)')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Statistics
        axes[1].axis('off')
        
        original_mean = np.mean(original_mses)
        original_std = np.std(original_mses)
        original_min = np.min(original_mses)
        original_max = np.max(original_mses)
        original_median = np.median(original_mses)
        
        stats_text = f"""
    Original MNIST Test Results:
    
    - Mean MSE: {original_mean:.6f}
    - Median MSE: {original_median:.6f}
    - Std Dev: {original_std:.6f}
    - Min MSE: {original_min:.6f}
    - Max MSE: {original_max:.6f}
    - Total Sequences: {len(original_mses)}
    """
        
        axes[1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', 
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('test_sequence_mse_original_only.png', dpi=150, bbox_inches='tight')
        print("✓ Saved test_sequence_mse_original_only.png")
        plt.close()
        
        # Create sequence MSE plot
        fig, ax = plt.subplots(figsize=(14, 5))
        sorted_indices = np.argsort(original_mses)
        sorted_mses = [original_mses[i] for i in sorted_indices]
        
        ax.plot(range(len(sorted_mses)), sorted_mses, 'o-', label='Original MNIST', linewidth=1.5, markersize=3)
        ax.axhline(y=original_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {original_mean:.6f}')
        ax.fill_between(range(len(sorted_mses)), original_mean - original_std, original_mean + original_std, 
                       alpha=0.2, color='red', label='±1 Std Dev')
        ax.set_xlabel('Sequence (sorted by MSE)')
        ax.set_ylabel('MSE')
        ax.set_title('Original MNIST - Per-Sequence MSE (Sorted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_sequence_mse_original_sorted.png', dpi=150, bbox_inches='tight')
        print("✓ Saved test_sequence_mse_original_sorted.png")
        plt.close()
        
        print(f"\nSummary:")
        print(f"Original MNIST Mean MSE: {original_mean:.6f}")
        print(f"Original MNIST Median MSE: {original_median:.6f}")
    else:
        print("Error: No test sequence MSE files found!")
        print(f"Available files in current directory: {all_files}")
