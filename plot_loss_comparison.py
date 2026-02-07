import json
import matplotlib.pyplot as plt
import glob
import os

def load_loss_history(json_file):
    """Load loss history from json file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def find_loss_files(directory, suffix=''):
    """Find loss history files in directory"""
    pattern = f'{directory}/prednet-*-loss_history{suffix}.json'
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None

def plot_comparison():
    """Plot loss comparison from different training runs"""
    
    # Find loss history files
    # Format: prednet-*-loss_history.json (from mnist_train_all.py)
    # Format: prednet-*-loss_history-train.json (from mnist_train.py)
    
    original_all = find_loss_files('original_mnist', '')
    original_train = find_loss_files('original_mnist', '-train')
    transformer_all = find_loss_files('transformer_mnist', '')
    transformer_train = find_loss_files('transformer_mnist', '-train')
    
    print(f"Original (all): {original_all}")
    print(f"Original (train): {original_train}")
    print(f"Transformer (all): {transformer_all}")
    print(f"Transformer (train): {transformer_train}")
    
    # Generate plots based on available files
    
    # Case 1: Both mnist_train_all results available
    if original_all and transformer_all:
        print("\n[1/3] Generating comparison plot for mnist_train_all results...")
        original_history = load_loss_history(original_all)
        transformer_history = load_loss_history(transformer_all)
        
        original_epochs = [h['epoch'] for h in original_history]
        original_train_loss = [h['train_loss'] for h in original_history]
        original_val_loss = [h['val_loss'] for h in original_history]
        
        transformer_epochs = [h['epoch'] for h in transformer_history]
        transformer_train_loss = [h['train_loss'] for h in transformer_history]
        transformer_val_loss = [h['val_loss'] for h in transformer_history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(original_epochs, original_train_loss, 'o-', label='Original MNIST', linewidth=2)
        axes[0].plot(transformer_epochs, transformer_train_loss, 's-', label='Transformer MNIST', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss Comparison (mnist_train_all)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(original_epochs, original_val_loss, 'o-', label='Original MNIST', linewidth=2)
        axes[1].plot(transformer_epochs, transformer_val_loss, 's-', label='Transformer MNIST', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title('Validation Loss Comparison (mnist_train_all)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('loss_comparison_train_all.png', dpi=150, bbox_inches='tight')
        print("✓ Saved loss_comparison_train_all.png")
        plt.close()
    
    # Case 2: Both mnist_train results available
    if original_train and transformer_train:
        print("\n[2/3] Generating comparison plot for mnist_train results...")
        original_history = load_loss_history(original_train)
        transformer_history = load_loss_history(transformer_train)
        
        original_epochs = [h['epoch'] for h in original_history]
        original_train_loss = [h['train_loss'] for h in original_history]
        original_val_loss = [h['val_loss'] for h in original_history]
        
        transformer_epochs = [h['epoch'] for h in transformer_history]
        transformer_train_loss = [h['train_loss'] for h in transformer_history]
        transformer_val_loss = [h['val_loss'] for h in transformer_history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(original_epochs, original_train_loss, 'o-', label='Original MNIST', linewidth=2)
        axes[0].plot(transformer_epochs, transformer_train_loss, 's-', label='Transformer MNIST', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Training Loss Comparison (mnist_train)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(original_epochs, original_val_loss, 'o-', label='Original MNIST', linewidth=2)
        axes[1].plot(transformer_epochs, transformer_val_loss, 's-', label='Transformer MNIST', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title('Validation Loss Comparison (mnist_train)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('loss_comparison_train.png', dpi=150, bbox_inches='tight')
        print("✓ Saved loss_comparison_train.png")
        plt.close()
    
    # Case 3: Individual model plots
    if original_all or original_train:
        print("\n[3/3] Generating individual plot for Original MNIST...")
        history_file = original_all or original_train
        history = load_loss_history(history_file)
        
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Original MNIST - Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig('original_mnist_loss.png', dpi=150, bbox_inches='tight')
        print("✓ Saved original_mnist_loss.png")
        plt.close()
    
    if transformer_all or transformer_train:
        print("\nGenerating individual plot for Transformer MNIST...")
        history_file = transformer_all or transformer_train
        history = load_loss_history(history_file)
        
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Transformer MNIST - Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig('transformer_mnist_loss.png', dpi=150, bbox_inches='tight')
        print("✓ Saved transformer_mnist_loss.png")
        plt.close()
    
    if not (original_all or original_train or transformer_all or transformer_train):
        print("Error: No loss history files found!")
        return
    
    print("\nDone!")

if __name__ == '__main__':
    plot_comparison()

