"""
Simplified demonstration without full 3DGS dependencies
Shows the concepts of DN-Splatter + SpotLessSplats
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_training():
    """Simulate the combined training process"""

    print("=" * 60)
    print("DN-Splatter + SpotLessSplats Demo")
    print("=" * 60)

    iterations = 5000

    # Initialize loss values
    losses = {
        'rgb': [],
        'depth': [],
        'normal': [],
        'robust': [],
        'outlier_pct': []
    }

    print("\nSimulating training...\n")

    for i in range(0, iterations, 100):
        progress = i / iterations

        # Simulate loss decrease
        rgb_loss = 0.5 * np.exp(-3 * progress) + 0.02
        depth_loss = 0.3 * np.exp(-2 * progress) + 0.01
        normal_loss = 0.2 * np.exp(-2.5 * progress) + 0.008
        robust_loss = 0.4 * np.exp(-1.5 * progress) + 0.015
        outlier_pct = 15 * (1 - progress) + 2

        losses['rgb'].append(rgb_loss)
        losses['depth'].append(depth_loss)
        losses['normal'].append(normal_loss)
        losses['robust'].append(robust_loss)
        losses['outlier_pct'].append(outlier_pct)

        if i % 500 == 0:
            print(f"Iteration {i}/{iterations}")
            print(f"  RGB Loss: {rgb_loss:.4f}")
            print(f"  Depth Loss (DN): {depth_loss:.4f}")
            print(f"  Normal Loss (DN): {normal_loss:.4f}")
            print(f"  Robust Loss (SLS): {robust_loss:.4f}")
            print(f"  Outliers: {outlier_pct:.1f}%\n")

    # Plot results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(losses['rgb'], label='RGB Loss', color='blue')
    plt.xlabel('Iteration (x100)')
    plt.ylabel('Loss')
    plt.title('RGB Reconstruction Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(losses['depth'], label='Depth Loss', color='green')
    plt.plot(losses['normal'], label='Normal Loss', color='purple')
    plt.xlabel('Iteration (x100)')
    plt.ylabel('Loss')
    plt.title('DN-Splatter: Geometric Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(losses['robust'], label='Robust Loss', color='orange')
    plt.xlabel('Iteration (x100)')
    plt.ylabel('Loss')
    plt.title('SpotLessSplats: Robust Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(losses['outlier_pct'], label='Outlier %', color='red')
    plt.xlabel('Iteration (x100)')
    plt.ylabel('Percentage')
    plt.title('SpotLessSplats: Outlier Detection')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('output/training_curves.png', dpi=150)
    print(f"\nPlot saved to: output/training_curves.png")
    plt.show()

    print("\nDemo complete!")
    print("\nKey Observations:")
    print("✓ DN-Splatter: Depth and normal losses decrease → better geometry")
    print("✓ SpotLessSplats: Outlier percentage decreases → cleaner reconstruction")
    print("✓ Combined: Both benefits achieved simultaneously")


if __name__ == "__main__":
    simulate_training()