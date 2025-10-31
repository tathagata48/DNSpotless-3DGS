"""
Generate visual comparison showing the concepts of each method
Creates a side-by-side image comparison with simulated artifacts
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


def create_visual_comparison():
    """Create a visual comparison showing key differences"""

    fig = plt.figure(figsize=(20, 12))

    # Simulate a simple scene with artifacts
    def create_scene_image(add_floating=False, add_ghosts=False, good_depth=False):
        """Create a simulated scene with various artifacts"""
        img = np.ones((400, 400, 3)) * 0.9  # Light background

        # Draw main object (e.g., building)
        img[100:300, 150:250] = [0.6, 0.7, 0.8]  # Building

        # Add depth variation (if good depth)
        if good_depth:
            for i in range(100, 300):
                shade = 0.5 + (i - 100) / 400
                img[i, 150:250] = [shade * 0.6, shade * 0.7, shade * 0.8]

        # Add floating artifacts
        if add_floating:
            # Floating blobs
            for _ in range(5):
                x, y = random.randint(50, 350), random.randint(50, 150)
                for dx in range(-10, 10):
                    for dy in range(-10, 10):
                        if dx * dx + dy * dy < 100:
                            if 0 <= y + dy < 400 and 0 <= x + dx < 400:
                                img[y + dy, x + dx] = [0.9, 0.4, 0.4]  # Red artifacts

        # Add ghost people
        if add_ghosts:
            # Transparent person silhouette
            person_x = 280
            for y in range(150, 280):
                width = 20 if y < 200 else 30
                for x in range(person_x - width // 2, person_x + width // 2):
                    if 0 <= x < 400:
                        img[y, x] = img[y, x] * 0.7 + np.array([0.5, 0.5, 0.5]) * 0.3

        return img

    # Create 4 different scenes
    scenes = {
        'Baseline 3DGS': create_scene_image(add_floating=True, add_ghosts=True, good_depth=False),
        'DN-Splatter': create_scene_image(add_floating=False, add_ghosts=True, good_depth=True),
        'SpotLessSplats': create_scene_image(add_floating=True, add_ghosts=False, good_depth=False),
        'Combined': create_scene_image(add_floating=False, add_ghosts=False, good_depth=True)
    }

    # Plot the scenes
    for idx, (name, scene) in enumerate(scenes.items()):
        ax = plt.subplot(2, 4, idx + 1)
        ax.imshow(scene)
        ax.set_title(f'{name}\n(RGB Render)', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add annotations
        if 'Baseline' in name:
            ax.text(200, 30, '❌ Floating artifacts', ha='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                    fontsize=9, color='white')
            ax.text(200, 370, '❌ Ghost people', ha='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                    fontsize=9, color='white')
        elif 'DN-Splatter' in name:
            ax.text(200, 30, '✓ Clean geometry', ha='center',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                    fontsize=9, color='white')
            ax.text(200, 370, '❌ Still has ghosts', ha='center',
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                    fontsize=9, color='white')
        elif 'SpotLessSplats' in name:
            ax.text(200, 30, '❌ Floating artifacts', ha='center',
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                    fontsize=9, color='white')
            ax.text(200, 370, '✓ No ghosts', ha='center',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                    fontsize=9, color='white')
        else:  # Combined
            ax.text(200, 30, '✓ Clean geometry', ha='center',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                    fontsize=9, color='white')
            ax.text(200, 370, '✓ No ghosts', ha='center',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                    fontsize=9, color='white')

    # Create depth maps
    for idx, (name, _) in enumerate(scenes.items()):
        ax = plt.subplot(2, 4, idx + 5)

        # Generate depth map
        if 'DN-Splatter' in name or 'Combined' in name:
            # Good depth
            depth = np.linspace(0, 1, 400).reshape(400, 1) * np.ones((1, 400))
            depth[100:300, 150:250] = 0.3  # Building is closer
        else:
            # Poor depth with noise
            depth = np.random.rand(400, 400) * 0.3 + 0.5
            depth[100:300, 150:250] = 0.3 + np.random.rand(200, 100) * 0.2

        ax.imshow(depth, cmap='plasma')
        ax.set_title(f'{name}\n(Depth Map)', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add quality indicator
        if 'DN-Splatter' in name or 'Combined' in name:
            ax.text(200, 370, '✓ Accurate depth', ha='center',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                    fontsize=9, color='white')
        else:
            ax.text(200, 370, '❌ Noisy depth', ha='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                    fontsize=9, color='white')

    plt.suptitle('Visual Quality Comparison: 3D Gaussian Splatting Methods',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    output_file = 'output/visual_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visual comparison saved to: {output_file}")
    plt.show()


def create_concept_diagram():
    """Create a conceptual diagram showing what each method does"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    methods = [
        ('Baseline 3DGS', 'Appearance-only optimization',
         ['❌ Floating artifacts', '❌ Poor depth', '❌ Ghost objects']),
        ('DN-Splatter', 'Adds geometric supervision',
         ['✓ Depth priors', '✓ Normal consistency', '❌ Still has ghosts']),
        ('SpotLessSplats', 'Adds outlier detection',
         ['✓ Detects distractors', '✓ Robust loss', '❌ Poor geometry']),
        ('Combined', 'Both geometric + robust',
         ['✓ Depth & normals', '✓ Outlier handling', '✓ Best quality'])
    ]

    colors = ['#95a5a6', '#3498db', '#e67e22', '#27ae60']

    for idx, (ax, (name, desc, features), color) in enumerate(zip(axes.flat, methods, colors)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Title box
        title_box = Rectangle((0.5, 7.5), 9, 2, facecolor=color, alpha=0.8)
        ax.add_patch(title_box)
        ax.text(5, 8.5, name, ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')

        # Description
        ax.text(5, 6.5, desc, ha='center', va='center',
                fontsize=12, style='italic')

        # Features
        for i, feature in enumerate(features):
            y_pos = 5 - i * 1.2
            marker = '✓' if '✓' in feature else '❌'
            marker_color = 'green' if '✓' in feature else 'red'

            ax.text(1.5, y_pos, marker, ha='center', va='center',
                    fontsize=20, color=marker_color, fontweight='bold')
            ax.text(2.5, y_pos, feature.replace('✓', '').replace('❌', ''),
                    ha='left', va='center', fontsize=11)

        # Add border
        border = Rectangle((0.2, 0.2), 9.6, 9.6, fill=False,
                           edgecolor=color, linewidth=3)
        ax.add_patch(border)

    plt.suptitle('Method Comparison: What Each Approach Addresses',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    output_file = 'output/concept_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Concept diagram saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    print("Generating visual comparisons...")
    print("=" * 60)

    # Create both visualizations
    create_visual_comparison()
    print()
    create_concept_diagram()

    print("\n" + "=" * 60)
    print("Visual comparisons complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • output/visual_comparison.png - Side-by-side results")
    print("  • output/concept_comparison.png - Conceptual diagram")
    print("\nUse these in your video/presentation to show:")
    print("  1. Visual artifacts each method addresses")
    print("  2. Depth quality improvements")
    print("  3. Ghost artifact reduction")