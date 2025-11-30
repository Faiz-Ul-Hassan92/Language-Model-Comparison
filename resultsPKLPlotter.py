import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('lstm/baseline/lstm_results.pkl', 'rb') as f:
    results = pickle.load(f)



results_data = []
for model_name, metrics in results.items():
    results_data.append({
        'Model': model_name,
        'Optimizer': metrics['optimizer'].upper(),
        'Test Perplexity': metrics['test_perplexity'],
        'Test Accuracy': metrics['test_accuracy'],
        'Test Loss': metrics['test_loss'],
        'Training Time (s)': metrics['training_time']
    })

df = pd.DataFrame(results_data)
df = df.sort_values('Test Perplexity')  # Sort by perplexity (lower is better)

print("\n" + df.to_string(index=False))
print("\n" + "="*80)

# Summary statistics
print("\nSUMMARY STATISTICS:")
print("-"*80)
best_perplexity = df.loc[df['Test Perplexity'].idxmin()]
best_accuracy = df.loc[df['Test Accuracy'].idxmax()]
fastest = df.loc[df['Training Time (s)'].idxmin()]

print(f"Best Perplexity: {best_perplexity['Optimizer']} ({best_perplexity['Test Perplexity']:.2f})")
print(f"Best Accuracy: {best_accuracy['Optimizer']} ({best_accuracy['Test Accuracy']:.4f})")
print(f"Fastest Training: {fastest['Optimizer']} ({fastest['Training Time (s)']:.2f}s)")
print(f"\nPerplexity Range: {df['Test Perplexity'].min():.2f} - {df['Test Perplexity'].max():.2f}")
print(f"Accuracy Range: {df['Test Accuracy'].min():.4f} - {df['Test Accuracy'].max():.4f}")
print("="*80)


# Create visualizations
fig = plt.figure(figsize=(16, 10))

# Use consistent colors for optimizers
colors = {'ADAM': '#1f77b4', 'RMSPROP': '#ff7f0e', 'SGD': '#2ca02c'}
optimizer_colors = [colors[opt] for opt in df['Optimizer']]

# 1. Perplexity Comparison (Lower is better)
ax1 = plt.subplot(2, 3, 1)
bars1 = ax1.bar(df['Optimizer'], df['Test Perplexity'], color=optimizer_colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
ax1.set_title('Test Perplexity by Optimizer\n(Lower is Better)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, df['Test Perplexity'].max() * 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Accuracy Comparison (Higher is better)
ax2 = plt.subplot(2, 3, 2)
bars2 = ax2.bar(df['Optimizer'], df['Test Accuracy'], color=optimizer_colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Test Accuracy by Optimizer\n(Higher is Better)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Training Time Comparison
ax3 = plt.subplot(2, 3, 3)
bars3 = ax3.bar(df['Optimizer'], df['Training Time (s)'], color=optimizer_colors, alpha=0.8, edgecolor='black')
ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('Training Time by Optimizer', fontsize=13, fontweight='bold')
ax3.set_ylim(0, df['Training Time (s)'].max() * 1.15)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Loss Comparison
ax4 = plt.subplot(2, 3, 4)
bars4 = ax4.bar(df['Optimizer'], df['Test Loss'], color=optimizer_colors, alpha=0.8, edgecolor='black')
ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax4.set_title('Test Loss by Optimizer\n(Lower is Better)', fontsize=13, fontweight='bold')
ax4.set_ylim(0, df['Test Loss'].max() * 1.15)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
# Add value labels on bars
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. Performance vs Training Time Scatter
ax5 = plt.subplot(2, 3, 5)
for idx, row in df.iterrows():
    ax5.scatter(row['Training Time (s)'], row['Test Perplexity'], 
               c=colors[row['Optimizer']], s=200, alpha=0.8, edgecolor='black', linewidth=2)
    ax5.annotate(row['Optimizer'], 
                (row['Training Time (s)'], row['Test Perplexity']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold')
ax5.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Test Perplexity', fontsize=12, fontweight='bold')
ax5.set_title('Performance vs Training Time\n(Bottom-left is Best)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, linestyle='--')

# 6. Normalized Comparison (All metrics 0-1 scale)
ax6 = plt.subplot(2, 3, 6)
# Normalize metrics (0=worst, 1=best)
norm_perplexity = 1 - (df['Test Perplexity'] - df['Test Perplexity'].min()) / (df['Test Perplexity'].max() - df['Test Perplexity'].min())
norm_accuracy = df['Test Accuracy']
norm_time = 1 - (df['Training Time (s)'] - df['Training Time (s)'].min()) / (df['Training Time (s)'].max() - df['Training Time (s)'].min())

x = np.arange(len(df))
width = 0.25

bars1 = ax6.bar(x - width, norm_perplexity, width, label='Perplexity (norm)', alpha=0.8, edgecolor='black')
bars2 = ax6.bar(x, norm_accuracy, width, label='Accuracy', alpha=0.8, edgecolor='black')
bars3 = ax6.bar(x + width, norm_time, width, label='Speed (norm)', alpha=0.8, edgecolor='black')

ax6.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
ax6.set_title('Normalized Performance Comparison\n(Higher is Better)', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(df['Optimizer'])
ax6.legend(fontsize=9)
ax6.set_ylim(0, 1.1)
ax6.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('lstm/baseline/lstm_optimizer_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved comparison plots to: transformer/baseline/transformer_optimizer_comparison.png")
plt.show()

# Additional: Ranking table
print("\n" + "="*80)
print("OPTIMIZER RANKING")
print("="*80)
ranking_data = []
for idx, row in df.iterrows():
    perplexity_rank = df['Test Perplexity'].rank().loc[idx]
    accuracy_rank = df['Test Accuracy'].rank(ascending=False).loc[idx]
    time_rank = df['Training Time (s)'].rank().loc[idx]
    avg_rank = (perplexity_rank + accuracy_rank + time_rank) / 3
    
    ranking_data.append({
        'Optimizer': row['Optimizer'],
        'Perplexity Rank': int(perplexity_rank),
        'Accuracy Rank': int(accuracy_rank),
        'Speed Rank': int(time_rank),
        'Average Rank': f"{avg_rank:.2f}"
    })

ranking_df = pd.DataFrame(ranking_data)
ranking_df = ranking_df.sort_values('Average Rank')
print("\n" + ranking_df.to_string(index=False))
print("\n(Rank 1 = Best, 3 = Worst)")
print("="*80)

