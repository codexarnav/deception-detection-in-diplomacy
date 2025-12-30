"""
Script to analyze potential feature leakage in game context features
Run this to check if game_score or game_score_delta correlate with deception_state
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/processed_data.csv')

print("="*70)
print("FEATURE CORRELATION ANALYSIS - Checking for Target Leakage")
print("="*70)

# Filter to only include valid deception states
df_clean = df[df['deception_state'].notna()].copy()

print(f"\nTotal samples: {len(df_clean)}")
print(f"Class distribution:\n{df_clean['deception_state'].value_counts()}\n")

# ============================================================================
# 1. Game Score Analysis
# ============================================================================
print("\n" + "="*70)
print("1. GAME SCORE ANALYSIS")
print("="*70)

# Group by deception state and analyze game_score
score_analysis = df_clean.groupby('deception_state')['game_score'].agg([
    'mean', 'std', 'median', 'min', 'max', 'count'
])
print("\nGame Score by Deception State:")
print(score_analysis)

# Statistical test
no_deception_scores = df_clean[df_clean['deception_state'] == 'no_deception']['game_score'].dropna()
deception_scores = df_clean[df_clean['deception_state'] == 'successful_deception']['game_score'].dropna()

from scipy.stats import mannwhitneyu
stat, p_value = mannwhitneyu(no_deception_scores, deception_scores, alternative='two-sided')
print(f"\nMann-Whitney U Test (game_score):")
print(f"  Statistic: {stat:.2f}")
print(f"  P-value: {p_value:.6f}")
if p_value < 0.05:
    print("  ⚠️ WARNING: Significant correlation detected! Potential target leakage.")
else:
    print("  ✓ No significant correlation. Feature appears safe.")

# ============================================================================
# 2. Game Score Delta Analysis
# ============================================================================
print("\n" + "="*70)
print("2. GAME SCORE DELTA ANALYSIS")
print("="*70)

delta_analysis = df_clean.groupby('deception_state')['game_score_delta'].agg([
    'mean', 'std', 'median', 'min', 'max', 'count'
])
print("\nGame Score Delta by Deception State:")
print(delta_analysis)

# Statistical test
no_deception_delta = df_clean[df_clean['deception_state'] == 'no_deception']['game_score_delta'].dropna()
deception_delta = df_clean[df_clean['deception_state'] == 'successful_deception']['game_score_delta'].dropna()

stat, p_value = mannwhitneyu(no_deception_delta, deception_delta, alternative='two-sided')
print(f"\nMann-Whitney U Test (game_score_delta):")
print(f"  Statistic: {stat:.2f}")
print(f"  P-value: {p_value:.6f}")
if p_value < 0.05:
    print("  ⚠️ WARNING: Significant correlation detected! Potential target leakage.")
else:
    print("  ✓ No significant correlation. Feature appears safe.")

# ============================================================================
# 3. Visualization
# ============================================================================
print("\n" + "="*70)
print("3. GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Game Score Distribution
axes[0, 0].hist([no_deception_scores, deception_scores], 
                bins=30, label=['No Deception', 'Successful Deception'], 
                alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Game Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Game Score Distribution by Deception State')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Game Score Delta Distribution
axes[0, 1].hist([no_deception_delta, deception_delta], 
                bins=30, label=['No Deception', 'Successful Deception'], 
                alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Game Score Delta')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Game Score Delta Distribution by Deception State')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Box plots for Game Score
df_clean.boxplot(column='game_score', by='deception_state', ax=axes[1, 0])
axes[1, 0].set_xlabel('Deception State')
axes[1, 0].set_ylabel('Game Score')
axes[1, 0].set_title('Game Score by Deception State (Boxplot)')
plt.sca(axes[1, 0])
plt.xticks(rotation=15)

# Plot 4: Box plots for Game Score Delta
df_clean.boxplot(column='game_score_delta', by='deception_state', ax=axes[1, 1])
axes[1, 1].set_xlabel('Deception State')
axes[1, 1].set_ylabel('Game Score Delta')
axes[1, 1].set_title('Game Score Delta by Deception State (Boxplot)')
plt.sca(axes[1, 1])
plt.xticks(rotation=15)

plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.savefig('feature_correlation_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: feature_correlation_analysis.png")

# ============================================================================
# 4. Correlation Matrix
# ============================================================================
print("\n" + "="*70)
print("4. CORRELATION MATRIX")
print("="*70)

# Create binary target
df_clean['deception_binary'] = (df_clean['deception_state'] == 'successful_deception').astype(int)

# Calculate correlations
features = ['game_score', 'game_score_delta', 'absolute_message_index', 
            'relative_message_index', 'deception_binary']
corr_matrix = df_clean[features].corr()

print("\nCorrelation with Deception Target:")
print(corr_matrix['deception_binary'].sort_values(ascending=False))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix (including Target)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Correlation heatmap saved to: correlation_heatmap.png")

# ============================================================================
# 5. Recommendations
# ============================================================================
print("\n" + "="*70)
print("5. RECOMMENDATIONS")
print("="*70)

threshold = 0.1  # Correlation threshold for concern
correlations = corr_matrix['deception_binary'].drop('deception_binary').abs()

print(f"\nFeature correlations with target (threshold: {threshold}):")
for feature, corr in correlations.items():
    if corr > threshold:
        print(f"  ⚠️ {feature}: {corr:.4f} - Consider removing or investigating")
    else:
        print(f"  ✓ {feature}: {corr:.4f} - Appears safe")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nReview the generated plots and statistics above.")
print("If any features show strong correlation or significant p-values,")
print("consider removing them from the model to prevent target leakage.")
