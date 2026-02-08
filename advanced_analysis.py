"""
Interactive Concrete Service Life Prediction Tool
Includes sensitivity analysis and comparative model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concrete_service_life_model import ConcreteServiceLifeModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def sensitivity_analysis(model, base_scenario, feature_names):
    """
    Perform sensitivity analysis - vary each feature individually
    """
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing how each feature affects service life predictions...")
    
    # Base scenario
    base_df = pd.DataFrame([base_scenario])
    base_prediction = model.predict(base_df)[0]
    
    print(f"\nBase Scenario Prediction: {base_prediction:.2f} years")
    print("\nFeature Variations (±20% from base):")
    print("-" * 80)
    
    sensitivity_results = []
    
    for i, feature in enumerate(feature_names):
        scenario_low = base_scenario.copy()
        scenario_high = base_scenario.copy()
        
        # Vary feature by ±20%
        scenario_low[feature] = base_scenario[feature] * 0.8
        scenario_high[feature] = base_scenario[feature] * 1.2
        
        pred_low = model.predict(pd.DataFrame([scenario_low]))[0]
        pred_high = model.predict(pd.DataFrame([scenario_high]))[0]
        
        impact = pred_high - pred_low
        pct_change = (impact / base_prediction) * 100 if base_prediction != 0 else 0
        
        sensitivity_results.append({
            'feature': feature,
            'base_value': base_scenario[feature],
            'low_value': scenario_low[feature],
            'high_value': scenario_high[feature],
            'pred_low': pred_low,
            'pred_high': pred_high,
            'impact': impact,
            'pct_change': abs(pct_change)
        })
        
        print(f"{feature:35s}")
        print(f"  Base value: {base_scenario[feature]:.4e}")
        print(f"  -20% → {pred_low:6.2f} years | +20% → {pred_high:6.2f} years")
        print(f"  Impact: {impact:+.2f} years ({pct_change:+.1f}%)")
        print()
    
    # Sort by absolute impact
    sensitivity_df = pd.DataFrame(sensitivity_results).sort_values('pct_change', ascending=False)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#ff7f0e' if 'temperature' in f or 'humidity' in f or 'oxygen' in f or 'resistance' in f 
              else '#1f77b4' for f in sensitivity_df['feature']]
    
    bars = ax.barh(range(len(sensitivity_df)), sensitivity_df['pct_change'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(sensitivity_df)))
    ax.set_yticklabels(sensitivity_df['feature'], fontsize=10)
    ax.set_xlabel('Impact on Service Life (% change)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity Analysis: Feature Impact on Service Life\n(±20% variation from base scenario)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='Initiation Stage Features'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='Propagation Stage Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/claude/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("Sensitivity analysis plot saved!")
    
    return sensitivity_df


def compare_models(X, y):
    """
    Compare multiple ML algorithms
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print("\nComparing different machine learning algorithms...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, random_state=42),
        'MLP Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = []
    
    print("\n" + "-" * 80)
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results.append({
            'Model': name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'RMSE': test_rmse,
            'MAE': test_mae
        })
        
        print(f"{name:20s} | R²: {test_r2:.4f} | RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}")
    
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # R² scores
    ax1 = axes[0]
    x_pos = np.arange(len(results_df))
    ax1.bar(x_pos, results_df['Test R²'], alpha=0.7, edgecolor='black', color='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('R² Score Comparison', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # RMSE
    ax2 = axes[1]
    ax2.bar(x_pos, results_df['RMSE'], alpha=0.7, edgecolor='black', color='coral')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('RMSE (years)', fontsize=11, fontweight='bold')
    ax2.set_title('RMSE Comparison (lower is better)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # MAE
    ax3 = axes[2]
    ax3.bar(x_pos, results_df['MAE'], alpha=0.7, edgecolor='black', color='lightgreen')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('MAE (years)', fontsize=11, fontweight='bold')
    ax3.set_title('MAE Comparison (lower is better)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison plot saved!")
    
    return results_df


def analyze_feature_interactions(df):
    """
    Analyze interactions between key features
    """
    print("\n" + "=" * 80)
    print("FEATURE INTERACTION ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Feature Interactions and Their Effect on Service Life', fontsize=16, fontweight='bold')
    
    # 1. W/C Ratio vs Cover Depth
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(df['wc_ratio'], df['cover_depth_mm'], 
                          c=df['service_life_years'], cmap='RdYlGn', 
                          s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax1.set_xlabel('Water/Cement Ratio', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cover Depth (mm)', fontsize=11, fontweight='bold')
    ax1.set_title('W/C Ratio vs Cover Depth\n(colored by Service Life)', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Service Life (years)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Temperature vs Humidity
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['temperature_celsius'], df['relative_humidity_percent'],
                          c=df['service_life_years'], cmap='RdYlGn',
                          s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relative Humidity (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Temperature vs Humidity\n(colored by Service Life)', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Service Life (years)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Chloride vs Diffusion Coefficient
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(df['chloride_concentration_percent'], df['diffusion_coefficient'],
                          c=df['service_life_years'], cmap='RdYlGn',
                          s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax3.set_xlabel('Chloride Concentration (% by weight)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Diffusion Coefficient (m²/s)', fontsize=11, fontweight='bold')
    ax3.set_title('Chloride vs Diffusion Coefficient\n(colored by Service Life)', fontsize=12)
    ax3.set_yscale('log')
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Service Life (years)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Oxygen vs Electrical Resistance
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(df['oxygen_supply'], df['electrical_resistance'],
                          c=df['service_life_years'], cmap='RdYlGn',
                          s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax4.set_xlabel('Oxygen Supply', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Electrical Resistance (Ω·m)', fontsize=11, fontweight='bold')
    ax4.set_title('Oxygen vs Electrical Resistance\n(colored by Service Life)', fontsize=12)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Service Life (years)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/feature_interactions.png', dpi=300, bbox_inches='tight')
    print("Feature interaction plots saved!")


def create_correlation_heatmap(df):
    """
    Create correlation heatmap
    """
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Calculate correlations
    corr_matrix = df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Matrix\n(including Service Life)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/claude/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Correlation heatmap saved!")
    
    # Print strong correlations with service life
    print("\nStrongest Correlations with Service Life:")
    print("-" * 80)
    service_life_corr = corr_matrix['service_life_years'].drop('service_life_years').sort_values(key=abs, ascending=False)
    for feature, corr in service_life_corr.items():
        print(f"{feature:35s} : {corr:+.4f}")


# Main execution
if __name__ == "__main__":
    # Load the previously generated data
    df = pd.read_csv('/home/claude/synthetic_concrete_data.csv')
    
    # Initialize and load model
    model = ConcreteServiceLifeModel()
    feature_names = model.feature_names
    X = df[feature_names]
    y = df['service_life_years']
    
    # Train the model
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.scaler = scaler
    model.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.model.fit(X_scaled, y)
    
    # Define base scenario for sensitivity analysis
    base_scenario = {
        'cover_depth_mm': 40,
        'wc_ratio': 0.45,
        'co2_concentration_percent': 0.04,
        'chloride_concentration_percent': 0.3,
        'diffusion_coefficient': 5e-7,
        'temperature_celsius': 20,
        'relative_humidity_percent': 75,
        'oxygen_supply': 6,
        'electrical_resistance': 100
    }
    
    # Run analyses
    sensitivity_df = sensitivity_analysis(model, base_scenario, feature_names)
    results_df = compare_models(X_scaled, y)
    analyze_feature_interactions(df)
    create_correlation_heatmap(df)
    
    # Save comparison results
    results_df.to_csv('/home/claude/model_comparison_results.csv', index=False)
    sensitivity_df.to_csv('/home/claude/sensitivity_analysis_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. sensitivity_analysis.png")
    print("  2. model_comparison.png")
    print("  3. feature_interactions.png")
    print("  4. correlation_heatmap.png")
    print("  5. model_comparison_results.csv")
    print("  6. sensitivity_analysis_results.csv")
    print("\n" + "=" * 80)
