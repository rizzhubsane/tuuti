"""
Concrete Service Life Prediction Model
Based on K. Tuutti's Two-Stage Model (Initiation + Propagation)

Features:
Initiation Stage:
1. Concrete cover depth (x) - mm
2. Water/cement ratio (w/c) - dimensionless
3. CO₂ concentration - %
4. Chloride concentration - % by weight
5. Diffusion coefficient (D) - m²/s

Propagation Stage:
6. Temperature (T) - °C
7. Relative humidity (RH) - %
8. Oxygen supply (O₂) - relative scale
9. Electrical resistance - Ω·m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ConcreteServiceLifeModel:
    """
    ML Model to predict concrete service life based on Tuutti's two-stage model
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'cover_depth_mm',
            'wc_ratio',
            'co2_concentration_percent',
            'chloride_concentration_percent',
            'diffusion_coefficient',
            'temperature_celsius',
            'relative_humidity_percent',
            'oxygen_supply',
            'electrical_resistance'
        ]
        
    def calculate_initiation_time(self, cover_depth, wc_ratio, co2_conc, 
                                  chloride_conc, diffusion_coeff):
        """
        Calculate initiation time based on carbonation and chloride penetration
        Based on equations from the paper
        """
        # Carbonation initiation (years)
        # From Figure 3: x = sqrt(D * t) relationship
        # Adjusted diffusion based on w/c ratio
        D_eff = diffusion_coeff * (1 + 2 * (wc_ratio - 0.4))  # Higher w/c = higher diffusion
        
        # Carbonation time: t = x² / (D_eff * CO2_factor)
        co2_factor = co2_conc / 0.032  # Normalized to typical atmospheric CO2
        t_carbonation = (cover_depth ** 2) / (D_eff * 1e9 * co2_factor + 1e-10)
        
        # Chloride initiation (years)
        # Critical chloride threshold ~0.4% by weight
        # Time increases with cover depth, decreases with chloride concentration
        chloride_factor = max(0.4 - chloride_conc, 0.01)
        t_chloride = (cover_depth ** 2) / (D_eff * 1e9 * (1/chloride_factor) + 1e-10)
        
        # Take minimum (whichever happens first)
        t_init = min(t_carbonation, t_chloride)
        
        return max(t_init, 0.1)  # Minimum 0.1 year
    
    def calculate_propagation_rate(self, temperature, rh, oxygen, resistance):
        """
        Calculate propagation rate and time to failure
        Based on environmental factors from the paper
        """
        # Temperature effect (Arrhenius-like relationship)
        temp_factor = np.exp(0.05 * (temperature - 20))  # Reference at 20°C
        
        # Humidity effect (peak corrosion at 80% RH)
        # From paper: extremely low under water, maximum at 80% RH
        if rh < 60:
            rh_factor = 0.2 * (rh / 60)
        elif 60 <= rh <= 90:
            rh_factor = 0.2 + 0.8 * ((rh - 60) / 20)  # Peak at 80%
        else:
            rh_factor = 1.0 - 0.3 * ((rh - 90) / 10)  # Decreases above 90%
        
        # Oxygen availability (relative scale)
        o2_factor = oxygen / 10.0  # Normalized
        
        # Electrical resistance (higher resistance = slower corrosion)
        resist_factor = 100.0 / (resistance + 10)
        
        # Combined propagation rate (mm/year)
        propagation_rate = temp_factor * rh_factor * o2_factor * resist_factor
        
        # Time for propagation to cause failure (years)
        # Assuming acceptable corrosion depth is a few mm (from paper)
        acceptable_loss = 3.0  # mm (from paper: "few millimeters")
        t_propagation = acceptable_loss / (propagation_rate + 0.01)
        
        return max(t_propagation, 0.5)  # Minimum 0.5 years
    
    def generate_synthetic_data(self, n_samples=2000):
        """
        Generate synthetic dataset based on physical principles from the paper
        """
        data = []
        
        for _ in range(n_samples):
            # Initiation stage features
            cover_depth = np.random.uniform(10, 100)  # mm
            wc_ratio = np.random.uniform(0.35, 0.7)  # typical range
            co2_conc = np.random.uniform(0.03, 0.08)  # % (0.032% typical, higher in urban)
            chloride_conc = np.random.uniform(0.0, 1.2)  # % by weight
            diffusion_coeff = np.random.uniform(1e-8, 1e-6)  # m²/s
            
            # Propagation stage features
            temperature = np.random.uniform(0, 40)  # °C
            rh = np.random.uniform(40, 100)  # %
            oxygen = np.random.uniform(1, 10)  # relative scale
            resistance = np.random.uniform(10, 200)  # Ω·m
            
            # Calculate service life components
            t_init = self.calculate_initiation_time(
                cover_depth, wc_ratio, co2_conc, chloride_conc, diffusion_coeff
            )
            
            t_prop = self.calculate_propagation_rate(
                temperature, rh, oxygen, resistance
            )
            
            # Total service life (years)
            service_life = t_init + t_prop
            
            # Add some realistic noise
            service_life *= np.random.uniform(0.9, 1.1)
            
            data.append([
                cover_depth, wc_ratio, co2_conc, chloride_conc, diffusion_coeff,
                temperature, rh, oxygen, resistance, service_life
            ])
        
        df = pd.DataFrame(data, columns=self.feature_names + ['service_life_years'])
        return df
    
    def train(self, X, y, model_type='random_forest'):
        """
        Train the ML model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Select and train model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            self.model = LinearRegression()
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        results = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, 
            scoring='r2', n_jobs=-1
        )
        results['cv_r2_mean'] = cv_scores.mean()
        results['cv_r2_std'] = cv_scores.std()
        
        return results
    
    def predict(self, X):
        """
        Predict service life for new data
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """
        Get feature importance (for tree-based models)
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None


def visualize_results(model, results, df):
    """
    Create comprehensive visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Concrete Service Life Prediction Model - Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(results['y_test'], results['y_pred_test'], alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(results['y_test'].min(), results['y_pred_test'].min())
    max_val = max(results['y_test'].max(), results['y_pred_test'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Service Life (years)', fontsize=11)
    ax1.set_ylabel('Predicted Service Life (years)', fontsize=11)
    ax1.set_title(f'Actual vs Predicted\nR² = {results["test_r2"]:.4f}', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax2 = axes[0, 1]
    residuals = results['y_test'] - results['y_pred_test']
    ax2.scatter(results['y_pred_test'], residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Service Life (years)', fontsize=11)
    ax2.set_ylabel('Residuals (years)', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance
    ax3 = axes[0, 2]
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        importance_df = importance_df.head(9)
        bars = ax3.barh(range(len(importance_df)), importance_df['importance'])
        ax3.set_yticks(range(len(importance_df)))
        ax3.set_yticklabels(importance_df['feature'], fontsize=9)
        ax3.set_xlabel('Importance', fontsize=11)
        ax3.set_title('Feature Importance', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Color code by stage
        colors = ['#1f77b4' if i < 5 else '#ff7f0e' for i in range(len(importance_df))]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Initiation Stage'),
            Patch(facecolor='#ff7f0e', label='Propagation Stage')
        ]
        ax3.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 4. Distribution of Service Life
    ax4 = axes[1, 0]
    ax4.hist(df['service_life_years'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax4.axvline(df['service_life_years'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean = {df["service_life_years"].mean():.1f} years')
    ax4.axvline(df['service_life_years'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median = {df["service_life_years"].median():.1f} years')
    ax4.set_xlabel('Service Life (years)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribution of Service Life', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cover Depth vs Service Life
    ax5 = axes[1, 1]
    scatter = ax5.scatter(df['cover_depth_mm'], df['service_life_years'], 
                         c=df['wc_ratio'], cmap='viridis', alpha=0.6, 
                         edgecolors='k', linewidth=0.5)
    ax5.set_xlabel('Concrete Cover Depth (mm)', fontsize=11)
    ax5.set_ylabel('Service Life (years)', fontsize=11)
    ax5.set_title('Cover Depth vs Service Life\n(colored by W/C ratio)', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('W/C Ratio', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Chloride Concentration vs Service Life
    ax6 = axes[1, 2]
    scatter2 = ax6.scatter(df['chloride_concentration_percent'], df['service_life_years'],
                          c=df['relative_humidity_percent'], cmap='RdYlBu_r', alpha=0.6,
                          edgecolors='k', linewidth=0.5)
    ax6.set_xlabel('Chloride Concentration (% by weight)', fontsize=11)
    ax6.set_ylabel('Service Life (years)', fontsize=11)
    ax6.set_title('Chloride Concentration vs Service Life\n(colored by RH%)', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=ax6)
    cbar2.set_label('Relative Humidity (%)', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_model_summary(results):
    """
    Print comprehensive model performance summary
    """
    print("=" * 80)
    print("CONCRETE SERVICE LIFE PREDICTION MODEL - PERFORMANCE SUMMARY")
    print("=" * 80)
    print("\nBased on K. Tuutti's Two-Stage Model (Initiation + Propagation)")
    print("\n" + "-" * 80)
    print("MODEL PERFORMANCE METRICS")
    print("-" * 80)
    print(f"\nTraining Set:")
    print(f"  R² Score:     {results['train_r2']:.4f}")
    print(f"  RMSE:         {results['train_rmse']:.4f} years")
    print(f"  MAE:          {results['train_mae']:.4f} years")
    
    print(f"\nTest Set:")
    print(f"  R² Score:     {results['test_r2']:.4f}")
    print(f"  RMSE:         {results['test_rmse']:.4f} years")
    print(f"  MAE:          {results['test_mae']:.4f} years")
    
    print(f"\nCross-Validation (5-fold):")
    print(f"  R² Score:     {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
    
    print("\n" + "=" * 80)


def demonstrate_predictions(model, df):
    """
    Demonstrate predictions on specific scenarios
    """
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS FOR DIFFERENT SCENARIOS")
    print("=" * 80)
    
    scenarios = [
        {
            'name': 'High Quality - Low Exposure',
            'cover_depth_mm': 50,
            'wc_ratio': 0.40,
            'co2_concentration_percent': 0.032,
            'chloride_concentration_percent': 0.1,
            'diffusion_coefficient': 1e-7,
            'temperature_celsius': 15,
            'relative_humidity_percent': 65,
            'oxygen_supply': 5,
            'electrical_resistance': 150
        },
        {
            'name': 'Medium Quality - Moderate Exposure',
            'cover_depth_mm': 35,
            'wc_ratio': 0.50,
            'co2_concentration_percent': 0.05,
            'chloride_concentration_percent': 0.3,
            'diffusion_coefficient': 5e-7,
            'temperature_celsius': 20,
            'relative_humidity_percent': 75,
            'oxygen_supply': 7,
            'electrical_resistance': 80
        },
        {
            'name': 'Low Quality - High Exposure (Coastal)',
            'cover_depth_mm': 25,
            'wc_ratio': 0.65,
            'co2_concentration_percent': 0.04,
            'chloride_concentration_percent': 0.8,
            'diffusion_coefficient': 8e-7,
            'temperature_celsius': 25,
            'relative_humidity_percent': 85,
            'oxygen_supply': 8,
            'electrical_resistance': 40
        },
        {
            'name': 'Excellent Quality - Urban Environment',
            'cover_depth_mm': 65,
            'wc_ratio': 0.35,
            'co2_concentration_percent': 0.06,
            'chloride_concentration_percent': 0.05,
            'diffusion_coefficient': 2e-7,
            'temperature_celsius': 18,
            'relative_humidity_percent': 70,
            'oxygen_supply': 6,
            'electrical_resistance': 180
        }
    ]
    
    for scenario in scenarios:
        name = scenario.pop('name')
        X_scenario = pd.DataFrame([scenario])
        predicted_life = model.predict(X_scenario)[0]
        
        print(f"\nScenario: {name}")
        print("-" * 80)
        print(f"  Cover Depth:              {scenario['cover_depth_mm']:.1f} mm")
        print(f"  W/C Ratio:                {scenario['wc_ratio']:.2f}")
        print(f"  CO₂ Concentration:        {scenario['co2_concentration_percent']:.3f}%")
        print(f"  Chloride Concentration:   {scenario['chloride_concentration_percent']:.2f}% by weight")
        print(f"  Diffusion Coefficient:    {scenario['diffusion_coefficient']:.2e} m²/s")
        print(f"  Temperature:              {scenario['temperature_celsius']:.1f}°C")
        print(f"  Relative Humidity:        {scenario['relative_humidity_percent']:.1f}%")
        print(f"  Oxygen Supply:            {scenario['oxygen_supply']:.1f}")
        print(f"  Electrical Resistance:    {scenario['electrical_resistance']:.1f} Ω·m")
        print(f"\n  → PREDICTED SERVICE LIFE: {predicted_life:.2f} years")
    
    print("\n" + "=" * 80)


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CONCRETE SERVICE LIFE PREDICTION MODEL")
    print("Based on K. Tuutti's Research (SP 65-13)")
    print("=" * 80)
    
    # Initialize model
    model = ConcreteServiceLifeModel()
    
    # Generate synthetic data
    print("\nGenerating synthetic dataset based on physical principles...")
    df = model.generate_synthetic_data(n_samples=2000)
    
    print(f"Dataset created: {len(df)} samples")
    print(f"\nService Life Statistics:")
    print(f"  Mean:     {df['service_life_years'].mean():.2f} years")
    print(f"  Median:   {df['service_life_years'].median():.2f} years")
    print(f"  Std Dev:  {df['service_life_years'].std():.2f} years")
    print(f"  Min:      {df['service_life_years'].min():.2f} years")
    print(f"  Max:      {df['service_life_years'].max():.2f} years")
    
    # Prepare features and target
    X = df[model.feature_names]
    y = df['service_life_years']
    
    # Train model
    print("\nTraining Random Forest model...")
    results = model.train(X, y, model_type='random_forest')
    
    # Print summary
    print_model_summary(results)
    
    # Show feature importance
    print("\nFEATURE IMPORTANCE RANKING:")
    print("-" * 80)
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        for idx, row in importance_df.iterrows():
            stage = "Initiation" if idx < 5 else "Propagation"
            print(f"{idx+1}. {row['feature']:35s} {row['importance']:.4f}  [{stage}]")
    
    # Demonstrate predictions
    demonstrate_predictions(model, df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig = visualize_results(model, results, df)
    plt.savefig('/home/claude/service_life_predictions.png', dpi=300, bbox_inches='tight')
    print("Visualization saved: service_life_predictions.png")
    
    # Save the dataset
    df.to_csv('/home/claude/synthetic_concrete_data.csv', index=False)
    print("Dataset saved: synthetic_concrete_data.csv")
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 80)
