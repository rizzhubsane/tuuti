# Concrete Service Life Prediction Model

## Based on K. Tuutti's Two-Stage Corrosion Model (SP 65-13)

---

## 📋 Project Overview

This project implements a machine learning model to predict the service life of concrete structures with regard to reinforcement corrosion, based on the seminal research by K. Tuutti from the Swedish Cement and Concrete Research Institute.

### Key Innovation: Two-Stage Model

The service life is divided into two distinct phases:

1. **Initiation Stage**: Period before corrosion begins (passivation breakdown)
2. **Propagation Stage**: Period during which active corrosion occurs until unacceptable damage

---

## 🎯 Features Used for Prediction

### Initiation Stage Features (5 features)
1. **Concrete Cover Depth (mm)** - Depth of concrete covering reinforcement
2. **Water/Cement Ratio** - Quality indicator (lower = better)
3. **CO₂ Concentration (%)** - Atmospheric carbon dioxide levels
4. **Chloride Concentration (% by weight)** - Critical for coastal/de-icing salt exposure
5. **Diffusion Coefficient (m²/s)** - Rate of substance penetration through concrete

### Propagation Stage Features (4 features)
6. **Temperature (°C)** - Environmental temperature
7. **Relative Humidity (%)** - Moisture content in air
8. **Oxygen Supply (relative scale)** - Availability for cathode reactions
9. **Electrical Resistance (Ω·m)** - Concrete's resistance to ion flow

---

## 📊 Model Performance

### Best Performing Model: MLP Neural Network
- **Test R² Score**: 0.9523
- **RMSE**: 6.28 years
- **MAE**: 3.49 years
- **Cross-Validation R²**: 0.918 ± 0.032

### Random Forest (Primary Model)
- **Test R² Score**: 0.9149
- **RMSE**: 8.39 years
- **MAE**: 3.99 years

---

## 🔍 Key Findings

### Feature Importance Ranking

1. **Relative Humidity** (29.97%) - Propagation Stage ⚠️ MOST CRITICAL
2. **Oxygen Supply** (28.74%) - Propagation Stage
3. **Temperature** (22.40%) - Propagation Stage
4. **Electrical Resistance** (15.21%) - Propagation Stage
5. **Diffusion Coefficient** (1.63%) - Initiation Stage
6. **Chloride Concentration** (0.59%) - Initiation Stage
7. **W/C Ratio** (0.53%) - Initiation Stage
8. **Cover Depth** (0.48%) - Initiation Stage
9. **CO₂ Concentration** (0.47%) - Initiation Stage

### Critical Insight
**Propagation stage features dominate** the prediction (88.32% combined importance), confirming that environmental conditions during active corrosion are more critical than initial material properties for overall service life.

---

## 📈 Sensitivity Analysis Results

Impact of ±20% variation on service life:

| Feature | Impact on Service Life |
|---------|----------------------|
| Relative Humidity | -214.0% (EXTREME) |
| Temperature | -38.7% (HIGH) |
| Electrical Resistance | +37.5% (HIGH) |
| Oxygen Supply | -24.5% (MODERATE) |
| W/C Ratio | -10.7% (LOW) |
| Chloride Concentration | -8.6% (LOW) |
| Diffusion Coefficient | -1.8% (MINIMAL) |
| CO₂ Concentration | -1.3% (MINIMAL) |
| Cover Depth | +0.7% (MINIMAL) |

---

## 🎯 Example Predictions

### Scenario 1: High Quality - Low Exposure
- Cover Depth: 50 mm
- W/C Ratio: 0.40
- Chloride: 0.10%
- Temperature: 15°C
- RH: 65%
- **Predicted Service Life: 28.52 years**

### Scenario 2: Medium Quality - Moderate Exposure
- Cover Depth: 35 mm
- W/C Ratio: 0.50
- Chloride: 0.30%
- Temperature: 20°C
- RH: 75%
- **Predicted Service Life: 5.72 years**

### Scenario 3: Low Quality - High Exposure (Coastal)
- Cover Depth: 25 mm
- W/C Ratio: 0.65
- Chloride: 0.80%
- Temperature: 25°C
- RH: 85%
- **Predicted Service Life: 1.78 years**

### Scenario 4: Excellent Quality - Urban Environment
- Cover Depth: 65 mm
- W/C Ratio: 0.35
- Chloride: 0.05%
- Temperature: 18°C
- RH: 70%
- **Predicted Service Life: 16.23 years**

---

## 📚 Dataset Statistics

- **Total Samples**: 2,000
- **Service Life Range**: 0.59 - 209.11 years
- **Mean Service Life**: 25.21 years
- **Median Service Life**: 13.98 years
- **Standard Deviation**: 30.16 years

---

## 🔬 Model Comparison

All models tested (ranked by Test R²):

1. **MLP Neural Network**: 0.9523
2. **Gradient Boosting**: 0.9439
3. **Random Forest**: 0.9149
4. **Lasso Regression**: 0.6594
5. **Ridge Regression**: 0.6572
6. **ElasticNet**: 0.5855
7. **AdaBoost**: 0.4921

---

## 🌡️ Critical Environmental Thresholds (from Tuutti's Research)

### Carbonation
- Atmospheric CO₂: 0.032% (higher in urban areas)
- Critical pH reduction: Concrete becomes unable to maintain passive steel state

### Chloride Attack
- Critical Cl⁻/OH⁻ ratio: > 0.61 triggers local corrosion
- Free chloride threshold: ~0.4% by weight

### Humidity Effects
- Maximum corrosion rate: ~80% RH
- Extremely low rates: Fully water-saturated conditions
- Gas diffusion: 36,000× faster in air than water-saturated concrete

### Temperature
- Accelerates all chemical processes
- Increases molecular mobility
- Exponential effect on corrosion rates

---

## 💡 Practical Recommendations

### To Maximize Service Life:

1. **Control Humidity Exposure** (CRITICAL)
   - Design for drainage
   - Avoid environments with 70-85% RH
   - Consider waterproofing in high-humidity zones

2. **Minimize Temperature Fluctuations**
   - Insulation where appropriate
   - Avoid direct solar exposure

3. **Increase Electrical Resistance**
   - Use low w/c ratio (<0.45)
   - Adequate curing
   - Quality concrete mix

4. **Reduce Oxygen Availability**
   - Waterproofing
   - Protective coatings
   - Limited to practical applications

5. **Material Quality (Initiation Stage)**
   - Adequate cover depth (≥40mm for normal exposure)
   - Low w/c ratio
   - Quality cement with chloride binding capacity

---

## 📁 Project Files

### Code Files
- `concrete_service_life_model.py` - Main ML model implementation
- `advanced_analysis.py` - Sensitivity and comparative analysis

### Data Files
- `synthetic_concrete_data.csv` - Generated dataset (2,000 samples)
- `model_comparison_results.csv` - Algorithm comparison metrics
- `sensitivity_analysis_results.csv` - Feature sensitivity data

### Visualizations
- `service_life_predictions.png` - Model performance and predictions
- `sensitivity_analysis.png` - Feature impact analysis
- `model_comparison.png` - Algorithm comparison
- `feature_interactions.png` - Multi-feature analysis
- `correlation_heatmap.png` - Feature correlation matrix

---

## 🔧 Usage

### Training a Model
```python
from concrete_service_life_model import ConcreteServiceLifeModel

# Initialize model
model = ConcreteServiceLifeModel()

# Generate synthetic data
df = model.generate_synthetic_data(n_samples=2000)

# Train
X = df[model.feature_names]
y = df['service_life_years']
results = model.train(X, y, model_type='random_forest')
```

### Making Predictions
```python
import pandas as pd

# Define scenario
scenario = {
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

# Predict
X_new = pd.DataFrame([scenario])
predicted_life = model.predict(X_new)
print(f"Predicted Service Life: {predicted_life[0]:.2f} years")
```

---

## 📖 References

**Primary Source:**
Tuutti, K. (1982). "Service life of structures with regard to corrosion of embedded steel." 
*Swedish Cement and Concrete Research Institute*, SP 65-13, Stockholm.

**Key Concepts:**
- Two-stage corrosion model (Initiation + Propagation)
- Carbonation and chloride-induced corrosion
- Environmental factors affecting corrosion rates
- Acceptable degree of corrosion for structural integrity

---

## 🎓 Author Notes

This implementation demonstrates how classical durability research can be enhanced with modern machine learning techniques. The model successfully captures the complex interactions between material properties and environmental conditions as described in Tuutti's foundational work.

The high importance of propagation stage features (particularly relative humidity) aligns with Tuutti's findings that environmental conditions during active corrosion are critical for service life prediction.

---

## ⚠️ Limitations & Disclaimers

1. **Synthetic Data**: Model trained on physics-based synthetic data, not real field measurements
2. **Simplified Model**: Real corrosion involves additional complexities not captured here
3. **Regional Variations**: Different climates and exposure classes may require model recalibration
4. **Safety Margin**: Always apply appropriate safety factors for structural design
5. **Professional Judgment**: This tool supports but does not replace engineering expertise

---

## 📊 Statistical Summary

**Model Reliability:**
- Training accuracy demonstrates good learning (R² = 0.979)
- Test accuracy shows excellent generalization (R² = 0.915)
- Low overfitting (difference < 7%)
- Cross-validation confirms robustness (R² = 0.918 ± 0.032)

**Prediction Accuracy:**
- 68% of predictions within ±8.4 years (1 RMSE)
- 95% of predictions within ±16.8 years (2 RMSE)
- Mean Absolute Error: 4.0 years

---

*Generated on February 6, 2026*
*Based on K. Tuutti's Research (SP 65-13)*
