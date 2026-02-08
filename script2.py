import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
df = pd.read_excel('real_carbonation_data.xls')

# Rename columns (Using the mapping we verified works)
df = df.rename(columns={
    'w/c (%)': 'wc_ratio', 
    'CO₂ (%)': 'co2_conc',
    'RH (%)': 'rh', 
    'Carbonation depth\nX (mm)': 'depth_mm',
    'Carbonation time (√d)': 'sqrt_days_input'
})

# Clean Data
df['time_days'] = df['sqrt_days_input'] ** 2
if df['wc_ratio'].mean() > 1.0:
    df['wc_ratio'] = df['wc_ratio'] / 100.0

# ---------------------------------------------------------
# THE "HYBRID K" TRICK
# ---------------------------------------------------------
# Instead of predicting Depth directly, we calculate the Rate Coefficient (K).
# K = Depth / sqrt(Time_Years)
# This removes "Time" from the AI training, so it doesn't matter if
# the data is short-term or long-term!

df['time_years'] = df['time_days'] / 365.0
# Avoid division by zero
df = df[df['time_years'] > 0.01] 

# Calculate the actual Rate K observed in the lab
df['K_rate'] = df['depth_mm'] / np.sqrt(df['time_years'])

# Remove outliers (lab errors where K is impossible)
df = df[df['K_rate'] < 50] 

# ==========================================
# 2. TRAIN MODEL TO PREDICT RATE 'K'
# ==========================================
features = ['wc_ratio', 'co2_conc', 'rh']
target = 'K_rate'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=4)
model.fit(X_train, y_train)

print(f"✅ Model Trained to predict Carbonation Rate (K).")
print(f"R² Score: {model.score(X_test, y_test):.4f}")

# ==========================================
# 3. PREDICT SERVICE LIFE
# ==========================================
def predict_service_life(wc, cover_mm, co2=0.04, rh=70):
    # Step 1: Ask AI for the Rate (K) for this specific mix
    # Note: co2 and rh must match the units in the training data (likely %)
    input_data = pd.DataFrame([[wc, co2, rh]], columns=features)
    predicted_K = model.predict(input_data)[0]
    
    print(f"\n--- Prediction Logic ---")
    print(f"Predicted Rate Coefficient (K): {predicted_K:.2f} mm/√year")
    
    # Step 2: Extrapolate using Physics (Depth = K * sqrt(t))
    years = np.linspace(0, 150, 300)
    depth_curve = predicted_K * np.sqrt(years)
    
    # Step 3: Find failure time
    # Check if/when the curve crosses the cover depth
    fail_idx = np.where(depth_curve >= cover_mm)[0]
    
    if len(fail_idx) > 0:
        life_expectancy = years[fail_idx[0]]
    else:
        life_expectancy = 150.0
        
    return life_expectancy, years, depth_curve

# ==========================================
# 4. RUN TEST
# ==========================================
my_wc = 0.40
my_cover = 40 # mm

# IMPORTANT: Check your CO2 units! 
# If dataset has CO2 like "3.0" for 3%, use 0.04. 
# If dataset uses "0.03", use 0.04.
# Based on your columns, it looks like percent.
life, t_axis, depth_axis = predict_service_life(my_wc, my_cover, co2=0.04, rh=70)

print(f"\n🔮 Final Result for w/c={my_wc}:")
print(f"Estimated Service Life: {life:.1f} years")

plt.figure(figsize=(10,6))
plt.plot(t_axis, depth_axis, label=f'Carbonation (w/c={my_wc})', color='blue', linewidth=2)
plt.axhline(my_cover, color='red', linestyle='--', label=f'Cover ({my_cover}mm)')
plt.xlabel('Time (Years)')
plt.ylabel('Carbonation Depth (mm)')
plt.title('Hybrid AI Prediction: Rate K predicted by ML, Time by Physics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 5. FINAL VISUALIZATION: THE COMPARISON
# ==========================================
print("\n📊 Generating Comparison Graph...")

# Scenario A: Poor Quality (Current practice in some places)
life_A, t_A, depth_A = predict_service_life(wc=0.55, cover_mm=30)

# Scenario B: High Performance (Your recommendation)
life_B, t_B, depth_B = predict_service_life(wc=0.40, cover_mm=40)

plt.figure(figsize=(12, 7))

# Plot Scenario A (Poor)
plt.plot(t_A, depth_A, color='red', linewidth=2, linestyle='-', 
         label=f'Standard Mix (w/c=0.55, 30mm)\nLife: {life_A:.1f} years')

# Plot Scenario B (Good)
plt.plot(t_B, depth_B, color='green', linewidth=2, linestyle='-', 
         label=f'High Performance (w/c=0.40, 40mm)\nLife: >{life_B:.0f} years')

# Reference Lines
plt.axhline(30, color='red', linestyle=':', alpha=0.5, label='Cover Limit (30mm)')
plt.axhline(40, color='green', linestyle=':', alpha=0.5, label='Cover Limit (40mm)')

# Styling
plt.fill_between(t_A, depth_A, 0, color='red', alpha=0.05) # Shading
plt.fill_between(t_B, depth_B, 0, color='green', alpha=0.05)

plt.title('AI Prediction: Impact of Concrete Quality on Service Life', fontsize=14)
plt.xlabel('Age of Structure (Years)', fontsize=12)
plt.ylabel('Carbonation Depth (mm)', fontsize=12)
plt.legend(loc='upper left', frameon=True, fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 150)
plt.ylim(0, 50)

plt.show()
print("✅ Comparison Graph Generated!")