import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# ==========================================
# 1. LOAD & CLEAN REAL DATA (FINAL FIX)
# ==========================================
try:
    # Load the Excel file
    df = pd.read_excel('real_carbonation_data.xls')
    print(f"✅ Success! Loaded {len(df)} rows.")

    # ---------------------------------------------------------
    # STEP A: RENAME COLUMNS (Matched to your specific file)
    # ---------------------------------------------------------
    df = df.rename(columns={
        'w/c (%)': 'wc_ratio', 
        'CO₂ (%)': 'co2_conc',
        'RH (%)': 'rh', 
        'Carbonation depth\nX (mm)': 'depth_mm',
        'Carbonation time (√d)': 'sqrt_days_input' # We need to convert this!
    })

    # ---------------------------------------------------------
    # STEP B: DATA CLEANING & CONVERSION
    # ---------------------------------------------------------
    # 1. Create 'time_days' from the square-root column
    # Since input is sqrt(days), we square it to get days: (√d)^2 = d
    df['time_days'] = df['sqrt_days_input'] ** 2

    # 2. Fix w/c ratio if it's in percentage (e.g., 50 instead of 0.5)
    # If the average is > 1, it's definitely a percentage.
    if df['wc_ratio'].mean() > 1.0:
        print("ℹ️  Converting w/c from % to ratio (dividing by 100)...")
        df['wc_ratio'] = df['wc_ratio'] / 100.0

    # 3. Drop rows with missing values (NaN) to prevent errors
    df = df.dropna(subset=['wc_ratio', 'time_days', 'depth_mm'])
    
    print(f"✅ Data Cleaned. Final columns: {list(df.columns)}")

except Exception as e:
    print(f"⚠️ Error: {e}")
    raise ValueError("Something went wrong loading the data.")
# ==========================================
# 2. FEATURE ENGINEERING (The "Physics-Informed" Trick)
# ==========================================
# Tuutti's paper proves x ~ sqrt(t). 
# We explicitly give this feature to the ML model to help it learn.
df['sqrt_time_yrs'] = np.sqrt(df['time_days'] / 365.0)

features = ['wc_ratio', 'co2_conc', 'rh', 'sqrt_time_yrs']
target = 'depth_mm'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ==========================================
# 3. TRAIN GRADIENT BOOSTING (Better for noisy data)
# ==========================================
# We switch from Random Forest to Gradient Boosting (XGBoost style)
# because it handles outliers in real data better.
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

print(f"✅ Model Trained on 'Real' Data. R² Score: {model.score(X_test, y_test):.4f}")

# ==========================================
# 4. PREDICT SERVICE LIFE (Inverse Solving)
# ==========================================
def predict_real_life(wc, cover_mm, env_co2=0.04, env_rh=70):
    # We predict depth for 100 years. Failure = Depth > Cover.
    years = np.linspace(1, 150, 150)
    
    inputs = pd.DataFrame({
        'wc_ratio': [wc]*150,
        'co2_conc': [env_co2]*150,
        'rh': [env_rh]*150,
        'sqrt_time_yrs': np.sqrt(years),
        'time_days': years * 365 # Include for completeness if model uses it
    })
    
    # Only keep columns used in training
    pred_depths = model.predict(inputs[features])
    
    # Find failure year
    fail_idx = np.where(pred_depths >= cover_mm)[0]
    
    if len(fail_idx) > 0:
        return years[fail_idx[0]], pred_depths
    else:
        return 150, pred_depths

# Run Test
my_wc = 0.55
my_cover = 30
years_life, curve = predict_real_life(my_wc, my_cover)

print(f"\n🔮 Prediction for w/c={my_wc}, Cover={my_cover}mm:")
print(f"Estimated Service Life: {years_life:.1f} years")

plt.plot(np.linspace(1, 150, 150), curve, label='Predicted Carbonation')
plt.axhline(my_cover, color='red', linestyle='--', label='Cover Limit')
plt.title(f'Real-Data Model Prediction (w/c={my_wc})')
plt.xlabel('Years')
plt.ylabel('Carbonation Depth (mm)')
plt.legend()
plt.show()