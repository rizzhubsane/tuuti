from concrete_service_life_model import ConcreteServiceLifeModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load and prepare model
model = ConcreteServiceLifeModel()
synthetic_data = pd.read_csv('synthetic_concrete_data.csv')

X = synthetic_data[model.feature_names]
y = synthetic_data['service_life_years']

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_scaled, y)

model.model = rf_model
model.scaler = scaler

# Test on your data
results = validator.test_model(my_data, model.model, model.scaler)