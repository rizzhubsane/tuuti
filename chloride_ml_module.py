"""
chloride_ml_module.py
=====================
Chloride-Specific Machine Learning Extension for K. Tuutti's Corrosion Model (SP 65-13)

This module extends the existing ConcreteServiceLifeModel with three dedicated
chloride sub-models:

  Model 1 — ChlorideInitiationModel
      Predicts time-to-corrosion-initiation via chloride penetration (Fick's law based).
      Features: cover depth, diffusion coefficient, surface concentration, threshold concentration.

  Model 2 — CorrosionTypeClassifier
      Binary classifier: predicts local vs. general corrosion based on the
      Cl⁻/OH⁻ > 0.61 threshold (Hausmann, 1967), cement type, and w/c ratio.

  Model 3 — ChlorideProfileModel
      Predicts free chloride concentration vs. depth profile (Figure 6 in paper).
      Features: cement type, w/c ratio, exposure time, depth increment.

Usage:
    from chloride_ml_module import ChlorideCombinedPipeline
    pipeline = ChlorideCombinedPipeline()
    pipeline.run_all()

Author: Extended from rizzhubsane/tuuti
Reference: Tuutti, K. (1982). SP 65-13. Swedish Cement and Concrete Research Institute.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# =============================================================================
# SHARED PHYSICS HELPERS (Tuutti / Fick's Law)
# =============================================================================

def ficks_initiation_time(cover_mm, D_m2s, C_surface_ppm, C_threshold_ppm):
    """
    Estimate initiation time (years) using Fick's 2nd Law solution.

    C(x,t) = C_s * erfc( x / (2 * sqrt(D*t)) )
    Rearranged for t when C(x,t) = C_threshold.

    Returns np.inf if threshold >= surface concentration.
    """
    if C_threshold_ppm >= C_surface_ppm:
        return np.inf
    x = cover_mm / 1000.0  # mm -> m
    ratio = C_threshold_ppm / C_surface_ppm          # erfc(u) = ratio
    # erfc(u) = ratio  =>  u = erfc_inv(ratio)
    # Numerical approach: solve erfc(u) = ratio via bisection
    from scipy.special import erfinv
    u = erfinv(1.0 - ratio)                          # erfc(u) = 1 - erf(u) = ratio
    # u = x / (2*sqrt(D*t))  =>  t = (x/(2u))^2 / D
    t_seconds = (x / (2.0 * u)) ** 2 / D_m2s
    t_years = t_seconds / (365.25 * 24 * 3600)
    return t_years


def cl_oh_ratio(cl_ppm, oh_equiv_per_litre):
    """
    Convert free Cl⁻ (ppm) and OH⁻ (equiv/L) to Cl⁻/OH⁻ ratio (g Cl⁻ / L).
    Threshold >0.61 => local corrosion (Hausmann 1967).
    Cl⁻ in ppm  -> g/L by dividing by 1000.
    """
    cl_g_per_l = cl_ppm / 1000.0
    return cl_g_per_l / (oh_equiv_per_litre * 35.45)   # 35.45 = Cl molar mass


# =============================================================================
# MODEL 1 — CHLORIDE INITIATION PERIOD PREDICTOR
# =============================================================================

class ChlorideInitiationModel:
    """
    Predicts corrosion initiation time (years) due to chloride penetration.

    Features
    --------
    cover_depth_mm        : concrete cover over reinforcement (mm)
    diffusion_coeff       : effective Cl⁻ diffusivity (m²/s)
    surface_cl_ppm        : free chloride concentration at surface (ppm)
    threshold_cl_ppm      : critical free chloride concentration at steel (ppm)
    wc_ratio              : water/cement ratio (affects D, included as auxiliary)
    temperature_celsius   : mean annual temperature (affects D via Arrhenius)
    """

    FEATURE_NAMES = [
        "cover_depth_mm",
        "diffusion_coeff",
        "surface_cl_ppm",
        "threshold_cl_ppm",
        "wc_ratio",
        "temperature_celsius",
    ]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    # ------------------------------------------------------------------
    def generate_data(self, n=2000):
        """
        Synthetic dataset grounded in Tuutti's Figure 8 parameter ranges.
        Physics-based label via Fick's law + temperature correction (Arrhenius).
        """
        cover   = np.random.uniform(10, 120, n)           # mm
        D_base  = np.random.uniform(1e-13, 1e-10, n)      # m²/s
        C_surf  = np.random.uniform(2000, 25000, n)        # ppm (free Cl⁻)
        C_thr   = np.random.uniform(500, 8000, n)          # ppm threshold
        wc      = np.random.uniform(0.35, 0.80, n)
        temp    = np.random.uniform(-5, 35, n)             # °C

        # Arrhenius temperature correction on D (activation energy ~41.8 kJ/mol)
        T_ref = 293.15   # 20°C reference
        Ea    = 41800    # J/mol
        R     = 8.314
        D_eff = D_base * np.exp(-Ea / R * (1/(temp + 273.15) - 1/T_ref))

        t_years = np.array([
            ficks_initiation_time(cover[i], D_eff[i], C_surf[i], C_thr[i])
            for i in range(n)
        ])

        # Cap and add noise
        t_years = np.clip(t_years, 0.1, 300)
        t_years *= np.random.lognormal(0, 0.08, n)   # ±8% noise

        df = pd.DataFrame({
            "cover_depth_mm":      cover,
            "diffusion_coeff":     D_base,
            "surface_cl_ppm":      C_surf,
            "threshold_cl_ppm":    C_thr,
            "wc_ratio":            wc,
            "temperature_celsius": temp,
            "initiation_years":    t_years,
        })
        return df.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    def train(self, df):
        X = df[self.FEATURE_NAMES].values
        y = np.log1p(df["initiation_years"].values)    # log-transform skewed target

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model = GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                               learning_rate=0.05, random_state=42)
        self.model.fit(X_tr_s, y_tr)

        y_pred = self.model.predict(X_te_s)
        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)

        print(f"\n[Model 1] Chloride Initiation Period Predictor")
        print(f"  R²   : {r2:.4f}")
        print(f"  RMSE : {rmse:.4f}  (log-years space)")
        print(f"  MAE  : {mae:.4f}  (log-years space)")

        self.is_trained = True
        return {"r2": r2, "rmse": rmse, "mae": mae,
                "X_te": X_te, "y_te": y_te, "y_pred": y_pred}

    # ------------------------------------------------------------------
    def predict(self, scenario: dict):
        X = np.array([[scenario[f] for f in self.FEATURE_NAMES]])
        X_s = self.scaler.transform(X)
        return float(np.expm1(self.model.predict(X_s)[0]))

    # ------------------------------------------------------------------
    def feature_importance(self):
        return dict(zip(self.FEATURE_NAMES, self.model.feature_importances_))


# =============================================================================
# MODEL 2 — CORROSION TYPE CLASSIFIER  (Local vs. General)
# =============================================================================

class CorrosionTypeClassifier:
    """
    Binary classifier: predicts whether corrosion will be LOCAL (pitting)
    or GENERAL (uniform) given pore solution chemistry.

    Threshold: Cl⁻/OH⁻ > 0.61 => LOCAL  (Hausmann 1967, confirmed by Tuutti)

    Features
    --------
    free_cl_ppm         : free chloride concentration in pore solution (ppm)
    oh_equiv_per_litre  : OH⁻ concentration (equiv/L)
    cement_type_enc     : encoded cement type (0=OPC, 1=Blended, 2=Slag)
    wc_ratio            : water/cement ratio
    temperature_celsius : affects ion mobility
    relative_humidity   : affects pore solution concentration
    """

    FEATURE_NAMES = [
        "free_cl_ppm",
        "oh_equiv_per_litre",
        "cement_type_enc",
        "wc_ratio",
        "temperature_celsius",
        "relative_humidity",
    ]
    CLASSES = ["General", "Local"]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    # ------------------------------------------------------------------
    def generate_data(self, n=3000):
        free_cl   = np.random.uniform(100, 25000, n)      # ppm
        oh        = np.random.uniform(0.05, 1.2, n)       # equiv/L
        cem_type  = np.random.randint(0, 3, n)             # 0,1,2
        wc        = np.random.uniform(0.35, 0.80, n)
        temp      = np.random.uniform(5, 40, n)
        rh        = np.random.uniform(50, 100, n)

        # Physics label — add realistic noise around the 0.61 threshold
        ratio = np.array([cl_oh_ratio(free_cl[i], oh[i]) for i in range(n)])

        # Noise to create a non-perfectly-separable boundary
        ratio_noisy = ratio * np.random.lognormal(0, 0.15, n)

        # OPC cement type allows slightly higher threshold (chemical buffering)
        threshold = np.where(cem_type == 0, 0.68,
                    np.where(cem_type == 1, 0.61, 0.55))

        label = (ratio_noisy > threshold).astype(int)

        df = pd.DataFrame({
            "free_cl_ppm":          free_cl,
            "oh_equiv_per_litre":   oh,
            "cement_type_enc":      cem_type.astype(float),
            "wc_ratio":             wc,
            "temperature_celsius":  temp,
            "relative_humidity":    rh,
            "corrosion_type":       label,
            "cl_oh_ratio":          ratio,
        })
        return df

    # ------------------------------------------------------------------
    def train(self, df):
        X = df[self.FEATURE_NAMES].values
        y = df["corrosion_type"].values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                learning_rate=0.08, random_state=42)
        self.model.fit(X_tr_s, y_tr)
        y_pred = self.model.predict(X_te_s)

        print(f"\n[Model 2] Corrosion Type Classifier (Local vs. General)")
        print(classification_report(y_te, y_pred, target_names=self.CLASSES))

        self.is_trained = True
        return {"y_te": y_te, "y_pred": y_pred, "X_te": X_te}

    # ------------------------------------------------------------------
    def predict_proba(self, scenario: dict):
        X = np.array([[scenario[f] for f in self.FEATURE_NAMES]])
        X_s = self.scaler.transform(X)
        prob = self.model.predict_proba(X_s)[0]
        return {"General": prob[0], "Local": prob[1],
                "prediction": self.CLASSES[int(self.model.predict(X_s)[0])]}

    # ------------------------------------------------------------------
    def feature_importance(self):
        return dict(zip(self.FEATURE_NAMES, self.model.feature_importances_))


# =============================================================================
# MODEL 3 — CHLORIDE CONCENTRATION PROFILE PREDICTOR
# =============================================================================

class ChlorideProfileModel:
    """
    Predicts free Cl⁻ concentration (ppm) at a given depth and time —
    i.e., the full chloride profile (Figure 6 in Tuutti).

    Features
    --------
    depth_mm            : depth from surface (mm)
    exposure_years      : time since first chloride exposure (years)
    surface_cl_ppm      : surface free chloride concentration (ppm)
    diffusion_coeff     : effective Cl⁻ diffusivity (m²/s)
    wc_ratio            : water/cement ratio
    cement_type_enc     : 0=OPC, 1=Blended (affects binding capacity)
    temperature_celsius : temperature
    """

    FEATURE_NAMES = [
        "depth_mm",
        "exposure_years",
        "surface_cl_ppm",
        "diffusion_coeff",
        "wc_ratio",
        "cement_type_enc",
        "temperature_celsius",
    ]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    # ------------------------------------------------------------------
    def _fick_profile(self, x_mm, t_years, C_s, D):
        """Analytical Fick's 2nd law profile."""
        x = x_mm / 1000.0
        t = t_years * 365.25 * 24 * 3600
        if t <= 0:
            return C_s if x == 0 else 0.0
        arg = x / (2.0 * np.sqrt(D * t))
        return C_s * erfc(arg)

    # ------------------------------------------------------------------
    def generate_data(self, n=5000):
        depth   = np.random.uniform(0, 60, n)          # mm
        t_exp   = np.random.uniform(0.5, 50, n)        # years
        C_surf  = np.random.uniform(3000, 22000, n)    # ppm
        D       = np.random.uniform(1e-13, 5e-11, n)   # m²/s
        wc      = np.random.uniform(0.35, 0.80, n)
        cem     = np.random.randint(0, 2, n).astype(float)
        temp    = np.random.uniform(5, 35, n)

        # Temperature correction on D
        Ea, R, T_ref = 41800, 8.314, 293.15
        D_eff = D * np.exp(-Ea / R * (1/(temp + 273.15) - 1/T_ref))

        # Blended cement binds more Cl⁻ -> lower free Cl⁻ (scale factor)
        binding_factor = np.where(cem == 1, 0.75, 1.0)

        cl_conc = np.array([
            self._fick_profile(depth[i], t_exp[i], C_surf[i], D_eff[i])
            for i in range(n)
        ]) * binding_factor

        cl_conc *= np.random.lognormal(0, 0.10, n)   # 10% noise
        cl_conc = np.clip(cl_conc, 0, None)

        df = pd.DataFrame({
            "depth_mm":             depth,
            "exposure_years":       t_exp,
            "surface_cl_ppm":       C_surf,
            "diffusion_coeff":      D,
            "wc_ratio":             wc,
            "cement_type_enc":      cem,
            "temperature_celsius":  temp,
            "free_cl_ppm":          cl_conc,
        })
        return df

    # ------------------------------------------------------------------
    def train(self, df):
        X = df[self.FEATURE_NAMES].values
        y = np.log1p(df["free_cl_ppm"].values)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.model = RandomForestRegressor(n_estimators=300, max_depth=12,
                                           min_samples_leaf=5, random_state=42,
                                           n_jobs=-1)
        self.model.fit(X_tr_s, y_tr)
        y_pred = self.model.predict(X_te_s)

        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)

        print(f"\n[Model 3] Chloride Concentration Profile Predictor")
        print(f"  R²   : {r2:.4f}")
        print(f"  RMSE : {rmse:.4f}  (log-ppm space)")
        print(f"  MAE  : {mae:.4f}  (log-ppm space)")

        self.is_trained = True
        return {"r2": r2, "rmse": rmse, "mae": mae,
                "X_te": X_te, "y_te": y_te, "y_pred": y_pred}

    # ------------------------------------------------------------------
    def predict_profile(self, base_scenario: dict, depths=np.arange(0, 65, 2)):
        """
        Given a scenario dict (without depth_mm), predict full Cl⁻ profile
        over a range of depths.
        """
        rows = []
        for d in depths:
            s = dict(base_scenario)
            s["depth_mm"] = d
            X = np.array([[s[f] for f in self.FEATURE_NAMES]])
            X_s = self.scaler.transform(X)
            pred = float(np.expm1(self.model.predict(X_s)[0]))
            rows.append({"depth_mm": d, "free_cl_ppm": pred})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def feature_importance(self):
        return dict(zip(self.FEATURE_NAMES, self.model.feature_importances_))


# =============================================================================
# COMBINED PIPELINE
# =============================================================================

class ChlorideCombinedPipeline:
    """
    Trains all three chloride sub-models and produces a unified results report
    + visualisation dashboard matching the style of rizzhubsane/tuuti.
    """

    def __init__(self):
        self.m1 = ChlorideInitiationModel()
        self.m2 = CorrosionTypeClassifier()
        self.m3 = ChlorideProfileModel()
        self.results = {}

    # ------------------------------------------------------------------
    def run_all(self, n_samples=2000, save_plots=True):
        print("=" * 60)
        print("  CHLORIDE ML MODULE — Tuutti (SP 65-13)")
        print("=" * 60)

        # --- Generate & train ---
        df1 = self.m1.generate_data(n_samples)
        r1  = self.m1.train(df1)

        df2 = self.m2.generate_data(n_samples)
        r2  = self.m2.train(df2)

        df3 = self.m3.generate_data(n_samples)
        r3  = self.m3.train(df3)

        self.results = {"model1": r1, "model2": r2, "model3": r3}

        # --- Example predictions ---
        self._print_example_predictions()

        # --- Plots ---
        if save_plots:
            self._plot_dashboard(r1, r2, r3, df1, df2, df3)

        print("\n✅  All three chloride models trained successfully.")
        return self.results

    # ------------------------------------------------------------------
    def _print_example_predictions(self):
        print("\n" + "=" * 60)
        print("  EXAMPLE PREDICTIONS")
        print("=" * 60)

        # M1 — Initiation
        scenarios_m1 = [
            {"label": "Coastal bridge (high exposure)",
             "cover_depth_mm": 30, "diffusion_coeff": 1e-11,
             "surface_cl_ppm": 18000, "threshold_cl_ppm": 1500,
             "wc_ratio": 0.55, "temperature_celsius": 22},
            {"label": "Inland structure (low exposure)",
             "cover_depth_mm": 60, "diffusion_coeff": 1e-12,
             "surface_cl_ppm": 5000, "threshold_cl_ppm": 3000,
             "wc_ratio": 0.40, "temperature_celsius": 15},
        ]
        print("\n[Model 1] Predicted Chloride Initiation Period:")
        for s in scenarios_m1:
            t = self.m1.predict(s)
            print(f"  {s['label']:<40} → {t:>7.1f} years")

        # M2 — Corrosion type
        scenarios_m2 = [
            {"label": "Atlantic seawater exposure",
             "free_cl_ppm": 15000, "oh_equiv_per_litre": 0.3,
             "cement_type_enc": 0, "wc_ratio": 0.55,
             "temperature_celsius": 20, "relative_humidity": 85},
            {"label": "Baltic Sea (dilute chloride)",
             "free_cl_ppm": 3000, "oh_equiv_per_litre": 0.8,
             "cement_type_enc": 1, "wc_ratio": 0.42,
             "temperature_celsius": 15, "relative_humidity": 75},
        ]
        print("\n[Model 2] Predicted Corrosion Type:")
        for s in scenarios_m2:
            res = self.m2.predict_proba(s)
            print(f"  {s['label']:<40} → {res['prediction']}  "
                  f"(P_local={res['Local']:.2f}, P_general={res['General']:.2f})")

        # M3 — Profile (print at selected depths)
        base_m3 = {
            "exposure_years": 10, "surface_cl_ppm": 12000,
            "diffusion_coeff": 5e-12, "wc_ratio": 0.50,
            "cement_type_enc": 0, "temperature_celsius": 20,
        }
        profile = self.m3.predict_profile(base_m3, depths=[0, 10, 20, 30, 40, 50])
        print("\n[Model 3] Predicted Chloride Profile (OPC, w/c=0.50, 10 yr exposure):")
        print("  Depth (mm)  |  Free Cl⁻ (ppm)")
        print("  " + "-" * 30)
        for _, row in profile.iterrows():
            print(f"  {int(row.depth_mm):>8} mm  |  {row.free_cl_ppm:>10.0f}")

    # ------------------------------------------------------------------
    def _plot_dashboard(self, r1, r2, r3, df1, df2, df3):
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "Chloride ML Module — Tuutti SP 65-13\n"
            "Three Dedicated Sub-Models for Chloride-Induced Corrosion",
            fontsize=14, fontweight="bold", y=0.98
        )
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ---- Row 1: Model 1 ----
        ax1a = fig.add_subplot(gs[0, 0])   # actual vs predicted
        ax1b = fig.add_subplot(gs[0, 1])   # feature importance
        ax1c = fig.add_subplot(gs[0, 2])   # initiation time vs cover depth

        y_te1 = np.expm1(r1["y_te"])
        y_pr1 = np.expm1(r1["y_pred"])
        ax1a.scatter(y_te1, y_pr1, alpha=0.3, s=8, color="#2196F3")
        lim = [0, np.percentile(y_te1, 99)]
        ax1a.plot(lim, lim, "r--", lw=1.5)
        ax1a.set_xlabel("Actual initiation (years)")
        ax1a.set_ylabel("Predicted initiation (years)")
        ax1a.set_title(f"Model 1: Initiation Period\nR²={r2_score(y_te1, y_pr1):.3f}")
        ax1a.set_xlim(lim); ax1a.set_ylim(lim)

        fi1 = self.m1.feature_importance()
        ax1b.barh(list(fi1.keys()), list(fi1.values()), color="#2196F3")
        ax1b.set_xlabel("Importance")
        ax1b.set_title("Model 1: Feature Importance")

        # Theoretical Fick's curve overlay
        covers = np.linspace(10, 100, 50)
        t_fick = [ficks_initiation_time(c, 1e-12, 15000, 2000) for c in covers]
        t_fick = np.clip(t_fick, 0, 200)
        ax1c.plot(covers, t_fick, "k-", lw=2, label="Fick's Law (D=1e-12)")
        # Scatter a subset of training data
        subset = df1.sample(300, random_state=0)
        sc = ax1c.scatter(subset["cover_depth_mm"], subset["initiation_years"],
                          c=np.log10(subset["diffusion_coeff"]),
                          cmap="coolwarm", alpha=0.4, s=10)
        plt.colorbar(sc, ax=ax1c, label="log₁₀(D)")
        ax1c.set_xlabel("Cover depth (mm)")
        ax1c.set_ylabel("Initiation time (years)")
        ax1c.set_title("Model 1: Cover Depth vs Initiation\n(colour = diffusivity)")
        ax1c.legend(fontsize=7)

        # ---- Row 2: Model 2 ----
        ax2a = fig.add_subplot(gs[1, 0])   # confusion matrix
        ax2b = fig.add_subplot(gs[1, 1])   # feature importance
        ax2c = fig.add_subplot(gs[1, 2])   # Cl⁻/OH⁻ decision boundary

        cm = confusion_matrix(r2["y_te"], r2["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["General", "Local"])
        disp.plot(ax=ax2a, colorbar=False, cmap="Blues")
        ax2a.set_title("Model 2: Confusion Matrix\n(Corrosion Type)")

        fi2 = self.m2.feature_importance()
        ax2b.barh(list(fi2.keys()), list(fi2.values()), color="#FF5722")
        ax2b.set_xlabel("Importance")
        ax2b.set_title("Model 2: Feature Importance")

        # Cl⁻/OH⁻ scatter (colour by label)
        df2["cl_oh"] = df2["free_cl_ppm"] / 1000.0 / (df2["oh_equiv_per_litre"] * 35.45)
        colors = df2["corrosion_type"].map({0: "#4CAF50", 1: "#F44336"})
        ax2c.scatter(df2["cl_oh"].clip(0, 3), df2["oh_equiv_per_litre"],
                     c=colors, alpha=0.2, s=6)
        ax2c.axvline(0.61, color="black", lw=1.5, ls="--", label="Threshold 0.61")
        from matplotlib.patches import Patch
        ax2c.legend(handles=[
            Patch(color="#4CAF50", label="General"),
            Patch(color="#F44336", label="Local"),
            plt.Line2D([0], [0], color="black", ls="--", label="Hausmann threshold"),
        ], fontsize=7)
        ax2c.set_xlabel("Cl⁻/OH⁻ ratio (g/L·equiv)")
        ax2c.set_ylabel("OH⁻ (equiv/L)")
        ax2c.set_title("Model 2: Decision Boundary\n(Cl⁻/OH⁻ vs OH⁻)")
        ax2c.set_xlim(0, 3)

        # ---- Row 3: Model 3 ----
        ax3a = fig.add_subplot(gs[2, 0])   # actual vs predicted
        ax3b = fig.add_subplot(gs[2, 1])   # feature importance
        ax3c = fig.add_subplot(gs[2, 2])   # predicted profiles at different times

        y_te3 = np.expm1(r3["y_te"])
        y_pr3 = np.expm1(r3["y_pred"])
        ax3a.scatter(y_te3, y_pr3, alpha=0.3, s=8, color="#9C27B0")
        lim3 = [0, np.percentile(y_te3, 99)]
        ax3a.plot(lim3, lim3, "r--", lw=1.5)
        ax3a.set_xlabel("Actual Cl⁻ (ppm)")
        ax3a.set_ylabel("Predicted Cl⁻ (ppm)")
        ax3a.set_title(f"Model 3: Chloride Profile\nR²={r2_score(y_te3, y_pr3):.3f}")
        ax3a.set_xlim(lim3); ax3a.set_ylim(lim3)

        fi3 = self.m3.feature_importance()
        ax3b.barh(list(fi3.keys()), list(fi3.values()), color="#9C27B0")
        ax3b.set_xlabel("Importance")
        ax3b.set_title("Model 3: Feature Importance")

        # Predicted profiles at t=5, 15, 30, 50 years
        depths = np.arange(0, 62, 2)
        base = {"surface_cl_ppm": 15000, "diffusion_coeff": 2e-12,
                "wc_ratio": 0.50, "cement_type_enc": 0, "temperature_celsius": 20}
        cmap3 = plt.cm.viridis
        for i, yr in enumerate([5, 15, 30, 50]):
            base["exposure_years"] = yr
            prof = self.m3.predict_profile(base, depths=depths)
            ax3c.plot(prof["free_cl_ppm"], prof["depth_mm"],
                      color=cmap3(i / 4), label=f"{yr} yr", lw=2)

        ax3c.axvline(2000, color="red", ls="--", lw=1, label="Threshold ~2000 ppm")
        ax3c.set_xlabel("Free Cl⁻ (ppm)")
        ax3c.set_ylabel("Depth (mm)")
        ax3c.invert_yaxis()
        ax3c.set_title("Model 3: Chloride Profiles\n(OPC, w/c=0.50, varying time)")
        ax3c.legend(fontsize=7)

        plt.savefig("chloride_dashboard.png", dpi=150, bbox_inches="tight")
        print("\n  Saved: chloride_dashboard.png")
        plt.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pipeline = ChlorideCombinedPipeline()
    pipeline.run_all(n_samples=2000, save_plots=True)
