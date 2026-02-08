# Tuutti-AI: Modernizing Concrete Durability with Machine Learning

## 🏗️ Project Overview
This project modernizes the classical **Tuutti Model (1982)** for concrete durability by integrating it with **Machine Learning**.

Traditional civil engineering relies on static nomograms to predict when a structure will corrode. This project replaces those manual charts with a **Hybrid Physics-ML Predictor** capable of analyzing complex, multi-variable concrete mix designs to predict service life with high accuracy.

---

## 📄 The Theory: Tuutti's Model (1982)
The project is grounded in K. Tuutti's foundational paper, *"Service Life of Structures with Regard to Corrosion of Embedded Steel"*.

### Key Takeaways from the Paper:
1.  **The Two-Stage Model:** Service life is not a single event but the sum of two distinct phases:
    * **Initiation Stage ($t_{init}$):** The time it takes for $CO_2$ or Chlorides to penetrate the concrete cover and reach the steel.
    * **Propagation Stage ($t_{prop}$):** The active corrosion period leading to cracking.
2.  **The Square-Root Law:** Carbonation depth ($x$) is proportional to the square root of time ($x = K \sqrt{t}$).
3.  **The Role of Permeability:** The water/cement ratio ($w/c$) is the single most critical factor; diffusion resistance drops exponentially as $w/c$ increases.

> **Project Goal:** To use ML to predict the complex material coefficient ($K$) that Tuutti identified as "the least certain parameter", and then use his physical framework to project it forward in time.

---

## 🛣️ My Journey: From "Flatlines" to Physics
Building this model wasn't a straight line. Here is how the development process evolved:

### Phase 1: The "Pure ML" Failure
* **Attempt:** I initially trained a Gradient Boosting model to predict **Carbonation Depth** directly from inputs (w/c, humidity, time).
* **The Problem:** Tree-based models cannot extrapolate. When asked to predict beyond the training data (e.g., Year 50), the model "flatlined," predicting constant depth forever. It didn't understand that time moves forward.

### Phase 2: The "Hybrid K" Solution
* **The Pivot:** Instead of asking AI to predict *Depth*, I asked it to predict the **Rate Coefficient ($K$)**.
* **The Math:**
    $$K_{experimental} = \frac{Depth_{measured}}{\sqrt{Time}}$$
* **The Workflow:**
    1.  I calculated the observed $K$ for all 2,163 samples in the dataset.
    2.  I trained the ML model to predict $K$ based on mix design ($w/c$, cover, environment).
    3.  I plugged the predicted $K$ back into Tuutti's physics equation: $Depth = K_{pred} \times \sqrt{Time}$.
* **Result:** This solved the flatline issue, producing smooth, realistic parabolic curves that respect the laws of physics.

---

## 📊 Final Results
The model was validated using a real-world dataset of **2,163 experimental samples**, achieving an **$R^2$ accuracy of 0.76** (high for noisy experimental data).

The most powerful validation came from a "Sensitivity Analysis" comparing two common mix designs:

### The "Before & After" Comparison

| Parameter | **Scenario A: Standard Mix** | **Scenario B: High Performance** |
| :--- | :--- | :--- |
| **Design** | $w/c = 0.55$, Cover = 30mm | $w/c = 0.40$, Cover = 40mm |
| **Predicted Rate ($K$)** | **7.28** $mm/\sqrt{year}$ | **1.62** $mm/\sqrt{year}$ |
| **Predicted Service Life** | 🔴 **17.1 Years** | 🟢 **> 150 Years** |

**Conclusion:** The model correctly identified that a small improvement in mix quality ($w/c$ 0.55 $\to$ 0.40) does not just linearly improve life—it exponentially extends it, effectively making the structure immune to carbonation for over a century.

---

## 🛠️ Tech Stack & How to Run
* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`
* **Algorithm:** Gradient Boosting Regressor (sklearn)

### Quick Start
1.  Clone the repo:
    ```bash
    git clone [https://github.com/yourusername/tuutti-ai.git](https://github.com/yourusername/tuutti-ai.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib openpyxl
    ```
3.  Run the predictor:
    ```bash
    python script.py
    ```

---

## 🧠 Credits
* **Original Theory:** K. Tuutti, *"Service Life of Structures with Regard to Corrosion of Embedded Steel"*, ACI, 1982.
* **Dataset:** Concrete Carbonation Data (Mendeley Data).
