# Quick Reference Guide: Concrete Service Life Prediction

## 🚀 Quick Start

### Input Parameters Required

| Parameter | Unit | Typical Range | Description |
|-----------|------|---------------|-------------|
| Cover Depth | mm | 20-100 | Concrete covering reinforcement |
| W/C Ratio | - | 0.35-0.70 | Water to cement ratio |
| CO₂ Concentration | % | 0.03-0.08 | Atmospheric (0.032% typical) |
| Chloride Concentration | % weight | 0.0-1.2 | From sea/de-icing salts |
| Diffusion Coefficient | m²/s | 1e-8 to 1e-6 | Material permeability |
| Temperature | °C | 0-40 | Average environmental temp |
| Relative Humidity | % | 40-100 | Average RH at structure |
| Oxygen Supply | scale | 1-10 | Relative availability |
| Electrical Resistance | Ω·m | 10-200 | Concrete quality indicator |

---

## 📈 Decision Matrix for Design

### Target Service Life Guidelines

| Structure Type | Minimum Service Life | Recommended Parameters |
|----------------|---------------------|----------------------|
| Temporary structures | 10 years | w/c ≤ 0.60, cover ≥ 25mm |
| Residential buildings | 50 years | w/c ≤ 0.50, cover ≥ 40mm |
| Commercial buildings | 75 years | w/c ≤ 0.45, cover ≥ 50mm |
| Infrastructure (bridges) | 100 years | w/c ≤ 0.40, cover ≥ 65mm |
| Critical infrastructure | 120+ years | w/c ≤ 0.35, cover ≥ 75mm |

---

## ⚠️ Critical Thresholds

### Immediate Concern Levels

**HIGH RISK CONDITIONS:**
- Chloride > 0.6% by weight
- RH consistently 75-85%
- Temperature > 30°C average
- Cover depth < 25mm
- W/C ratio > 0.60

**MODERATE RISK CONDITIONS:**
- Chloride 0.3-0.6% by weight
- RH 65-75%
- Temperature 20-30°C
- Cover depth 25-40mm
- W/C ratio 0.50-0.60

**LOW RISK CONDITIONS:**
- Chloride < 0.3% by weight
- RH < 65% or > 95%
- Temperature < 20°C
- Cover depth > 40mm
- W/C ratio < 0.50

---

## 🌍 Environmental Classifications

### Exposure Class Guide

**XC1-XC4: Carbonation Induced Corrosion**
- XC1 (Dry): Indoor, low humidity
- XC2 (Wet, rarely dry): Long-term water contact
- XC3 (Moderate humidity): Sheltered outdoor
- XC4 (Cyclic wet/dry): Exposed outdoor

**XS1-XS3: Chloride from Seawater**
- XS1 (Airborne salt): > 1km from coast
- XS2 (Permanently submerged): Below waterline
- XS3 (Tidal/splash zones): Highest risk

**XD1-XD3: Chloride from De-icing**
- XD1 (Moderate): Occasional salting
- XD2 (Wet, rarely dry): Road elements
- XD3 (Cyclic wet/dry): Bridge decks

---

## 🔧 Design Recommendations by Environment

### Coastal Structures (High Chloride)
```
✓ Cover depth: ≥ 50mm (splash zone: ≥65mm)
✓ W/C ratio: ≤ 0.40
✓ Add corrosion inhibitors
✓ Consider stainless steel reinforcement
✓ Apply protective coatings
✓ Target electrical resistance: > 150 Ω·m
```

### Urban Structures (High CO₂)
```
✓ Cover depth: ≥ 40mm
✓ W/C ratio: ≤ 0.45
✓ Use blended cements (slag, fly ash)
✓ Ensure proper curing
✓ Consider carbonation-resistant concrete
```

### Cold Climate with De-icing
```
✓ Cover depth: ≥ 50mm
✓ W/C ratio: ≤ 0.40
✓ Low permeability essential
✓ Air entrainment for freeze-thaw
✓ Waterproofing membranes
✓ Drainage design critical
```

### Hot, Humid Tropical
```
✓ Cover depth: ≥ 45mm
✓ W/C ratio: ≤ 0.42
✓ Control RH through ventilation
✓ Higher cement content
✓ Good quality control crucial
```

---

## 📊 Feature Priority for Different Scenarios

### New Construction (Design Phase)
**Priority Order:**
1. W/C Ratio (controllable, permanent effect)
2. Cover Depth (design parameter)
3. Cement Type/Quality
4. Expected environmental exposure

### Existing Structure Assessment
**Priority Order:**
1. Relative Humidity (dominant factor)
2. Temperature monitoring
3. Chloride ingress testing
4. Carbonation depth measurement

### Repair/Retrofit Decisions
**Focus On:**
1. Environmental control (humidity, drainage)
2. Protective coatings (reduce oxygen/water)
3. Cathodic protection (if severe)
4. Material quality for repairs

---

## 🎯 Practical Calculation Examples

### Example 1: Residential Building Design
```
Location: Moderate urban environment
Target: 75 years service life

INPUT PARAMETERS:
- Cover depth: 45 mm
- W/C ratio: 0.45
- CO₂: 0.05% (urban)
- Chloride: 0.05% (low)
- Diffusion: 3e-7 m²/s
- Temperature: 18°C
- RH: 70%
- O₂ supply: 6
- Resistance: 140 Ω·m

EXPECTED OUTCOME: ~50-80 years
RECOMMENDATION: ACCEPTABLE for residential use
```

### Example 2: Bridge Deck with De-icing
```
Location: Highway bridge, winter salting
Target: 100 years service life

INPUT PARAMETERS:
- Cover depth: 65 mm
- W/C ratio: 0.38
- CO₂: 0.04%
- Chloride: 0.40% (moderate exposure)
- Diffusion: 2e-7 m²/s
- Temperature: 12°C (average)
- RH: 75%
- O₂ supply: 7
- Resistance: 170 Ω·m

EXPECTED OUTCOME: ~40-60 years
RECOMMENDATION: Consider additional protection
  - Corrosion inhibitors
  - Surface sealer
  - Enhanced drainage
```

### Example 3: Coastal Structure
```
Location: Marine splash zone
Target: 50 years minimum

CRITICAL INPUTS:
- Cover depth: 75 mm (MINIMUM)
- W/C ratio: 0.35 (HIGH QUALITY)
- Chloride: 0.90% (high exposure)
- RH: 85% (high)
- Temperature: 25°C
- Resistance: 180 Ω·m (excellent)

EXPECTED OUTCOME: ~15-25 years
RECOMMENDATION: Additional measures REQUIRED
  - Stainless steel reinforcement
  - Protective coatings (epoxy)
  - Sacrificial anode system
  - Regular maintenance program
```

---

## 🔍 Inspection & Monitoring Recommendations

### Regular Monitoring Schedule

**Year 0-10:**
- Visual inspection: Annual
- Corrosion potential mapping: Every 3 years
- Chloride profiling: Every 5 years

**Year 10-25:**
- Visual inspection: Every 6 months
- Corrosion potential: Every 2 years
- Chloride profiling: Every 3 years
- Carbonation testing: Every 5 years

**Year 25+:**
- Visual inspection: Quarterly
- Corrosion potential: Annual
- Chloride profiling: Every 2 years
- Structural assessment: Every 5 years

### Warning Signs

🚨 **Immediate Action Required:**
- Visible rust staining
- Spalling concrete
- Exposed reinforcement
- Wide cracks (>0.3mm)

⚠️ **Investigation Needed:**
- Surface discoloration
- Fine cracks
- Efflorescence
- Moisture accumulation

---

## 💡 Cost-Benefit Considerations

### Prevention vs Repair Costs

| Strategy | Relative Cost | Typical Increase in Service Life |
|----------|--------------|--------------------------------|
| Reduce w/c by 0.05 | +5% initial | +15-25% service life |
| Increase cover by 15mm | +2% initial | +20-30% service life |
| Corrosion inhibitors | +3% initial | +30-50% service life |
| Protective coating | +8% initial | +40-60% service life |
| Stainless steel rebar | +200% material | +200-300% service life |
| Major repair/retrofit | +60-80% initial | Variable recovery |

**Rule of Thumb:** Every $1 spent on quality at construction saves $5-10 in future repairs.

---

## 📞 When to Consult a Specialist

Seek expert consultation when:
- ✓ Predicted service life < Target by >20%
- ✓ Chloride levels > 0.6% detected
- ✓ Visible corrosion damage present
- ✓ Critical infrastructure (bridges, hospitals)
- ✓ Uncertain environmental classification
- ✓ Special exposure conditions
- ✓ Heritage/historical structures

---

## 📚 Additional Resources

### Testing Standards
- **ASTM C876**: Corrosion potentials
- **ASTM C1152**: Chloride penetration
- **ASTM C1556**: Chloride diffusion
- **ASTM C1543**: Electrical resistance

### Design Codes
- **ACI 318**: Building Code Requirements
- **EN 206**: Concrete specifications
- **BS 8500**: Durability requirements
- **fib Model Code**: Service life design

---

## ⚙️ Model Limitations

**This model does NOT account for:**
- Freeze-thaw damage
- Alkali-silica reaction (ASR)
- Sulfate attack
- Mechanical loading effects
- Construction defects
- Poor workmanship
- Unusual chemical exposure

**Always consider:**
- Local experience and historical performance
- Multiple deterioration mechanisms
- Safety factors for critical structures
- Regular inspection and maintenance

---

*This quick reference is based on the ML model implementation of K. Tuutti's research and should be used in conjunction with appropriate design codes and professional engineering judgment.*

*Last Updated: February 6, 2026*
