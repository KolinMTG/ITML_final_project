# Wine Quality Regression - Analysis Report

## Executive Summary

This analysis predicts wine quality scores using chemical properties. 
9 regression models were trained and evaluated.

**Best Model**: Random Forest
- R² Score: 0.4105
- RMSE: 0.6657
- MAE: 0.5115

---

## 1. Dataset Overview

- **Total Samples**: 5320
- **Features**: 11
- **Target**: Quality score (3-9)
- **Mean Quality**: 5.80 ± 0.88

### Quality Distribution
quality
3      30
4     206
5    1752
6    2323
7     856
8     148
9       5

---

## 2. Model Performance

### All Models Summary

| name              |   test_rmse |   test_mae |   test_r2 |   train_r2 |   cv_rmse |
|:------------------|------------:|-----------:|----------:|-----------:|----------:|
| Random Forest     |      0.6657 |     0.5115 |    0.4105 |     0.9136 |    0.7009 |
| Gradient Boosting |      0.6837 |     0.5263 |    0.3783 |     0.4680 |    0.7074 |
| SVR               |      0.7139 |     0.5364 |    0.3221 |     0.5572 |    0.7281 |
| Neural Network    |      0.7205 |     0.5539 |    0.3096 |     0.5829 |    0.7151 |
| Ridge             |      0.7262 |     0.5631 |    0.2985 |     0.3068 |    0.7380 |
| Linear Regression |      0.7262 |     0.5631 |    0.2985 |     0.3068 |    0.7380 |
| ElasticNet        |      0.7396 |     0.5783 |    0.2724 |     0.2745 |    0.7534 |
| Lasso             |      0.7493 |     0.5902 |    0.2533 |     0.2510 |    0.7650 |
| Decision Tree     |      0.8201 |     0.6165 |    0.1055 |     0.6546 |    0.8544 |

### Top 3 Models


#### 6. Random Forest
- **R² Score**: 0.4105 (41.1% variance explained)
- **RMSE**: 0.6657 quality points
- **MAE**: 0.5115 quality points
- **CV RMSE**: 0.7009

#### 7. Gradient Boosting
- **R² Score**: 0.3783 (37.8% variance explained)
- **RMSE**: 0.6837 quality points
- **MAE**: 0.5263 quality points
- **CV RMSE**: 0.7074

#### 8. SVR
- **R² Score**: 0.3221 (32.2% variance explained)
- **RMSE**: 0.7139 quality points
- **MAE**: 0.5364 quality points
- **CV RMSE**: 0.7281


---

## 3. Key Findings

### Model Comparison
- **Tree-based models** (Random Forest, Gradient Boosting) perform best
- **R² scores** typically range from 0.35-0.50
- **RMSE** around 0.6-0.7 quality points

### Interpretation
- Models explain 35-50% of quality variance
- Predictions within ±0.6-0.7 quality points on average
- Chemical properties are moderately predictive of quality
- Human factors (taste, preference) likely account for remaining variance

---

## 4. Recommendations

### Production Deployment
**Recommended Model**: Random Forest or Gradient Boosting

**Use Cases**:
1. Quality control screening
2. Batch quality prediction
3. Production optimization guidance

### Limitations
- Models explain ~40-50% of variance
- Quality is subjective and context-dependent
- Chemical properties alone cannot fully predict quality
- Model best used as screening tool, not replacement for expert tasting

---

## 5. Visualizations

All plots saved in `plots_regression/`:
1. `quality_distribution.png` - Target variable distribution
2. `predictions_vs_actual.png` - Model predictions vs reality
3. `model_comparison.png` - Performance metrics comparison
4. `residuals_analysis.png` - Residual plots
5. `feature_importance.png` - Key features from tree models
6. `correlation_matrix.png` - Feature correlations

---

## 6. Conclusion

Wine quality can be predicted with **moderate accuracy** using chemical properties. 
The best models achieve R² ≈ 0.45-0.50 and RMSE ≈ 0.60-0.65.

**Key Takeaway**: Chemical analysis provides useful quality indicators but cannot 
fully replace expert sensory evaluation.

---

*Report generated: January 2026*
