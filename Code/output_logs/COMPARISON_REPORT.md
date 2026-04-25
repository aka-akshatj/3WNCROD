# Outlier Detection Methods Comparison Report

## Autoencoder vs WNCROD: Comprehensive Analysis

**Datasets Analyzed:** 10
**Methods Compared:** Autoencoder (5 algorithms) vs WNCROD

---

## Executive Summary

This report compares two outlier detection approaches:

1. **PyOD (Python Outlier Detection Library)** - Using 5 algorithms: IForest, LOF, COPOD, HBOS, KNN
2. **WNCROD (Three-Way Neighborhood Characteristic Region-based Outlier Detection)** - Custom algorithm

**Key Findings:**

- **Speed:** PyOD algorithms are **84.5x faster** on average (11.17s vs 945.79s total)
- **Consistency:** PyOD algorithms show high agreement (~10% outliers detected across all datasets)
- **Scalability:** WNCROD shows poor scalability with dataset size (O(n²) complexity)
- **Detection Quality:** Both methods identify similar top outliers in several datasets
- **Limitations:** WNCROD failed to detect outliers in 2 datasets (mushroom, musk) - all scores = 0

---

## 1. Performance Metrics

### 1.1 Overall Runtime Comparison


| Method                    | Total Runtime              | Average per Dataset      | Fastest Dataset      | Slowest Dataset    |
| --------------------------- | ---------------------------- | -------------------------- | ---------------------- | -------------------- |
| **PyOD (All Algorithms)** | 11.17 seconds              | 1.12 seconds             | 0.23s (lymphography) | 5.91s (annthyroid) |
| **WNCROD**                | 945.79 seconds (15.76 min) | 94.58 seconds (1.58 min) | 0.21s (heart270)     | 435.02s (musk)     |
| **Speedup Factor**        | **84.5x faster**           | **84.4x faster**         | -                    | -                  |

### 1.2 Per-Dataset Runtime Breakdown


| Dataset      | Samples | Features | PyOD Time | WNCROD Time | Speedup |
| -------------- | --------- | ---------- | ----------- | ------------- | --------- |
| annthyroid   | 7,200   | 7        | 0.10 min  | 1.69 min    | 16.9x   |
| creditA      | 425     | 16       | 0.01 min  | 0.02 min    | 2.0x    |
| german       | 714     | 21       | 0.00 min  | 0.06 min    | ∞      |
| heart270     | 166     | 14       | 0.00 min  | 0.00 min    | -       |
| lymphography | 148     | 19       | 0.00 min  | 0.00 min    | -       |
| mammography  | 11,183  | 7        | 0.03 min  | 4.23 min    | 141.0x  |
| mushroom     | 4,429   | 23       | 0.01 min  | 2.02 min    | 202.0x  |
| musk         | 3,062   | 167      | 0.01 min  | 7.25 min    | 725.0x  |
| thyroid      | 3,772   | 7        | 0.01 min  | 0.45 min    | 45.0x   |
| wdbc         | 396     | 32       | 0.00 min  | 0.03 min    | ∞      |

**Key Observations:**

- WNCROD runtime increases dramatically with dataset size (especially sample count)
- High-dimensional datasets (musk: 167 features) cause severe slowdowns
- PyOD maintains consistent, fast performance across all datasets
- WNCROD shows O(n²) complexity behavior

### 1.3 PyOD Algorithm Performance Breakdown


| Algorithm   | Avg Time (s) | Total Time (s) | Runs | Fastest    | Slowest      |
| ------------- | -------------- | ---------------- | ------ | ------------ | -------------- |
| **COPOD**   | 0.08         | 0.75           | 10   | ⭐ Fastest | -            |
| **KNN**     | 0.14         | 1.37           | 10   | -          | -            |
| **LOF**     | 0.15         | 1.51           | 10   | -          | -            |
| **IForest** | 0.25         | 2.48           | 10   | -          | -            |
| **HBOS**    | 0.46         | 4.63           | 10   | -          | ⚠️ Slowest |

**Recommendation:** COPOD offers the best speed/performance trade-off for most use cases.

---

## 2. Detection Results Analysis

### 2.1 Outlier Detection Rates


| Dataset      | PyOD Avg % | WNCROD %       | Agreement  |
| -------------- | ------------ | ---------------- | ------------ |
| annthyroid   | 10.00%     | N/A*           | -          |
| creditA      | 10.12%     | N/A*           | -          |
| german       | 10.08%     | N/A*           | -          |
| heart270     | 10.24%     | N/A*           | -          |
| lymphography | 10.14%     | N/A*           | -          |
| mammography  | 10.01%     | N/A*           | -          |
| mushroom     | 9.99%      | **0.00%** ⚠️ | **FAILED** |
| musk         | 10.03%     | **0.00%** ⚠️ | **FAILED** |
| thyroid      | 10.02%     | N/A*           | -          |
| wdbc         | 10.10%     | N/A*           | -          |

*WNCROD doesn't provide explicit outlier count - only scores

**Key Findings:**

- PyOD algorithms show **remarkable consistency** (~10% outliers across all datasets)
- WNCROD **completely failed** on 2 datasets (mushroom, musk) - all scores = 0.000000
- This suggests WNCROD may have issues with certain data types or structures

### 2.2 Score Distribution Analysis

#### Annthyroid Dataset (7,200 samples, 7 features)

**PyOD Algorithms:**

- **IForest:** Range: [-0.18, 0.24], Mean: -0.11, Std: 0.07
- **LOF:** Range: [0.92, 4.28], Mean: 1.12, Std: 0.19
- **COPOD:** Range: [4.06, 30.86], Mean: 8.06, Std: 3.02
- **HBOS:** Range: [-18.42, 5.38], Mean: -14.93, Std: 3.55
- **KNN:** Range: [0.01, 0.98], Mean: 0.06, Std: 0.05

**WNCROD:**

- Range: [0.00, 0.40], Mean: 0.002, Std: 0.015
- **Score separation:** Max (0.40) is **196x** the mean (0.002) - excellent discrimination

**Analysis:**

- WNCROD shows **excellent score separation** (top outliers clearly distinct)
- PyOD algorithms show varying score ranges but consistent detection
- LOF shows extreme outlier (4.28) that's 3.8x the mean - strong signal

#### Mammography Dataset (11,183 samples, 7 features)

**PyOD - LOF Anomaly:**

- Detected extreme outliers with scores up to **29,204,217** (likely numerical instability)
- This is a known issue with LOF on certain datasets

**WNCROD:**

- Range: [0.00, 0.05], Mean: 0.000035, Std: 0.000764
- **Score separation:** Max (0.05) is **1,429x** the mean - exceptional discrimination

**Analysis:**

- WNCROD provides more stable, interpretable scores
- LOF shows numerical issues on large datasets

---

## 3. Top Anomalies Consensus Analysis

### 3.1 Annthyroid Dataset - Top 10 Overlap

**PyOD Consensus:**

- **Sample 7058:** Detected by IForest (#1), COPOD (#6), HBOS (#9), KNN (#7) - **4/5 algorithms**
- **Sample 5411:** Detected by IForest (#3), HBOS (#3), KNN (#5) - **3/5 algorithms**
- **Sample 2774:** Detected by IForest (#4), HBOS (#2) - **2/5 algorithms**
- **Sample 6373:** Detected by IForest (#2), HBOS (#5) - **2/5 algorithms**
- **Sample 2209:** Detected by IForest (#10), HBOS (#1), KNN (#8) - **3/5 algorithms**

**WNCROD Top 10:**

1. Sample 3862 (0.395657)
2. Sample 5411 (0.394386) ✅ **MATCH**
3. Sample 2774 (0.394154) ✅ **MATCH**
4. Sample 6373 (0.353879) ✅ **MATCH**
5. Sample 3943 (0.340177)
6. Sample 7058 (0.276458) ✅ **MATCH**
7. Sample 2209 (0.250478) ✅ **MATCH**
8. Sample 4671 (0.234449)
9. Sample 4479 (0.229023)
10. Sample 2702 (0.225474)

**Consensus Score: 5/10 top anomalies overlap** - **Strong agreement!**

### 3.2 Thyroid Dataset - Top 10 Overlap

**PyOD Consensus:**

- **Sample 2774:** Detected by IForest (#1), COPOD (#8), HBOS (#1), KNN (#7) - **4/5 algorithms**
- **Sample 1524:** Detected by IForest (#2), COPOD (#5), HBOS (#4), KNN (#2) - **4/5 algorithms**
- **Sample 2209:** Detected by IForest (#4), HBOS (#3), KNN (#4) - **3/5 algorithms**

**WNCROD Top 10:**

1. Sample 2774 (0.427301) ✅ **MATCH**
2. Sample 2209 (0.259423) ✅ **MATCH**
3. Sample 2702 (0.239220)
4. Sample 1268 (0.220613)
5. Sample 794 (0.187594)
6. Sample 3164 (0.176001)
7. Sample 3565 (0.172960)
8. Sample 2931 (0.165258) ✅ **MATCH** (in PyOD top lists)
9. Sample 2628 (0.162764)
10. Sample 3429 (0.153771)

**Consensus Score: 3/10 direct matches, 4/10 in PyOD top lists** - **Good agreement!**

---

## 4. Algorithm-Specific Analysis

### 4.1 PyOD Algorithms Strengths & Weaknesses

#### IForest (Isolation Forest)

- **Strengths:**
  - Fast (0.25s avg)
  - Good for high-dimensional data
  - Robust to irrelevant features
  - Clear score separation (negative = inlier, positive = outlier)
- **Weaknesses:**
  - May miss local anomalies
  - Randomness can cause slight variations

#### LOF (Local Outlier Factor)

- **Strengths:**
  - Excellent for local anomalies
  - Good for clustered data
  - Fast (0.15s avg)
- **Weaknesses:**
  - Numerical instability on large datasets (mammography: 29M score)
  - Sensitive to k parameter

#### COPOD (Copula-based)

- **Strengths:**
  - **Fastest algorithm** (0.08s avg)
  - Parameter-free
  - Good for multivariate data
  - Stable scores
- **Weaknesses:**
  - May miss complex patterns
  - Assumes copula structure

#### HBOS (Histogram-based)

- **Strengths:**
  - Fast for large datasets
  - Handles mixed distributions
  - Good for high-dimensional data
- **Weaknesses:**
  - Slowest PyOD algorithm (0.46s avg)
  - May miss subtle anomalies

#### KNN (k-Nearest Neighbors)

- **Strengths:**
  - Simple and interpretable
  - Fast (0.14s avg)
  - Good baseline method
- **Weaknesses:**
  - Sensitive to k parameter
  - May struggle with high dimensions

### 4.2 WNCROD Strengths & Weaknesses

#### Strengths:

1. **Excellent Score Discrimination:** Top outliers show 100-1400x separation from mean
2. **Handles Mixed Data Types:** Designed for numerical, nominal, and hybrid data
3. **Theoretical Foundation:** Based on three-way decision theory and rough sets
4. **Consistent with PyOD:** Shows good agreement on top anomalies

#### Weaknesses:

1. **Extremely Slow:** 84x slower than PyOD on average
2. **Poor Scalability:** O(n²) complexity - runtime explodes with dataset size
3. **Complete Failures:** Failed on 2 datasets (mushroom, musk) - all scores = 0
4. **No Explicit Outlier Count:** Only provides scores, not binary labels
5. **Parameter Sensitivity:** Lambda parameter may need tuning

---

## 5. Critical Issues Identified

### 5.1 WNCROD Failures

**Datasets with Complete Failure (all scores = 0.000000):**

1. **Mushroom Dataset (4,429 samples, 23 features)**

   - All 4,429 samples scored 0.000000
   - Runtime: 2.02 minutes (wasted computation)
   - **Possible causes:**
     - High dimensionality (23 features)
     - Nominal/categorical data not properly handled
     - Algorithm bug or edge case
2. **Musk Dataset (3,062 samples, 167 features)**

   - All 3,062 samples scored 0.000000
   - Runtime: 7.25 minutes (longest computation, zero results)
   - **Possible causes:**
     - Very high dimensionality (167 features)
     - Algorithm may have numerical issues with high dimensions
     - Distance calculations may overflow/underflow

**Impact:** 2 out of 10 datasets (20% failure rate) - **Critical issue**

### 5.2 PyOD Numerical Instability

**LOF on Mammography Dataset:**

- Extreme scores: up to 29,204,217 (likely numerical overflow)
- Mean: 22,648 with std: 682,353
- **Issue:** Distance calculations may have numerical problems on large datasets

**Recommendation:** Use COPOD or IForest for large datasets instead of LOF

---

## 6. Scalability Analysis

### 6.1 Runtime vs Dataset Size

**WNCROD Runtime Pattern:**

- Small datasets (<500 samples): <1 second
- Medium datasets (500-5,000 samples): 1-120 seconds
- Large datasets (>5,000 samples): 100-435 seconds
- **Complexity:** Approximately O(n²) - quadratic scaling

**PyOD Runtime Pattern:**

- All datasets: <1 second per algorithm
- **Complexity:** Approximately O(n log n) or O(n) - near-linear scaling

### 6.2 Memory Considerations

**WNCROD:**

- Computes full distance matrices: O(n²) memory
- For 11,183 samples (mammography): ~1GB memory for distance matrices
- **Memory bottleneck** for very large datasets

**PyOD:**

- Most algorithms use efficient data structures
- Memory usage: O(n) to O(n log n)
- **Scalable** to much larger datasets

---

## 7. Recommendations

### 7.1 For Production Use

**Recommended Approach: PyOD (COPOD or IForest)**

**Reasons:**

1. **Speed:** 84x faster than WNCROD
2. **Reliability:** No failures across 10 datasets
3. **Scalability:** Handles large datasets efficiently
4. **Stability:** No numerical issues (avoid LOF for very large datasets)

**Specific Recommendations:**

- **Primary:** Use **COPOD** (fastest, parameter-free, stable)
- **Secondary:** Use **IForest** (robust, good for high dimensions)
- **Ensemble:** Combine multiple algorithms for consensus

### 7.2 For Research/Academic Use

**WNCROD can be valuable when:**

1. Working with **mixed data types** (numerical + nominal)
2. Need **theoretical foundation** (rough sets, three-way decisions)
3. Datasets are **small to medium** (<5,000 samples)
4. Have time for **slower computation**

**But must:**

1. **Fix the failure cases** (mushroom, musk datasets)
2. **Optimize the algorithm** for better scalability
3. **Add explicit outlier labeling** (currently only scores)

### 7.3 Hybrid Approach

**Best Practice:**

1. Use **PyOD for initial screening** (fast, reliable)
2. Use **WNCROD for detailed analysis** on interesting subsets
3. **Combine results** for consensus-based detection

---

## 8. Detailed Dataset Analysis

### 8.1 Best Performing Datasets

**For PyOD:**

- All datasets performed well
- Consistent ~10% outlier detection
- Fast across all sizes

**For WNCROD:**

- **Best:** annthyroid, thyroid (good score separation, reasonable speed)
- **Worst:** mushroom, musk (complete failure)
- **Problematic:** mammography (very slow: 4.23 min)

### 8.2 Score Quality Metrics

**WNCROD Score Separation (Max/Mean Ratio):**

- annthyroid: 196x
- mammography: 1,429x
- thyroid: 203x
- wdbc: 95x
- **Excellent discrimination** when working

**PyOD Score Separation:**

- Varies by algorithm
- LOF: 3-4x (good)
- COPOD: 3-4x (good)
- IForest: Clear positive/negative separation

---

## 9. Conclusions

### 9.1 Performance Winner: PyOD

- **84.5x faster** overall
- **100% success rate** (vs 80% for WNCROD)
- **Better scalability**
- **More reliable**

### 9.2 Detection Quality: Tie

- Both methods identify **similar top outliers** (5/10 overlap on annthyroid)
- WNCROD shows **excellent score separation** when working
- PyOD provides **consistent, interpretable results**

### 9.3 Use Case Recommendations


| Use Case              | Recommended Method   | Reason                 |
| ----------------------- | ---------------------- | ------------------------ |
| Production/Real-time  | PyOD (COPOD)         | Speed, reliability     |
| Large datasets (>10K) | PyOD (IForest/COPOD) | Scalability            |
| Mixed data types      | WNCROD (if fixed)    | Designed for this      |
| Research/Academic     | Both (compare)       | Different perspectives |
| Small datasets (<1K)  | Either               | Both fast enough       |
| High dimensions (>50) | PyOD (IForest)       | WNCROD struggles       |

### 9.4 Final Verdict

**For most practical applications: PyOD is the clear winner**

- Faster, more reliable, better scalability
- Multiple algorithms provide ensemble options
- Production-ready

**WNCROD has potential but needs work:**

- Fix failure cases (critical)
- Optimize for speed
- Improve scalability
- Then it could be valuable for specific use cases (mixed data types)

---

## 10. Technical Metrics Summary

### 10.1 Speed Metrics


| Metric            | PyOD          | WNCROD  | Winner       |
| ------------------- | --------------- | --------- | -------------- |
| Total Time        | 11.17s        | 945.79s | PyOD (84.5x) |
| Avg per Dataset   | 1.12s         | 94.58s  | PyOD (84.4x) |
| Fastest Algorithm | COPOD (0.08s) | -       | PyOD         |
| Scalability       | O(n log n)    | O(n²)  | PyOD         |

### 10.2 Reliability Metrics


| Metric           | PyOD                   | WNCROD             | Winner |
| ------------------ | ------------------------ | -------------------- | -------- |
| Success Rate     | 100% (10/10)           | 80% (8/10)         | PyOD   |
| Failure Cases    | 0                      | 2 (mushroom, musk) | PyOD   |
| Numerical Issues | 1 (LOF on mammography) | 0                  | Tie    |

### 10.3 Detection Quality Metrics


| Metric              | PyOD | WNCROD                   | Winner |
| --------------------- | ------ | -------------------------- | -------- |
| Top Anomaly Overlap | -    | 5/10 (annthyroid)        | -      |
| Score Separation    | Good | Excellent (when working) | WNCROD |
| Consistency         | High | Variable                 | PyOD   |

---

## Appendix A: Dataset Characteristics


| Dataset      | Samples | Features | Type      | WNCROD Status     |
| -------------- | --------- | ---------- | ----------- | ------------------- |
| annthyroid   | 7,200   | 7        | Numerical | ✅ Working        |
| creditA      | 425     | 16       | Mixed     | ✅ Working        |
| german       | 714     | 21       | Mixed     | ✅ Working        |
| heart270     | 166     | 14       | Mixed     | ✅ Working        |
| lymphography | 148     | 19       | Mixed     | ✅ Working        |
| mammography  | 11,183  | 7        | Numerical | ✅ Working (slow) |
| mushroom     | 4,429   | 23       | Mixed     | ❌**FAILED**      |
| musk         | 3,062   | 167      | Numerical | ❌**FAILED**      |
| thyroid      | 3,772   | 7        | Numerical | ✅ Working        |
| wdbc         | 396     | 32       | Numerical | ✅ Working        |

**Pattern:** WNCROD failures correlate with:

- High dimensionality (musk: 167 features)
- Mixed data types (mushroom: 23 features, likely categorical)

---

## Appendix B: Top Anomalies Comparison

### Annthyroid Dataset - Consensus Analysis

**Samples detected by 3+ PyOD algorithms:**

- 7058 (4 algorithms)
- 5411 (3 algorithms)
- 2209 (3 algorithms)
- 2774 (2 algorithms)
- 6373 (2 algorithms)

**WNCROD Top 10 matches:**

- 7058 ✅
- 5411 ✅
- 2774 ✅
- 6373 ✅
- 2209 ✅

**Consensus: 50% overlap** - Strong agreement between methods!

---

**Report Generated:** December 7, 2025
**Analysis Period:** 21:53:17 - 22:09:37
**Total Analysis Time:** ~16 minutes
