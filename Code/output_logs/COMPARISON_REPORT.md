# Outlier Detection Methods Comparison Report

## PyOD Algorithms Performance Analysis

**Report Generated:** 2026-05-06 11:45:44

**Datasets Analyzed:** 10

**Methods Compared:** AutoEncoder, COPOD, HBOS, IForest, KNN, LOF

---


## 1. Algorithm Performance Overview

| Algorithm | Avg Time (s) | Total Time (s) | Avg Outlier Rate | Avg ROC-AUC | Runs |
|-----------|--------------|----------------|------------------|-------------|------|
| AutoEncoder | 22.5770 | 225.7700 | 0.00% | 0.9102 | 10 |
| COPOD | 0.0930 | 0.9300 | 0.00% | 0.9459 | 10 |
| HBOS | 0.5060 | 5.0600 | 0.00% | 0.9262 | 10 |
| IForest | 0.2720 | 2.7200 | 0.00% | 0.9529 | 10 |
| KNN | 0.1080 | 1.0800 | 0.00% | 0.9147 | 10 |
| LOF | 0.1450 | 1.4500 | 0.00% | 0.6893 | 10 |


## 2. Per-Dataset Results


### Dataset: annthyroid

- **Shape:** 7200 samples × 6 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 53.9400 | N/A | N/A | 0.7366 |
| COPOD | 0.6400 | N/A | N/A | 0.7760 |
| HBOS | 4.9300 | N/A | N/A | 0.6243 |
| IForest | 0.3200 | N/A | N/A | 0.8353 |
| KNN | 0.2400 | N/A | N/A | 0.7067 |
| LOF | 0.2600 | N/A | N/A | 0.7075 |

### Dataset: creditA_plus_42_variant1

- **Shape:** 425 samples × 15 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 3.0300 | N/A | N/A | 0.9335 |
| COPOD | 0.0000 | N/A | N/A | 0.9921 |
| HBOS | 0.0000 | N/A | N/A | 0.9217 |
| IForest | 0.2300 | N/A | N/A | 0.9878 |
| KNN | 0.0100 | N/A | N/A | 0.9343 |
| LOF | 0.0100 | N/A | N/A | 0.5178 |

### Dataset: german_1_14_variant1

- **Shape:** 714 samples × 20 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 5.0900 | N/A | N/A | 0.9389 |
| COPOD | 0.0100 | N/A | N/A | 0.9730 |
| HBOS | 0.0100 | N/A | N/A | 0.9683 |
| IForest | 0.2700 | N/A | N/A | 0.9769 |
| KNN | 0.0300 | N/A | N/A | 0.9593 |
| LOF | 0.3800 | N/A | N/A | 0.9014 |

### Dataset: heart270_2_16_variant1

- **Shape:** 166 samples × 13 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 1.2100 | N/A | N/A | 0.9875 |
| COPOD | 0.0000 | N/A | N/A | 0.9929 |
| HBOS | 0.0000 | N/A | N/A | 0.9721 |
| IForest | 0.2300 | N/A | N/A | 0.9883 |
| KNN | 0.0000 | N/A | N/A | 0.9742 |
| LOF | 0.0000 | N/A | N/A | 0.7621 |

### Dataset: lymphography

- **Shape:** 148 samples × 18 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 1.0800 | N/A | N/A | 0.9765 |
| COPOD | 0.0000 | N/A | N/A | 0.9941 |
| HBOS | 0.0100 | N/A | N/A | 1.0000 |
| IForest | 0.2200 | N/A | N/A | 1.0000 |
| KNN | 0.0100 | N/A | N/A | 0.9894 |
| LOF | 0.0100 | N/A | N/A | 0.9707 |

### Dataset: mammography

- **Shape:** 11183 samples × 6 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 72.5900 | N/A | N/A | 0.8703 |
| COPOD | 0.0200 | N/A | N/A | 0.9053 |
| HBOS | 0.0100 | N/A | N/A | 0.8503 |
| IForest | 0.3700 | N/A | N/A | 0.8661 |
| KNN | 0.4600 | N/A | N/A | 0.8461 |
| LOF | 0.4900 | N/A | N/A | 0.7398 |

### Dataset: mushroom_p_221_variant1

- **Shape:** 4429 samples × 22 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 31.0700 | N/A | N/A | 0.7951 |
| COPOD | 0.0300 | N/A | N/A | 0.9449 |
| HBOS | 0.0100 | N/A | N/A | 0.9714 |
| IForest | 0.2900 | N/A | N/A | 0.9056 |
| KNN | 0.0900 | N/A | N/A | 0.8521 |
| LOF | 0.0900 | N/A | N/A | 0.6107 |

### Dataset: musk

- **Shape:** 3062 samples × 166 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 28.5500 | N/A | N/A | 0.9801 |
| COPOD | 0.2200 | N/A | N/A | 0.9463 |
| HBOS | 0.0800 | N/A | N/A | 1.0000 |
| IForest | 0.2900 | N/A | N/A | 0.9999 |
| KNN | 0.1100 | N/A | N/A | 0.9534 |
| LOF | 0.0700 | N/A | N/A | 0.4271 |

### Dataset: thyroid

- **Shape:** 3772 samples × 6 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 26.5400 | N/A | N/A | 0.9587 |
| COPOD | 0.0100 | N/A | N/A | 0.9393 |
| HBOS | 0.0000 | N/A | N/A | 0.9582 |
| IForest | 0.2800 | N/A | N/A | 0.9762 |
| KNN | 0.1100 | N/A | N/A | 0.9505 |
| LOF | 0.1300 | N/A | N/A | 0.8075 |

### Dataset: wdbc_M_39_variant1

- **Shape:** 396 samples × 31 features

| Algorithm | Time (s) | Outliers Detected | Rate | ROC-AUC |
|-----------|----------|------------------|------|---------|
| AutoEncoder | 2.6700 | N/A | N/A | 0.9252 |
| COPOD | 0.0000 | N/A | N/A | 0.9956 |
| HBOS | 0.0100 | N/A | N/A | 0.9962 |
| IForest | 0.2200 | N/A | N/A | 0.9927 |
| KNN | 0.0200 | N/A | N/A | 0.9813 |
| LOF | 0.0100 | N/A | N/A | 0.4486 |

## 3. Statistical Summary

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| AutoEncoder Time (s) | 1.0800 | 72.5900 | 22.5770 | 23.6882 |
| AutoEncoder ROC-AUC | 0.7366 | 0.9875 | 0.9102 | 0.0801 |
| COPOD Time (s) | 0.0000 | 0.6400 | 0.0930 | 0.1931 |
| COPOD ROC-AUC | 0.7760 | 0.9956 | 0.9459 | 0.0636 |
| HBOS Time (s) | 0.0000 | 4.9300 | 0.5060 | 1.4748 |
| HBOS ROC-AUC | 0.6243 | 1.0000 | 0.9262 | 0.1094 |
| IForest Time (s) | 0.2200 | 0.3700 | 0.2720 | 0.0464 |
| IForest ROC-AUC | 0.8353 | 1.0000 | 0.9529 | 0.0576 |
| KNN Time (s) | 0.0000 | 0.4600 | 0.1080 | 0.1365 |
| KNN ROC-AUC | 0.7067 | 0.9894 | 0.9147 | 0.0840 |
| LOF Time (s) | 0.0000 | 0.4900 | 0.1450 | 0.1649 |
| LOF ROC-AUC | 0.4271 | 0.9707 | 0.6893 | 0.1757 |

## 4. Key Findings

- **Fastest Algorithm:** COPOD

- **Best Average ROC-AUC:** IForest (0.9529)

- **Most Consistent:** AutoEncoder (outlier rate std: 0.0000)
