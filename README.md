# Heart Disease Prediction via Multivariable Spatial-Temporal Architecture

An advanced open-source machine learning framework designed to evaluate patient cardiac risk using the **Framingham Heart Study** dataset. 

This repository adapts research methodologies originally developed for multi-variable spatial-temporal data processing—specifically integrating a weighted **Multivariable Distance Metric**, **Pre-Prediction Patient Clustering**, and **Iterative Subsampled Ensemble Sampling** to stabilize predictions on sparse, heterogeneous medical data.

---

## 📌 Project Overview & Scope

Heart disease remains a leading cause of mortality worldwide. Standard global machine learning classifiers often evaluate individual patient diagnostic records in isolation, missing nuanced risk clusters formed by overlapping demographic, temporal vital sign, and clinical survey indicators.

### Core Objectives:
* **Heterogeneous Feature Integration:** Encode high-dimensional patient indicators (demographics, vital signs, lifestyle habits, and medical history) into a unified similarity space.
* **Pre-Prediction Patient Segmentation:** Cluster similar patient risk profiles prior to executing classification algorithms to reduce false-positive noise.
* **Prediction Stability:** Apply iterative bootstrap subsampling across patient clusters to minimize variance and stabilize classification outputs on unbalanced clinical datasets.

---

## 🛠️ Architecture & Research Methodology

This project adapts the core algorithmic sequence introduced in published research on multi-variable predictive modeling (*"Predictive Crime Mapping for Smart City"*):

1. **Multivariable Distance Metric (`MultivariableDistanceMetric`):**
   Formulates a custom similarity metric that calculates weighted distance deltas across three primary clinical feature families:
   $$\text{Distance} = (W_{\text{demo}} \times \Delta_{\text{demographic}}) + (W_{\text{vital}} \times \Delta_{\text{vitals}}) + (W_{\text{survey}} \times \Delta_{\text{tokens}})$$
   * **Demographic/Spatial Proxies:** Age, gender (`male`).
   * **Temporal Vitals:** Heart rate, cigarettes per day (`cigsPerDay`).
   * **Clinical Survey Tokens:** Binary risk indicators (`diabetes`, `prevalentHyp`, `BPMeds`, `currentSmoker`).

2. **Cluster-Then-Forecast Engine (`ClusterPipeline`):**
   Groups patient records into localized risk clusters based on distance thresholds before evaluating target risk outcomes (`TenYearCHD`), transforming global classification into cohort-specific prediction.

3. **Iterative Subsampled Ensemble Model (`IterativeSubsampledPredictor`):**
   Executes iterative decision-tree fits across random subsets (100 iterations on 80% random sampling) within formed clusters to generate high-confidence consensus predictions.

