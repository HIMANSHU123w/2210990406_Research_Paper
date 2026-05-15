# A Lightweight Hybrid Machine Learning Framework for Phishing URL Detection and Classification

> **Industry Oriented Hands-on Experience (22CS422) — Final Research Project Submission**

---

## 📋 Project & Submission Meta Information

| Attribute | Details |
| :--- | :--- |
| **Project Title** | A Lightweight Hybrid Machine Learning Framework for Phishing URL Detection and Classification |
| **Type** | Research |
| **Author / Student** | Himanshu Mehta (Roll No: **2210990406**) |
| **Email** | himanshu406.be22@chitkara.edu.in |
| **Team Details** | Single Author Research Project |
| **Mentor / Supervisor** | Dr. Simar Virk (Assistant Professor) |
| **University** | Department of Computer Science and Engineering, Chitkara University, Punjab |
| **Current Status** | **Submitted to Conference** |

---

## 🎯 Executive Summary

Phishing attacks represent one of the most critical cybersecurity vectors globally. Traditional detection systems rely heavily on static blacklisting (which is reactive to zero-day links) or heavy Deep Learning architectures (which induce unacceptable inference latency at the client side). 

This research introduces a highly optimized, **Lightweight Hybrid Machine Learning Framework** that achieves state-of--the-art accuracy while maintaining extreme computational efficiency. By fusing **Random Forest (RF)** and **Extreme Gradient Boosting (XGBoost)** as base learners within a **Stacking Classifier** architecture orchestrated by a **Logistic Regression** meta-learner, the model dynamically filters non-linear semantic and lexical anomalies directly from raw URLs without executing secondary network requests.

### 🌟 Key Performance Highlights
- **Precision:** `98.9%`
- **F1-Score:** `98.7%`
- **Inference Latency:** Optimized for sub-millisecond local boundary evaluation.
- **Robustness:** Formally cross-validated against overlapping synthetic noise distributions via SMOTE balancing.

---

## 📂 Directory Structure & Deliverables Guide

This repository bundle compiles the entire academic, technical, and analytical lifecycle of the project:

```text
.
├── AI_Based_Phishing_Detection_Paper.pdf        # Full IEEE/IOHE Compliant Research Paper
├── AI_Based_Phishing_Detection_Project_Report.pdf # Formatted University Project Report
├── AI_Based_Phishing_Detection_Abstract.pdf     # Concise 300-word Pre-Research Abstract
├── Final_IOHE_External_PPT.pdf                  # Official 12-Slide Defense Presentation
├── himanshu_research.ipynb                      # Primary Executable Jupyter Notebook
├── PhishingDetection_Cell_Wise_Code.md          # Granular Layman Code Explanations & Cell Docs
└── PhishingDetection/                           # Empirical Results Subfolder
    ├── metrics/                                 # Cross-Validation JSONs, Model Latency & Logs
    ├── plots/                                   # 19 High-Resolution Analytical Visualizations
    └── results/                                 # Aggregated CSV Comparisons & Summary Reports
    └── datasets/                                # Datasets
    └── checkpoints/                             # Model Checkpoints
    └── models/                                  # Trained Models
```

---

## 🛠 Methodology & Pipeline Overview

1. **Feature Engineering:** Extracts pure lexical metrics (URL length, special character frequencies, entropy) paired with structural flags (subdomain lengths, secure connection markers) directly into a consolidated **18-feature pipeline**.
2. **Data Integration:** Utilizes two core standard benchmarks for internal mapping:
   - **PhiUSIIL Phishing URL Dataset** (134,850 consolidated records)
   - **Web Page Phishing Detection Dataset** (73,575 auxiliary testing instances)
3. **Class Balancing:** Addresses synthetic class skews programmatically via **SMOTE** (Synthetic Minority Over-sampling Technique).
4. **Stacking Ensemble:** Base classifiers capture diverse structural splits, while the secondary meta-learner corrects residual variance, achieving excellent generalizability on unseen data vectors.
5. **Model Interpretability (White-Box AI):** Integrates **SHAP** (SHapley Additive exPlanations) to provide mathematical transparency to security operations teams regarding exactly which URL token triggered an alert.

---

## 🚀 Future Scope
- **Edge Deployment:** Directly convertible into a low-footprint client-side browser extension.
- **Threat Intelligence API:** Integration ready for downstream automated enterprise network boundary firewalls.