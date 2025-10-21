# 🏦 Bank Customer Churn Prediction Service

This repository hosts a robust **Machine Learning Microservice** dedicated to forecasting customer attrition (churn) for the companion MERN-stack banking application. The goal is to provide a reliable predictive layer that identifies customers likely to close their accounts (`Exited = 1`), enabling proactive business interventions.

## ⚙️ Model and Pipeline Architecture

The core of this project is a highly structured, two-phase ML workflow designed for high performance and reliable deployment.

### Phase 1: Training and Artifact Generation

The training pipeline maximizes predictive power on complex, imbalanced customer data.

| Component | Description | Rationale |
| :--- | :--- | :--- |
| **Algorithm** | **Random Forest Classifier** (`n_estimators=100`) | Chosen for its stability, high performance, and robustness against non-linear feature relationships. |
| **Imbalance Handling** | `class_weight='balanced'` | Crucial for this churn problem; it automatically adjusts weights to penalize misclassification of the rare minority class (`Exited=1`), leading to better **Recall**. |
| **Feature Engineering** | Extensive creation of custom features, including: `BalanceToSalaryRatio`, `ProductUsage` (interaction), and **Age/Tenure Group** dummies. | Introduces business logic and non-linear data patterns, significantly boosting the model's ability to extract signal. |
| **Feature Scaling** | **StandardScaler** | Applied only to continuous features (`CreditScore`, `Balance`, etc.) *after* splitting, ensuring feature consistency and preventing large-range features from dominating. |
| **Artifacts Saved** | `simple_rf_model.pkl`, `fitted_standard_scaler.pkl`, `fitted_label_encoder.pkl` | Essential for deployment, guaranteeing the **exact same transformations and model state** are used for future predictions. |

### Phase 2: Inference Pipeline (Deployment Ready)

The deployment structure is built for production, ensuring **zero data leakage** and **full consistency** with the training environment.

The inference process is executed sequentially via dedicated functions (e.g., `load_artifacts`, `replicate_feature_engineering`, `preprocess_and_align`, `predict_churn`).

| Inference Step | Key Action | Importance |
| :--- | :--- | :--- |
| **Artifact Loading** | Loads the three saved `.pkl` files (Model, Scaler, Encoder). | Ensures the deployment uses the established $\mu$, $\sigma$, and feature mappings. |
| **Feature Replication** | Re-calculates all custom engineered features (e.g., ratios, groups) on the new raw data. | Guarantees the raw input data is enriched identically to the training data. |
| **Preprocessing & Alignment** | Uses the **loaded scaler** to transform numerical data. Performs One-Hot Encoding. **Crucially, it aligns the final feature set:** absent One-Hot Encoded columns (categories not present in the new batch) are added with a value of **0**. | Maintains the mandatory feature count, order, and structure expected by the trained model. |
| **Prediction** | Uses `model.predict_proba()` to output the **churn probability** (and classification based on a set threshold). | Provides the actionable insight (risk score) for the business application. |

## 🔌 Integration with MERN Stack

This repository serves as the predictive backend logic. The output (churn probability and classification) is designed to be easily consumed by the main MERN application through a simple API endpoint (e.g., using Flask/FastAPI), allowing for real-time risk assessment and targeted user interventions.
