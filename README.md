This repository contains the implementation of a comprehensive data-driven driving risk assessment framework developed using the NGSIM (US-101) dataset. 
Dataset: The experiments were conducted using the publicly available NGSIM (Next Generation Simulation) dataset provided by the U.S. Department of Transportation (FHWA). The dataset can be accessed from:
https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/
The framework integrates:
•	Savitzky–Golay (SG) filtering for trajectory smoothing
•	Computation of risk indicators (TIT, CPI-MADR1, CPI-MADR2)
•	Fuzzy C-Means (FCM) clustering for vehicle-level risk labeling
•	Extraction of driving behavior features
•	Feature selection and hyperparameter optimization
•	Spearman-based risk score computation
•	Machine learning-based risk prediction and regression modeling
This repository provides an integrated, data-driven driving risk assessment framework that transforms raw trajectory data into interpretable risk indicators, clustering-based risk labels, and a continuous risk score through Spearman-weighted aggregation. The framework also checks how well driving risk can be predicted using machine learning models and finds the key behaviors that most affect the risk at the vehicle level.
METHODOLOGY FRAMEWORK
The proposed system consists of three main modules:
1)  Risk Indicator Computation
•	TIT_t1, TIT_t2, TIT_t3, CPI_MADR1, CPI_MADR2
•	Min-Max normalization
•	FCM clustering
•	Risk level assignment (R0–Rk)
2) Driving Behavior Feature Extraction
•	Basic vehicle features 
•	Leading vehicle features 
•	Spaceheadway features
•	Jerk and acceleration 
•	Relative speed and spacing features
•	Microscale behavior metrics
•	High-correlation filtering (ρ > 0.85)
•	Class imbalance handling (Random UnderSampling)
3) Modeling, Risk Scoring and Evaluation
•	Feature selection (importance-based thresholding)
•	Algorithm comparison:
-XGBoost
-Random Forest
-Extra Trees
•	Hyperparameter optimization
•	Spearman-based driving risk score computation
  -Calculation of Spearman’s ρ between risk indicators and risk labels
  -Weighted aggregation of normalized indicators
  -Generation of vehicle-level driving risk scores
•	 Performance evaluation
•	Classification stage:
  -Accuracy
  -F1-score
  -Precision
  -Recall
  -AUC
•	Regression stage (risk score prediction):
     -MAE
  -MSE
     -RMSE
        - R²
The Python scripts used in this framework are listed below in sequential order:
00_savgol_filter.py
01_gap_and_space_headway_features.py
02_tit_cpi_computation.py
03_risk_indicator_normalization.py
04_fcm_clustering.py
05_jerk_computation.py
06_preceding_vehicle_extraction.py
07_basic_vehicle_characteristics.py
08_velocity_difference_and_acceleration_update.py
09_relative_comparison_preceding_vehicle.py
10_relative_change_features.py
11_microscale_behavior_features.py
12_behavior_correlation_analysis.py
13_merge_all_features.py
14_merge_cluster_labels_with_features.py
15_initial_feature_selection.py
16_high_correlation_filtering.py
17_random_undersampling.py
18_feature_selection_after_undersampling.py
19_xgboost_feature_selection_085.py
20_extratrees_feature_selection_085.py
21_randomforest_feature_selection_085.py
22_xgboost_hyperparameter_optimization.py
23_tit_cpi_normalization.py
24_spearman_risk_score_computation.py
25_risk_score_label_table.py
26_risk_score_performance_evaluation.py
Required Libraries:
pandas
numpy
scipy
scikit-learn
xgboost
scikit-fuzzy
matplotlib
imbalanced-learn

If you use this repository in your research, please cite: Aslantaş, M. and Gundogdu K.F,
Enhancing Sustainable Traffic Safety Through Machine Learning: A Risk Assessment and Feature Selection Framework Using NGSIM Data, 2026.

