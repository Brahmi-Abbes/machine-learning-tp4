# Machine Learning — Practical Session 4 (TP4)
## Overfitting, Data Leakage, and Model Generalization

**Academic Year:** 2026–2027  
**Degree:** First-year Master's Degree in Machine Learning  
**Institution:** National Higher School of Advanced Technologies  

---

### Project Overview
This repository contains the complete implementation, experimental results, and report for Practical Session 4 (TP4), focusing on detecting and preventing data leakage, diagnosing overfitting, applying regularization, and evaluating model generalization under chronological validation.

### Repository Structure
```
machine-learning-tp4/
├── Work_session_4_Overfitting_data_leakage_and_model_generalization.pdf
├── README.md
└── TP4_Machine_Learning/
    ├── tp4.py                              # Main executable script
    ├── hybrid_student_performance_1200.csv # Official student survey dataset
    ├── requirements.txt                    # Project dependencies
    ├── README.md                           # Detailed TP4 documentation and results
    └── results/                            # Generated evaluation tables and figures
        ├── final_discussion.txt
        ├── final_model_comparison.csv
        ├── part_b_summary.csv
        ├── part_c1_complexity.csv
        ├── part_c2_feature_complexity.csv
        ├── part_d1_cohort_encoding.csv
        ├── part_d2_feature_selection.csv
        ├── part_d3_regularization_coefficients.csv
        └── plots/
            ├── part_a_rolling_cgpa.png
            ├── part_c1_tree_depth_overfitting.png
            ├── part_c2_polynomial_degree_cv.png
            ├── part_d3_regularization.png
            └── part_d4_final_models_comparison.png
```

### Quick Start
To reproduce the experiments and generate all tables and figures:
```bash
cd TP4_Machine_Learning
pip install -r requirements.txt
python tp4.py
```
For detailed methodology, empirical analysis, and discussion answers, refer to [TP4_Machine_Learning/README.md](file:///c:/Users/BAB%20AL%20SAFA/Desktop/TPs/ML/TP4/machine-learning-tp4/TP4_Machine_Learning/README.md).
