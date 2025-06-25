Parkinson's Disease Prediction

A machine learning model that predicts Parkinson’s disease using voice data. It uses SMOTE for class balancing, RFE for feature selection, and an ensemble classifier (Random Forest, SVM, and XGBoost).

 Problem Statement
Early detection of Parkinson’s disease is challenging but crucial. This project predicts the presence of Parkinson’s using voice features.

 Dataset
- Source: UCI Machine Learning Repository  
- Features extracted from voice samples of individuals with and without Parkinson’s.

 Technologies Used
- Python
- Scikit-learn
- XGBoost
- SMOTE (imblearn)
- Google Colab
- GitHub

 ML Techniques
- Data scaling with `StandardScaler`
- Outlier removal using IQR
- Class balancing with `SMOTE`
- Feature selection using `RFE`
- Ensemble model: `VotingClassifier` with Random Forest, SVM, and XGBoost

 Accuracy
- Achieved ~95% accuracy on test data, and ~94% average with 5-fold cross-validation.

 Results
- Classification Report & Confusion Matrix for evaluation
- Ensemble approach outperforms single classifiers

 How to Run
1. Upload `parkinsons.data` file to Colab
2. Run the `ipynb` notebook step-by-step
3. See accuracy, classification report, and confusion matrix
