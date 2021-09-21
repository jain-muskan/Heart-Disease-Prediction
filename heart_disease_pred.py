# -*- coding: utf-8 -*-
"""Heart Disease Prediction with CDM Distance Module"""

import pandas as pd
import numpy as np 
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading & Imputation
filename = 'framingham.csv'
df = pd.read_csv(filename)
knn = KNNImputer(n_neighbors=1)
missing_cols = df[['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']]
imputed_cols = pd.DataFrame(knn.fit_transform(missing_cols), columns=missing_cols.columns)
df = df.drop(columns=['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose'])
df = pd.concat([df, imputed_cols], axis=1)

# 2. Research Architecture: Multivariable Distance Metric (CDM)
class MultivariableDistanceMetric:
    """Computes a weighted distance between patient records."""
    def __init__(self, w_location=10.0, w_temporal=1.0, w_survey=2.0):
        self.w_location = w_location
        self.w_temporal = w_temporal
        self.w_survey = w_survey

    def calculate_distance(self, record_i, record_j):
        loc_i = np.array([record_i.get('age', 0), record_i.get('male', 0)])
        loc_j = np.array([record_j.get('age', 0), record_j.get('male', 0)])
        delta_location = np.sum(np.abs(loc_i - loc_j))

        time_i = np.array([record_i.get('cigsPerDay', 0), record_i.get('heartRate', 0)])
        time_j = np.array([record_j.get('cigsPerDay', 0), record_j.get('heartRate', 0)])
        delta_temporal = np.sum(np.abs(time_i - time_j))

        tokens_i = {k for k in ['diabetes', 'prevalentHyp', 'BPMeds', 'currentSmoker'] if record_i.get(k, 0) == 1}
        tokens_j = {k for k in ['diabetes', 'prevalentHyp', 'BPMeds', 'currentSmoker'] if record_j.get(k, 0) == 1}
        delta_survey = len(tokens_i.symmetric_difference(tokens_j))

        return (self.w_location * delta_location) + (self.w_temporal * delta_temporal) + (self.w_survey * delta_survey)

# 3. Standard ML Setup
X = df.drop(['TenYearCHD'], axis=1)
y = df['TenYearCHD']
os = RandomOverSampler(0.5)
x_os, y_os = os.fit_sample(X, y)
x_os = pd.DataFrame(x_os, columns=X.columns)
y_os = pd.Series(y_os, name='TenYearCHD')
x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.4, random_state=0)

rfcl = RandomForestClassifier(class_weight='balanced', n_estimators=250, max_features='auto').fit(x_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rfcl.predict(x_test)):.4f}")
