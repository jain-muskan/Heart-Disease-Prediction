# -*- coding: utf-8 -*-
"""Heart Disease Prediction Baseline Script"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# 1. Load and Impute Data
filename = 'framingham.csv'
df = pd.read_csv(filename)

knn = KNNImputer(n_neighbors=1)
missing_cols = df[['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']]
imputed_cols = pd.DataFrame(knn.fit_transform(missing_cols), columns=missing_cols.columns)

df = df.drop(columns=['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose'])
df = pd.concat([df, imputed_cols], axis=1)

# 2. Preprocessing & Splitting
X = df.drop(['TenYearCHD'], axis=1)
y = df['TenYearCHD']

os = RandomOverSampler(0.5)
x_os, y_os = os.fit_sample(X, y)
x_os = pd.DataFrame(x_os, columns=X.columns)
y_os = pd.Series(y_os, name='TenYearCHD')

x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.4, random_state=0)

scaler = RobustScaler()
xs_train = scaler.fit_transform(x_train)
xs_test = scaler.fit_transform(x_test)

# 3. Model Training & Evaluation
logr = LogisticRegression(class_weight='balanced', random_state=0).fit(xs_train, y_train)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, logr.predict(xs_test)):.4f}")

rfcl = RandomForestClassifier(class_weight='balanced', n_estimators=250, max_features='auto').fit(x_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rfcl.predict(x_test)):.4f}")
