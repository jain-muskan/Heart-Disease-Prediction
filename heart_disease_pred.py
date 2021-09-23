# -*- coding: utf-8 -*-
"""Heart Disease Prediction with Clustering & Iterative Sampling Modules"""

import pandas as pd
import numpy as np 
import random
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Data Loading & Imputation
filename = 'framingham.csv'
df = pd.read_csv(filename)
knn = KNNImputer(n_neighbors=1)
missing_cols = df[['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose']]
imputed_cols = pd.DataFrame(knn.fit_transform(missing_cols), columns=missing_cols.columns)
df = df.drop(columns=['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose'])
df = pd.concat([df, imputed_cols], axis=1)

# Multivariable Distance Metric
class MultivariableDistanceMetric:
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

# Cluster Pipeline
class ClusterPipeline:
    def __init__(self, threshold=15.0):
        self.threshold = threshold
        self.metric = MultivariableDistanceMetric()

    def fit_clusters(self, df_records):
        records_list = df_records.to_dict('records')
        clusters = []
        visited = set()
        for i, record_i in enumerate(records_list):
            if i in visited: continue
            current_cluster = [record_i]
            visited.add(i)
            for j, record_j in enumerate(records_list):
                if j not in visited:
                    if self.metric.calculate_distance(record_i, record_j) <= self.threshold:
                        current_cluster.append(record_j)
                        visited.add(j)
            clusters.append(current_cluster)
        return clusters

# Iterative Subsampled Predictor
class IterativeSubsampledPredictor:
    def __init__(self, iterations=100, sample_ratio=0.8):
        self.iterations = iterations
        self.sample_ratio = sample_ratio

    def predict_cluster(self, cluster_data, query_features, feature_cols):
        if len(cluster_data) < 3: return 0, 0.0
        X_cluster = np.array([[item[col] for col in feature_cols] for item in cluster_data])
        y_cluster = np.array([item['TenYearCHD'] for item in cluster_data])
        n_samples = max(2, int(len(cluster_data) * self.sample_ratio))
        prediction_counts = {}
        for _ in range(self.iterations):
            indices = random.sample(range(len(cluster_data)), n_samples)
            clf = DecisionTreeClassifier(random_state=None).fit(X_cluster[indices], y_cluster[indices])
            pred = clf.predict([query_features])[0]
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        best_pred = max(prediction_counts, key=prediction_counts.get)
        return best_pred, prediction_counts[best_pred] / self.iterations

pipeline = ClusterPipeline(threshold=15.0)
print(f"Clusters formed: {len(pipeline.fit_clusters(df.head(100)))}")
