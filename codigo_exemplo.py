from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest

from imblearn.over_sampling import SMOTETomek, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

train_df = pd.read_csv('train.csv', index_col=[0])
test_df = pd.read_csv('test.csv', index_col=[0])

X = train_df.drop(columns=['class'])
y = train_df['class']

iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(X)
mask = outliers != -1
X, y = X[mask], y[mask]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns), index=X.index)

test_poly = poly.transform(test_df)
test_poly_df = pd.DataFrame(test_poly, columns=poly.get_feature_names_out(X.columns), index=test_df.index)

X_poly_clipped = np.clip(X_poly_df.values, a_min=0, a_max=None)
test_poly_clipped = np.clip(test_poly_df.values, a_min=0, a_max=None)

X = pd.DataFrame(np.log1p(X_poly_clipped), columns=X_poly_df.columns, index=X_poly_df.index)
test_df = pd.DataFrame(np.log1p(test_poly_clipped), columns=test_poly_df.columns, index=test_poly_df.index)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

dt_pipeline = ImbPipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("smote", SMOTETomek(random_state=42, sampling_strategy='minority')),
    ("adasyn", ADASYN(random_state=42, sampling_strategy='minority')),
    ("dt", DecisionTreeClassifier(
        max_depth=9,
        min_samples_split=3,
        min_samples_leaf=4,
        criterion='gini',
        max_features=None,
        random_state=42
    ))
])

dt_pipeline.fit(X_train, y_train)
preds = dt_pipeline.predict(X_val)

acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds, average="macro")
print(f"Decision Tree - Accuracy: {acc:.4f} | F1: {f1:.4f}")
print(classification_report(y_val, preds))

dt_pipeline.fit(X, y)
final_predictions = dt_pipeline.predict(test_df)

def create_submission_file(predictions, test_df, submission_file_name="submission.csv"):
    submission_df = pd.DataFrame({'id': test_df.index, 'class': predictions})
    submission_df.to_csv(submission_file_name, index=False)
    print(f"\nSubmission file '{submission_file_name}' created successfully.")

create_submission_file(final_predictions, test_df)
