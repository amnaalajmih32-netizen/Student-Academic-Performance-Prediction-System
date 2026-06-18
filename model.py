import pandas as pd
import joblib
import os
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

# Configuration
BASE_DIR = r"C:\Users\WIN11\Desktop\FinalProject"
DATA_PATH = os.path.join(BASE_DIR, "student-mat-fixed.csv")

def train_academic_prototype():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError("❌ Data file not found in the specified path.")
        
        df = pd.read_csv(DATA_PATH, sep=';')
        
        def simplify_target(score):
            if score >= 16: return "High Performance"
            if score >= 10: return "Medium Performance"
            return "At Academic Risk"
        
        y = df['G3'].apply(simplify_target)
        X = df.drop(columns=['G3'])

        # Preprocessing Pipeline
        preprocessor = ColumnTransformer([
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), X.select_dtypes(include=['object']).columns),
            ('num', SimpleImputer(strategy='mean'), X.select_dtypes(exclude=['object']).columns)
        ])

        # Fixed random_state ensures reproducibility of experimental results
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 1. SHAP Interpretability Model (Trained on Train, Explained on Test)
        interpret_pipe = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
        interpret_pipe.fit(X_train, y_train)
        
        fitted_preprocessor = interpret_pipe.named_steps['pre']
        fitted_rf = interpret_pipe.named_steps['clf']
        
        # Explain on Test set for better scientific generalization
        X_test_transformed = fitted_preprocessor.transform(X_test)
        explainer = shap.TreeExplainer(fitted_rf)
        shap_values = explainer.shap_values(X_test_transformed)
        
        risk_idx = list(interpret_pipe.classes_).index("At Academic Risk")
        
        if isinstance(shap_values, list):
            shap_array = shap_values[risk_idx]
        else:
            shap_array = shap_values[:, :, risk_idx] if len(shap_values.shape) == 3 else shap_values

        global_shap = np.abs(shap_array).mean(axis=0)

        # 2. Final Calibrated Model
        calibrated_clf = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42),
            cv=3
        )
        final_pipe = Pipeline([('pre', fitted_preprocessor), ('clf', calibrated_clf)])
        final_pipe.fit(X_train, y_train)
        
        report_dict = classification_report(y_test, final_pipe.predict(X_test), output_dict=True)

        metrics = {
            "Classification_Report": report_dict,
            "Classes": list(final_pipe.classes_),
            "Features": list(X.columns),
            "Proc_Feature_Names": fitted_preprocessor.get_feature_names_out().tolist(),
            "SHAP_Global": global_shap.tolist(),
            "Risk_Class_Index": risk_idx
        }
        
        joblib.dump(final_pipe, os.path.join(BASE_DIR, "full_pipeline.pkl"))
        joblib.dump(metrics, os.path.join(BASE_DIR, "model_metrics.pkl"))
        print("✅ SUCCESS: Research-Grade Architecture Ready.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    train_academic_prototype()