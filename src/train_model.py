import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from preprocessing import load_and_clean_data, prepare_data_splits
from model import get_model_registry

def train_and_evaluate_all(data_path: str, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data
    df = load_and_clean_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data_splits(df)
    
    models, param_grids = get_model_registry()
    results = []
    best_overall_score = 0.0
    champion_pipeline = None
    champion_name = ""
    
    for name, model in models.items():
        print(f"\n--- Tuning and Training: {name} ---")
        
        # Build pipeline to completely avoid data leakage
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_pipe = grid_search.best_estimator_
        
        # Test Set Evaluation
        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe['model'], "predict_proba") else y_pred
        
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1_Score": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_proba),
            "CV_Best_F1": grid_search.best_score_
        }
        results.append(metrics)
        print(f"Test F1-Score: {metrics['F1_Score']:.4f} | ROC-AUC: {metrics['ROC_AUC']:.4f}")
        
        # Track the top performer based on Recall and F1-Score balance
        if metrics['F1_Score'] > best_overall_score:
            best_overall_score = metrics['F1_Score']
            champion_pipeline = best_pipe
            champion_name = name

    # Compile and display dataframe of metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Save Champion Model Pipeline
    champion_path = os.path.join(output_dir, "champion_breast_cancer_model.joblib")
    joblib.dump(champion_pipeline, champion_path)
    print(f"\n>>>> Champion Selected: {champion_name} saved successfully at {champion_path}!")
    
    # Generate Visual Artifacts for the project report
    generate_performance_plots(results_df, champion_pipeline, X_test, y_test, output_dir)
    return results_df

def generate_performance_plots(results_df, best_pipe, X_test, y_test, output_dir):
    sns.set_theme(style="whitegrid")
    
    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    melted_df = pd.melt(results_df, id_vars="Model", value_vars=["Accuracy", "F1_Score", "ROC_AUC"])
    sns.barplot(data=melted_df, x="Model", y="value", hue="variable", palette="viridis")
    plt.xticks(rotation=30, ha='right')
    plt.title("Comparative Performance Analysis of Evaluated Models")
    plt.ylabel("Score")
    plt.ylim(0.8, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_chart.png"), dpi=300)
    plt.close()
    
    # 2. Confusion Matrix & ROC Curve of Champion
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe['model'], "predict_proba") else y_pred
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False,
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    ax[0].set_title("Champion Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.4f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_title("Receiver Operating Characteristic (ROC)")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "champion_evaluation_metrics.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Assumes dataset is downloaded to local path
    train_and_evaluate_all(data_path="data/data.csv")