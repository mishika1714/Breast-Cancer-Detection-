from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def get_model_registry():
    """
    Returns a dictionary of uninitialized models and their corresponding 
    hyperparameter grids for optimization.
    """
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "Random_Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    param_grids = {
        "Logistic_Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"]
        },
        "Decision_Tree": {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10]
        },
        "Random_Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [4, 6, 8, None],
            "model__criterion": ["gini", "entropy"]
        },
        "SVM": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"]
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"]
        },
        "Naive_Bayes": {},  # No major hyperparameters to tune for standard GaussianNB
        "Gradient_Boosting": {
            "model__n_estimators": [50, 100, 150],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 4, 5]
        },
        "XGBoost": {
            "model__n_estimators": [50, 100, 150],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 4, 5]
        }
    }
    
    return models, param_grids