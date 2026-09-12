"""
Machine Learning Model Architectures and Quantitative Training Utilities
Supports: Decision Tree, Random Forest, LightGBM, XGBoost, and CatBoost.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def get_ml_model_definitions() -> Dict[str, Any]:
    """Returns initialized classical and gradient boosted tree regression models with financial time series tuned hyperparameters."""
    return {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=25,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=40,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=150,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            thread_count=-1
        )
    }


def extract_feature_importances(model: Any, feature_names: list) -> pd.DataFrame:
    """Extracts feature importance weights across linear and tree ensembles."""
    importances = None
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
        
    if importances is not None and len(importances) == len(feature_names):
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        # Normalize to 0-100 scale
        tot = df_imp["Importance"].sum()
        if tot > 0:
            df_imp["Normalized_Importance"] = (df_imp["Importance"] / tot) * 100.0
        else:
            df_imp["Normalized_Importance"] = df_imp["Importance"]
        return df_imp
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Normalized_Importance"])
