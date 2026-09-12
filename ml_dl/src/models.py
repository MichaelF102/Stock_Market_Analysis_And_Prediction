"""
Deep Learning & Baseline Model Architectures for Indian Equities.
Implements Zero Return Baseline, Ridge Baseline, SimpleRNN, LSTM, BiLSTM, and GRU models
with uniform configurations for objective empirical comparison, and advanced quantitative loss functions.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, Union
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
import tensorflow as tf
from .config import cfg, MODELS_DIR, set_seed


# ==============================================================================
# 1. Custom Quantitative Loss Functions (Beyond Standard MSE)
# ==============================================================================

@tf.keras.utils.register_keras_serializable(package="quant_dl")
class DirectionalPenaltyLoss(tf.keras.losses.Loss):
    """
    Asymmetric Directional Loss:
    Penalizes directional sign mismatch between predicted return (y_pred)
    and true return (y_true) using an asymmetric multiplier on top of robust Huber loss.
    
    L_dir = Huber(y_true, y_pred) * [1.0 + alpha * sigmoid(-gamma * y_true * y_pred)]
    """
    def __init__(self, alpha: float = 2.0, gamma: float = 50.0, delta: float = 0.05, name: str = "directional_penalty_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.delta = float(delta)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Base robust Huber loss to handle fat tails
        error = y_true - y_pred
        abs_error = tf.abs(error)
        huber = tf.where(
            abs_error <= self.delta,
            0.5 * tf.square(error),
            self.delta * (abs_error - 0.5 * self.delta)
        )
        
        # Directional sign penalty: when sign(y_true) != sign(y_pred), y_true * y_pred < 0
        sign_prod = y_true * y_pred
        directional_multiplier = 1.0 + self.alpha * tf.nn.sigmoid(-self.gamma * sign_prod)
        
        return tf.reduce_mean(huber * directional_multiplier)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "delta": self.delta
        })
        return config


@tf.keras.utils.register_keras_serializable(package="quant_dl")
class SharpeRatioLoss(tf.keras.losses.Loss):
    """
    Differentiable Negative Sharpe Ratio Loss:
    Treats continuous model return predictions as portfolio position weights (tanh(y_pred / tau)),
    evaluates realized portfolio returns on the batch, and minimizes negative Sharpe Ratio.
    
    L_sharpe = - (E[R_p]) / (sqrt(Var(R_p) + eps))
    """
    def __init__(self, temperature: float = 0.05, eps: float = 1e-6, name: str = "sharpe_ratio_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # Soft position sizing between -1.0 and +1.0
        positions = tf.tanh(y_pred / self.temperature)
        portfolio_returns = positions * y_true
        
        mean_ret = tf.reduce_mean(portfolio_returns)
        var_ret = tf.reduce_mean(tf.square(portfolio_returns - mean_ret))
        sharpe = mean_ret / (tf.sqrt(var_ret + self.eps))
        
        return -sharpe

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "eps": self.eps
        })
        return config


@tf.keras.utils.register_keras_serializable(package="quant_dl")
class CompositeQuantLoss(tf.keras.losses.Loss):
    """
    Composite Quantitative Loss Function:
    Multi-objective loss combining:
    1. Robust Huber Loss (Fat-tail regression accuracy)
    2. Directional Sign Penalty (Asymmetric directional correctness)
    3. Negative Sharpe Ratio (Risk-adjusted portfolio return)
    
    L_total = L_Huber + lambda_dir * L_Directional + lambda_sharpe * L_Sharpe
    """
    def __init__(
        self,
        lambda_dir: float = 1.0,
        lambda_sharpe: float = 0.1,
        delta: float = 0.05,
        alpha: float = 2.0,
        gamma: float = 50.0,
        temperature: float = 0.05,
        name: str = "composite_quant_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.lambda_dir = float(lambda_dir)
        self.lambda_sharpe = float(lambda_sharpe)
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        
        self.dir_loss = DirectionalPenaltyLoss(alpha=alpha, gamma=gamma, delta=delta)
        self.sharpe_loss = SharpeRatioLoss(temperature=temperature)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 1. Base Huber
        error = y_true - y_pred
        abs_error = tf.abs(error)
        huber_loss = tf.reduce_mean(tf.where(
            abs_error <= self.delta,
            0.5 * tf.square(error),
            self.delta * (abs_error - 0.5 * self.delta)
        ))
        
        # 2. Directional Penalty Loss
        dir_loss = self.dir_loss(y_true, y_pred)
        
        # 3. Sharpe Loss
        sharpe_loss = self.sharpe_loss(y_true, y_pred)
        
        return huber_loss + self.lambda_dir * dir_loss + self.lambda_sharpe * sharpe_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_dir": self.lambda_dir,
            "lambda_sharpe": self.lambda_sharpe,
            "delta": self.delta,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "temperature": self.temperature
        })
        return config


def get_loss_function(loss_name: Union[str, tf.keras.losses.Loss] = "mse") -> Any:
    """Factory to return standard or quantitative loss functions."""
    if isinstance(loss_name, tf.keras.losses.Loss):
        return loss_name
    
    name = str(loss_name).lower()
    if name in ["mse", "mean_squared_error"]:
        return "mse"
    elif name in ["mae", "mean_absolute_error"]:
        return "mae"
    elif name in ["huber", "huber_loss"]:
        return tf.keras.losses.Huber(delta=0.05)
    elif name in ["directional", "dir", "directional_loss"]:
        return DirectionalPenaltyLoss()
    elif name in ["sharpe", "sharpe_loss"]:
        return SharpeRatioLoss()
    elif name in ["composite", "quant", "composite_quant_loss"]:
        return CompositeQuantLoss()
    else:
        return "mse"


# ==============================================================================
# 2. Baseline Model Architectures
# ==============================================================================

class ZeroReturnBaseline:
    """Predicts a constant future return of zero."""
    def fit(self, X, y):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float32)


class RidgeBaseline:
    """Ridge regression baseline fitted on the latest time-step of features."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        # Use features from the most recent lookback day (t)
        if len(X.shape) == 3:
            return X[:, -1, :]
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = self._extract_features(X)
        self.model.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._extract_features(X)
        return self.model.predict(X_flat)


# ==============================================================================
# 3. Deep Learning Architectures
# ==============================================================================

def build_simple_rnn_model(input_shape: Tuple[int, int], loss: Union[str, Any] = "mse") -> Any:
    """Builds a SimpleRNN architecture."""
    set_seed(cfg.RANDOM_SEED)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.SimpleRNN(cfg.RECURRENT_UNITS, activation="tanh", name="simple_rnn_layer"),
        tf.keras.layers.Dropout(cfg.DROPOUT_RATE, name="dropout_layer"),
        tf.keras.layers.Dense(cfg.DENSE_UNITS, activation="relu", name="dense_intermediate"),
        tf.keras.layers.Dense(1, activation="linear", name="output_layer")
    ], name="SimpleRNN_Model")

    loss_obj = get_loss_function(loss)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss_obj,
        metrics=["mae"]
    )
    return model


def build_lstm_model(input_shape: Tuple[int, int], loss: Union[str, Any] = "mse") -> Any:
    """Builds an LSTM architecture with comparable capacity."""
    set_seed(cfg.RANDOM_SEED)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(cfg.RECURRENT_UNITS, activation="tanh", recurrent_activation="sigmoid", name="lstm_layer"),
        tf.keras.layers.Dropout(cfg.DROPOUT_RATE, name="dropout_layer"),
        tf.keras.layers.Dense(cfg.DENSE_UNITS, activation="relu", name="dense_intermediate"),
        tf.keras.layers.Dense(1, activation="linear", name="output_layer")
    ], name="LSTM_Model")

    loss_obj = get_loss_function(loss)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss_obj,
        metrics=["mae"]
    )
    return model


def build_gru_model(input_shape: Tuple[int, int], loss: Union[str, Any] = "mse") -> Any:
    """Builds a GRU architecture with comparable capacity."""
    set_seed(cfg.RANDOM_SEED)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.GRU(cfg.RECURRENT_UNITS, activation="tanh", recurrent_activation="sigmoid", name="gru_layer"),
        tf.keras.layers.Dropout(cfg.DROPOUT_RATE, name="dropout_layer"),
        tf.keras.layers.Dense(cfg.DENSE_UNITS, activation="relu", name="dense_intermediate"),
        tf.keras.layers.Dense(1, activation="linear", name="output_layer")
    ], name="GRU_Model")

    loss_obj = get_loss_function(loss)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss_obj,
        metrics=["mae"]
    )
    return model


def build_bilstm_model(input_shape: Tuple[int, int], loss: Union[str, Any] = "mse") -> Any:
    """Builds a Bidirectional LSTM (BiLSTM) architecture with comparable capacity."""
    set_seed(cfg.RANDOM_SEED)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(cfg.RECURRENT_UNITS // 2, activation="tanh", recurrent_activation="sigmoid"),
            name="bilstm_layer"
        ),
        tf.keras.layers.Dropout(cfg.DROPOUT_RATE, name="dropout_layer"),
        tf.keras.layers.Dense(cfg.DENSE_UNITS, activation="relu", name="dense_intermediate"),
        tf.keras.layers.Dense(1, activation="linear", name="output_layer")
    ], name="BiLSTM_Model")

    loss_obj = get_loss_function(loss)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=loss_obj,
        metrics=["mae"]
    )
    return model


def train_dl_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    epochs: int = cfg.EPOCHS,
    batch_size: int = cfg.BATCH_SIZE
) -> Tuple[Any, Dict, Dict]:
    """
    Trains a deep learning model with early stopping and learning rate scheduling.
    Returns (trained_model, history_dict, metadata_dict).
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.REDUCE_LR_FACTOR,
            patience=cfg.PATIENCE_REDUCE_LR,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print(f"\n==========================================")
    print(f" Training Model: {model_name} ")
    print(f" Input Shape: {X_train.shape} | Val Shape: {X_val.shape}")
    print(f"==========================================")

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time

    # Save trained model artifact
    save_path = MODELS_DIR / f"{model_name.lower()}.keras"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    meta = {
        "model_name": model_name,
        "total_params": model.count_params(),
        "train_time_sec": training_time,
        "stopped_epoch": len(history.history["loss"]),
        "best_val_loss": min(history.history["val_loss"]),
        "best_val_mae": min(history.history.get("val_mae", [0.0]))
    }

    return model, history.history, meta
