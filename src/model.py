import pandas as pd


import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from datetime import datetime, timedelta
from src.plots import plot_actual_vs_predicted_scatter, plot_actual_vs_predicted_timeseries, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_actual_vs_predicted_class_over_time



def train_test_split(X, y, test_size=0.2):
    # Split the data into training and testing sets
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    X_train.index = pd.to_datetime(X_train.index, errors='coerce')
    X_test.index = pd.to_datetime(X_test.index, errors='coerce')    

    y_train.index = pd.to_datetime(y_train.index, errors='coerce')
    y_test.index = pd.to_datetime(y_test.index, errors='coerce')
    return X_train, X_test, y_train, y_test


def fit_arima_model(X, y, order=(1, 0, 0)):
    model = ARIMA(y, exog=X, order=order)
    model_fit = model.fit()
    return model_fit

def make_predictions(model_fit, X):
    predictions = model_fit.predict(start=len(X) - len(X), end=len(X) - 1, exogenous=X)
    return predictions

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2_score_value = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)   
    return mse, r2_score_value, mae

def train_models(models, X_train, y_train):
    """
    Trains multiple time series models on the provided training data.
    Args:
        models (list): List of dictionaries containing model names, calls, and parameters.
        X_train (pd.DataFrame or np.ndarray): Training features (exogenous variables).
        y_train (pd.Series or np.ndarray): Training target series.
    Returns:
        dict: Dictionary containing trained models.
    """

    trained_models = {}
    if not isinstance(models, list):
        raise ValueError("The 'models' argument must be a list of dictionaries containing model information.")
    if len(y_train) == 0:
        raise ValueError("The training data is empty. Please provide valid data for training.")

    for model in models:
        model_name = model["name"]
        model_call = model["call"]
        model_params = model["params"]
        print(f"Preparing to train {model_name} with parameters: {model_params}")
        # Ensure the models directory exists
        if not os.path.exists("./models"):
            os.makedirs("./models")
        if model_name in ["SARIMAX", "ARIMA"]:
            # Pass exogenous variables if available
            if X_train is not None and X_train.shape[1] > 0:
                trained_model = model_call(y_train, exog=X_train, **model_params).fit()
            else:
                trained_model = model_call(y_train, **model_params).fit()
            # joblib.dump(trained_model, f"./models/{model_name}_model.pkl")
        else:
            trained_model = model_call(**model_params).fit(X_train, y_train)
            # joblib.dump(trained_model, f"./models/{model_name}_model.pkl")
        
        trained_models[model_name] = trained_model
        print(f"{model_name} trained successfully.\n")
    print("All models trained successfully.")

    return trained_models

def train_ml_models(models, X_train, y_train):
    """
    Trains multiple models on the provided training data.
    Args:
        models (list): List of dictionaries containing model names, calls, and parameters.
        X_train (pd.DataFrame or np.ndarray): Training features (exogenous variables).
        y_train (pd.Series or np.ndarray): Training target series.
    Returns:
        dict: Dictionary containing trained non-time series models.
    """
    
    trained_models = {}
    if not isinstance(models, list):
        raise ValueError("The 'models' argument must be a list of dictionaries containing model information.")
    if len(y_train) == 0:
        raise ValueError("The training data is empty. Please provide valid data for training.")

    for model in models:
        model_name = model["name"]
        model_call = model["call"]
        model_params = model["params"]
        print(f"Preparing to train {model_name} with parameters: {model_params}")
        
        if not os.path.exists("./models"):
            os.makedirs("./models")
        
        trained_model = model_call(**model_params).fit(X_train, y_train)
        #joblib.dump(trained_model, f"./models/{model_name}_model.pkl")
        
        trained_models[model_name] = trained_model
        print(f"{model_name} trained successfully.\n")
    
    print("All models trained successfully.")
    
    return trained_models

def train_time_series_models(models, X_train, y_train):
    """
    Trains multiple time series models on the provided training data.
    """
    trained_time_series_models = {}
    
    for model in models:
        model_name = model["name"]
        model_call = model["call"]
        model_params = model["params"]
        print(f"Preparing to train {model_name} with parameters: {model_params}")
        
        try:
            if model_name in ["SARIMAX", "ARIMA", "SARIMAX_Daily", "ARIMA_Simple", 
                            "ARIMA_Complex", "SARIMA_Weekly", "SARIMA_Monthly"]:
                # For SARIMAX/ARIMA models, pass endog as first argument
                model_instance = model_call(endog=y_train, exog=X_train, **model_params)
                trained_model = model_instance.fit(disp=False)
            
            elif model_name == "VAR":
                # For VAR, combine X and y into one dataframe
                combined_data = pd.concat([y_train, X_train], axis=1)
                model_instance = model_call(combined_data)
                trained_model = model_instance.fit(**model_params)
            
            elif model_name == "Prophet":
                # For Prophet, prepare data in required format
                prophet_data = pd.DataFrame({
                    'ds': y_train.index,
                    'y': y_train.values
                })
                model_instance = model_call(**model_params)
                for column in X_train.columns:
                    model_instance.add_regressor(column)
                trained_model = model_instance.fit(prophet_data)
            
            else:
                trained_model = model_call(**model_params).fit(X_train, y_train)
            
            trained_time_series_models[model_name] = trained_model
            print(f"{model_name} trained successfully.\n")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}\n")
            continue
    
    print("Time series models training complete.")
    return trained_time_series_models


def evaluate_models(trained_models,X_test, y_test,model_type = "classification",y_train = None):
    """
    Evaluates trained models on the test data and calculates Mean Squared Error (MSE).
    Args:
        trained_models (dict): Dictionary containing trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    Returns:
        pd.DataFrame: DataFrame containing model names, parameters, and MSE values.
    """
    results = []
    plot_paths = {}
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./figures/classification_performance", exist_ok=True)

    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...")
        if model_type in "time_series":
            start = len(model.data.endog)
            end = start + len(y_test) - 1
            predictions = model.predict(start=start, end=end, exog=X_test)
            predictions = predictions[:len(y_test)]  # Ensure predictions match test data length

            if np.isnan(predictions).any():
                print(f"Warning: {model_name} predictions contain NaN values. Skipping evaluation for this model.")
                continue
            if len(predictions) != len(y_test):
                print(f"Warning: {model_name} predictions length does not match test data length. Skipping evaluation for this model.")
                continue
            mse = mean_squared_error(y_test, predictions)
            r2_score_value = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)

            plot_actual_vs_predicted_timeseries(
                y_true=y_test,
                y_pred=predictions,
                index=y_test.index,
                save_path=f"./figures/{model_name}_ts.png",
                ylabel=y_test.name,
                title=f"{model_name}: Actual vs Predicted"
            )
            print(f"{model_name} MSE: {mse:.4f}, R^2: {r2_score_value:.4f}, MAE: {mae:.4f}")
            results.append({
                "Model": model_name,
                "Parameters": model.params,
                "MSE": mse,
                "MAE": mae,
                "R^2": r2_score_value, 
            })
        elif model_type == "regresssion":
            predictions = model.predict(X_test)
            if len(predictions) != len(y_test):
                print(f"Warning: {model_name} predictions length does not match test data length. Skipping evaluation for this model.")
                continue
            mae = mean_absolute_error(y_test.values, predictions)
            mse = mean_squared_error(y_test.values, predictions)
            r2_score_value = r2_score(y_test.values, predictions)
            print(f"{model_name} MSE: {mse:.4f}, R^2: {r2_score_value:.4f}, MAE: {mae:.4f}")
            results.append({
                "Model": model_name,
                "Parameters": model.get_params(),
                "MSE": mse,
                "MAE": mae,
                "R^2": r2_score_value,                
            })
        elif model_type == "classification":
            predictions = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:,1]
                plot_actual_vs_predicted_scatter(
                    y_true=y_test,
                    y_pred=proba,
                    save_path=f"./figures/{model_name}_prob_scatter.png",
                    title=f"{model_name}: Predicted Probability vs Actual"
                )

            if len(predictions) != len(y_test):
                print(f"Warning: {model_name} predictions length does not match test data length. Skipping evaluation for this model.")
                continue
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            print(f"{model_name} Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            results.append({
                "Model": model_name,
                "Parameters": model.get_params(),
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
            })

            cm_path  = f"./figures/classification_performance/{model_name}_confusion_matrix.png"
            roc_path = f"./figures/classification_performance/{model_name}_roc_curve.png"
            prc_path = f"./figures/classification_performance/{model_name}_pr_curve.png"

            plot_paths[model_name] = {
            "confusion_matrix": plot_confusion_matrix(y_test, predictions, labels=[0,1],save_path=cm_path),
            "roc_curve":        plot_roc_curve(y_test, proba, save_path=roc_path) if proba is not None else None,
            "pr_curve":         plot_precision_recall_curve(y_test, proba, save_path=prc_path) if proba is not None else None
            }

            plot_actual_vs_predicted_class_over_time(
                y_train,         # pd.Series, indexed by date
                y_test,          # pd.Series, indexed by date
                predictions,          # array-like, same length/index as y_test
                x_label = "Date",
                figsize=(10,8)
                title="Actual vs Predicted High Risk State Over Time",
                ylabel="High Risk State",
                save_path=f"./figures/classification_performance/actual_vs_predicted_class_over_time_{model_name}.png"
            )
            print("\nEvaluation complete.")

        else:
            print(f"Model {model_name} is not recognized for evaluation.")
    print("Evaluation complete.")
    return pd.DataFrame(results)
