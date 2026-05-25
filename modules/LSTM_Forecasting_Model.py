import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import os
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

os.makedirs("models", exist_ok=True)

# Loads saved LSTM model and its scaler from disk.
def load_lstm(ticker):
    model = load_model(f"models/{ticker}_lstm_model.keras")
    scaler = joblib.load(f"models/{ticker}_scaler.pkl")
    return model, scaler

# Trains and saves LSTM model for the ticker if one doesn't already exist, otherwise loads it.
def if_not_exists(ticker):
    model_path = f"models/{ticker}_lstm_model.keras"
    scaler_path = f"models/{ticker}_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):

        data = yf.download(tickers=ticker, period='3y', interval='1d', progress=False)
        cle = data[['Close']]

        normalizer = MinMaxScaler(feature_range=(0,1))
        ds_scaled = normalizer.fit_transform(np.array(cle).reshape(-1,1))
        joblib.dump(normalizer, scaler_path)

        train_size = int(len(ds_scaled)*0.70)
        ds_train = ds_scaled[0:train_size,:]
        ds_test = ds_scaled[train_size:,:]

        # Builds sliding-window sequences of length `step` for LSTM input.
        def create_ds(dataset, step):
            Xtrain, Ytrain = [], []
            for i in range(len(dataset)-step-1):
                a = dataset[i:(i+step), 0]
                Xtrain.append(a)
                Ytrain.append(dataset[i + step, 0])
            return np.array(Xtrain), np.array(Ytrain)

        time_stamp = 60
        X_train, y_train = create_ds(ds_train, time_stamp)
        X_test, y_test = create_ds(ds_test, time_stamp)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        plt.figure(figsize=(10,5))
        plt.plot(model.history.history['loss'], label='Training Loss')
        plt.plot(model.history.history['val_loss'], label='Validation Loss')
        plt.title(f'{ticker} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"models/{ticker}_loss_plot.png")
        plt.close()

        #Evaluate on training & testing sets

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict = normalizer.inverse_transform(train_predict)
        y_train_actual = normalizer.inverse_transform(y_train.reshape(-1,1))

        test_predict = normalizer.inverse_transform(test_predict)
        y_test_actual = normalizer.inverse_transform(y_test.reshape(-1,1))

        train_mse = mean_squared_error(y_train_actual, train_predict)
        test_mse = mean_squared_error(y_test_actual, test_predict)

        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        train_mae = mean_absolute_error(y_train_actual, train_predict)
        test_mae = mean_absolute_error(y_test_actual, test_predict)

        train_r2 = r2_score(y_train_actual, train_predict)
        test_r2 = r2_score(y_test_actual, test_predict)

        train_mape = mean_absolute_percentage_error(y_train_actual, train_predict)*100
        test_mape = mean_absolute_percentage_error(y_test_actual, test_predict)*100

        with open(f"models/{ticker}_metrics.txt", "w") as f:
            f.write("Training Metrics\n")
            f.write(f"MSE: {train_mse:.4f}\n")
            f.write(f"RMSE: {train_rmse:.4f}\n")
            f.write(f"MAE: {train_mae:.4f}\n")
            f.write(f"MAPE: {train_mape:.2f}%\n")
            f.write(f"R2: {train_r2:.4f}\n\n")

            f.write("Testing Metrics\n")
            f.write(f"MSE: {test_mse:.4f}\n")
            f.write(f"RMSE: {test_rmse:.4f}\n")
            f.write(f"MAE: {test_mae:.4f}\n")
            f.write(f"MAPE: {test_mape:.2f}%\n")
            f.write(f"R2: {test_r2:.4f}\n")

        model.save(model_path)

        return model, normalizer

    else:
        return load_lstm(ticker)

# Predicts closing prices for the next `future_days` using iterative single-step forecasting.
def predict_future(symbol, model, scaler, future_days=180):
    data = yf.download(symbol, period="3y", interval="1d", progress=False).dropna()

    close_prices = data[['Close']].values
    scaled_data = scaler.transform(close_prices)

    temp_input = scaled_data[-60:].tolist()
    predictions = []

    for _ in range(future_days):
        x_input = np.array(temp_input[-60:]).reshape(1, 60, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.append([yhat[0][0]])
        predictions.append(yhat[0][0])

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return data, np.round(predictions.flatten(), 2)
