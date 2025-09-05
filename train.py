import argparse, os, json, joblib
import numpy as np, pandas as pd, yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import add_features
from model import build_model

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i+lookback)])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

def main(args):
    df = yf.download(args.ticker, start=args.start, end=None if args.end=='auto' else args.end)
    df = add_features(df)

    features = df.drop(columns=['Close'])
    target = df['Close']

    # Use separate scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(features)
    y_scaled = scaler_y.fit_transform(target.values.reshape(-1,1))

    # Save both scalers
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler_X, os.path.join('artifacts','scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join('artifacts','scaler_y.pkl'))

    split1 = int(len(X_scaled)*0.7)
    split2 = int(len(X_scaled)*0.85)
    X_train, X_val, X_test = X_scaled[:split1], X_scaled[split1:split2], X_scaled[split2:]
    y_train, y_val, y_test = y_scaled[:split1], y_scaled[split1:split2], y_scaled[split2:]

    X_train, y_train = create_sequences(X_train, y_train, args.lookback)
    X_val, y_val = create_sequences(X_val, y_val, args.lookback)
    X_test, y_test = create_sequences(X_test, y_test, args.lookback)

    model = build_model((args.lookback, X_train.shape[2]), rnn=args.rnn)
    os.makedirs('models', exist_ok=True)
    es = EarlyStopping(patience=5, restore_best_weights=True)
    mc = ModelCheckpoint('models/model.keras', save_best_only=True)

    history = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                        epochs=args.epochs, batch_size=32,
                        callbacks=[es, mc])

    preds = model.predict(X_test)
    y_true = y_test

    # RMSE for old sklearn compatibility
    rmse = mean_squared_error(y_true, preds) ** 0.5
    mae = mean_absolute_error(y_true, preds)
    mape = np.mean(np.abs((y_true - preds)/y_true))*100

    metrics = {'RMSE': float(rmse), 'MAE': float(mae), 'MAPE': float(mape)}
    os.makedirs('outputs', exist_ok=True)
    with open(os.path.join('outputs','metrics.json'),'w') as f:
        json.dump(metrics,f,indent=2)
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='auto')
    parser.add_argument('--rnn', type=str, default='lstm', choices=['lstm','gru'])
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    main(args)
