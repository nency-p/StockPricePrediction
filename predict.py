import argparse, joblib, os, yfinance as yf, numpy as np
from tensorflow.keras.models import load_model
from utils import add_features

def main(args):
    scaler_X = joblib.load(os.path.join('artifacts','scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join('artifacts','scaler_y.pkl'))
    model = load_model('models/model.keras')

    df = yf.download(args.ticker, period='2y')
    df = add_features(df)

    X_scaled = scaler_X.transform(df.drop(columns=['Close']))
    X_seq = np.expand_dims(X_scaled[-args.lookback:], axis=0)

    pred_scaled = model.predict(X_seq)
    pred = scaler_y.inverse_transform(pred_scaled)

    print(f"Next-day predicted close: {float(pred[0][0]):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--lookback', type=int, default=60)
    args = parser.parse_args()
    main(args)
