import os

# --- Parameters ---
TICKER = "AAPL"          # Stock symbol (change to TSLA, MSFT, etc.)
START_DATE = "2015-01-01"
END_DATE = "auto"
MODEL_TYPE = "lstm"      # or "gru"
LOOKBACK = 60
EPOCHS = 30

# --- Train the model ---
train_cmd = f"python src/train.py --ticker {TICKER} --start {START_DATE} --end {END_DATE} --rnn {MODEL_TYPE} --lookback {LOOKBACK} --epochs {EPOCHS}"
print("Running:", train_cmd)
os.system(train_cmd)

# --- Predict next-day price ---
predict_cmd = f"python src/predict.py --ticker {TICKER} --lookback {LOOKBACK}"
print("Running:", predict_cmd)
os.system(predict_cmd)
