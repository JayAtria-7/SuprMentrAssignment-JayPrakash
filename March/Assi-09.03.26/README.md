# House Price Predictor (Linear Regression)

**Assignment:** Train a Linear Regression model, predict prices, and test with new input.

## What’s included
- `house_price_predictor.py` trains a Linear Regression model and saves it.
- A sample dataset is auto-created at `data/house_prices.csv` (if it doesn’t exist).
- Model is saved to `model/house_price_model.joblib`.

## Run
Train + then enter a new house interactively:
```bash
python house_price_predictor.py
```

Train only:
```bash
python house_price_predictor.py train
```

Predict using a saved model (CLI input):
```bash
python house_price_predictor.py predict --sqft 2000 --bedrooms 3 --bathrooms 2 --age 10 --distance-to-city 8
```

