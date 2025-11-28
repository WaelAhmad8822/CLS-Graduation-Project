# Retail Sales Forecasting API

This repository contains the notebooks, serialized model, and lightweight Flask API that together power a weekly retail sales forecasting service. The model ingests engineered store, promotion, and customer features and returns a gradient boosting–based forecast that is currently deployed to PythonAnywhere for quick experimentation and sharing.

Live demo: [https://waelalnaqiti5.pythonanywhere.com/](https://waelalnaqiti5.pythonanywhere.com/)  
The landing route simply confirms that the service is up, while `/predict` accepts JSON payloads for inference. The deployment mirrors the local setup described below and loads the bundled `gbr_pipeline.pkl` artifact so results are consistent between environments.

## Repository Layout

- `PREPROCESSING_&_MODELING.ipynb` – end-to-end data wrangling, feature engineering (time-based, lagged sales, markdown aggregations, categorical encoding), model selection, and evaluation.
- `UseFlask.ipynb` – minimal Flask application that exposes the serialized pipeline for inference.
- `gbr_pipeline.pkl` – fitted `GradientBoostingRegressor` pipeline (preprocessor + regressor) saved via `joblib`.
- `requirements.txt` – complete environment specification used both locally and on PythonAnywhere (includes Flask, scikit-learn, xgboost/lightgbm for experimentation, and supporting notebook tooling).

## Model Highlights

- Historical Walmart-style weekly sales data is merged with store metadata, engineered promotions (e.g., summed markdowns, binary promotion flag), and customer KPIs.
- Temporal signals include year/month/week/day features plus a 4-week lag per store/department to capture trend.
- Numerical columns are scaled with `MinMaxScaler`, categorical columns (store `Type`, fuel price bins, holidays) are one-hot encoded, and the resulting matrix feeds a `GradientBoostingRegressor`.
- The fitted pipeline (preprocessing + estimator) is persisted to `gbr_pipeline.pkl`, ensuring identical transformations at inference time.

## Running Locally

1. **Create environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Serve the API**
   ```bash
   jupyter nbconvert --to python UseFlask.ipynb
   python UseFlask.py
   ```
   The script loads `gbr_pipeline.pkl` relative to the repo root and starts Flask on `http://127.0.0.1:5000`.
3. **Request a prediction**
   ```bash
   curl -X POST http://127.0.0.1:5000/predict ^
        -H "Content-Type: application/json" ^
        -d "{\"Store\":[1],\"Dept\":[1],\"IsHoliday\":[0],\"Temperature\":[60.5],\"Fuel_Price\":[2.48],\"CPI\":[211.1],\"Unemployment\":[7.1],\"MarkDown1\":[0],\"MarkDown2\":[500],\"MarkDown3\":[0],\"MarkDown4\":[0],\"MarkDown5\":[0],\"Num_Customers\":[550],\"Avg_Spend_per_Customer\":[45.3],\"Loyalty_Avg\":[0.62]}"
   ```
   Supply all features expected by the pipeline (store metadata, markdown totals, engineered customer metrics, etc.). If you retrain with different columns, regenerate and redeploy `gbr_pipeline.pkl`.

## Reproducing Training

1. Place the original CSVs (`train.csv`, `features.csv`, `stores.csv`, `customer_train.csv`) under an accessible path.
2. Open `PREPROCESSING_&_MODELING.ipynb` in Jupyter/Colab.
3. Update the data paths in the loading cell if needed.
4. Run the notebook to reproduce preprocessing, randomized search across regressors (Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM), and evaluation metrics (RMSE, MAE, R², MAPE).
5. Export the chosen pipeline with `joblib.dump(GBR_pipeline, "gbr_pipeline.pkl")` for inference.

## Deploying on PythonAnywhere

- Upload `UseFlask.py`, `gbr_pipeline.pkl`, and `requirements.txt` to your PythonAnywhere file system.
- In the PythonAnywhere web tab, create a new Flask web app, point `WSGI` to the generated `UseFlask.py`, and ensure the virtualenv installs all dependencies (`pip install -r requirements.txt`).
- Restart the web app; hitting `/` should respond with `Now Run Successfully......`, confirming the model is live at `https://waelalnaqiti5.pythonanywhere.com/`.

## Notes & Next Steps

- Validate payload schemas before hitting `/predict` to avoid transformation errors (consider Marshmallow or Pydantic for production use).
- Monitor scikit-learn version compatibility when unpickling; keep training and inference versions aligned.
- Extend the Flask app with authentication, logging, or visualization endpoints as needed for your deployment.


