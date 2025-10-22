# Store Sales - Time Series Forecasting

This project is a complete machine learning workflow to forecast daily sales for *Corporación Favorita*, a large grocery retailer in Ecuador. The goal is to build a model that accurately predicts unit sales for thousands of items across 54 stores, using a rich dataset that includes sales, promotions, store metadata, and external economic factors.

This project demonstrates skills in:
* Complex data merging (joining 6 different data sources)
* Time series feature engineering (lags, rolling windows, date features)
* Exploratory Data Analysis (EDA)
* Building and validating a time-based machine learning model (LightGBM)
* Optimizing for a specific business metric (RMSLE)

## Problem Statement

Accurate demand forecasting is critical for a retail business.
* **Over-predicting** leads to overstocking, increased waste (especially for perishable goods), and high inventory costs.
* **Under-predicting** leads to empty shelves, lost sales, and poor customer satisfaction.

The objective is to build a model that can provide an accurate 16-day sales forecast for various product families at different stores.

## Data

The dataset is provided by Kaggle and split into six files:
* `train.csv`: The main training data with `date`, `store_nbr`, `family` (product type), `sales`, and `onpromotion`.
* `test.csv`: The test data for the 16 days we need to predict.
* `stores.csv`: Metadata for each store, including `city`, `state`, `type`, and `cluster` (a grouping of similar stores).
* `oil.csv`: Daily oil price. Ecuador's economy is oil-dependent, so this is a key external factor.
* `holidays_events.csv`: A complex file detailing national, regional, and local holidays and events.
* `transactions.csv`: Daily transaction counts for each store.

## Methodology

1.  **Data Loading & Preprocessing:**
    * All six CSV files were loaded into Pandas DataFrames.
    * Date columns were converted to `datetime` objects for time-based merging.
    * The `oil.csv` file's missing values (due to weekends) were filled using a forward-fill (`ffill`) strategy.

2.  **Feature Engineering:**
    * **Date Features:** Extracted `dayofweek`, `month`, `year`, `dayofmonth`, and `is_weekend` to capture seasonality.
    * **Lag Features:** Created `sales_lag_16` and `sales_lag_28` to provide the model with "memory" of past sales, shifting by 16 days to avoid data leakage from the test set.
    * **Rolling Window Features:** Calculated the 16-day and 28-day rolling mean of sales to capture recent trends.
    * **External Data:** Merged store, oil, holiday, and transaction data into a single, comprehensive training frame.

3.  **Modeling:**
    * **Model Choice:** A **LightGBM (LGBM) Regressor** was chosen. It is a gradient-boosting framework that is extremely fast, memory-efficient, and excellent at handling large tabular datasets. It also has built-in support for categorical features.
    * **Validation Strategy:** A time-based validation split is essential for time series data. The last 16 days of the training data were used as a validation set to mimic the test set, ensuring the model was evaluated on its ability to predict the "future."
    * **Evaluation Metric:** The model was optimized for **Root Mean Squared Logarithmic Error (RMSLE)**, as specified by the problem. This involved predicting the `log1p(sales)` and converting back with `expm1()` for the final submission.

## Results & Conclusion

The model performed well, with the most important features being the lag and rolling-window sales features. This confirms that recent sales trends are the strongest predictors of future sales. External factors like `onpromotion` and `transactions` also ranked highly.

This project provides a solid framework for retail demand forecasting. Future improvements could include more complex holiday features (e.g., "days until next holiday") and more advanced hyperparameter tuning.

## How to Run

1.  Clone this repository.
2.  Download the data from the [Kaggle competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and place the CSV files in the same directory.
3.  Install the required libraries: `pip install -r requirements.txt`
4.  Open and run the `store-sales-forecasting.ipynb` Jupyter Notebook.
