# Used Car Price Prediction - Kaggle Competition (Top 5% Finish)

![Car Banner](https://placehold.co/1200x400/334155/e2e8f0?text=Car+Price+Prediction)

URL: https://www.kaggle.com/competitions/masterclass3-predict-car-price/

## 🏆 Key Achievement: 4th Place out of 72 Teams 🏆

This project represents a successful entry into a Kaggle competition focused on predicting the prices of used cars. Through a systematic approach of feature engineering, model selection, and iterative improvement, **this solution achieved 4th place on the final leaderboard**, placing it in the top 5% of all submissions.

---

## 📖 Project Overview

The objective of this project was to develop a high-performing machine learning model to accurately predict the market price of used cars based on a wide range of features, such as make, model, year, mileage, engine specifications, and more.

This repository contains the complete Jupyter Notebook detailing the end-to-end workflow, from initial data exploration and cleaning to advanced feature engineering and final model training. The code is structured to be clean, readable, and well-documented, making it an ideal showcase of practical data science skills.

---

## 🛠️ Tech Stack & Libraries

* **Programming Language:** Python 3.x
* **Core Libraries:**
    * `pandas` & `numpy` for data manipulation and numerical operations.
    * `scikit-learn` for core machine learning utilities.
    * `matplotlib` & `seaborn` for data visualization.
* **Machine Learning Model:** `CatBoost` for its robust handling of categorical features and high performance.
* **Model Inspection:** `eli5` for feature importance analysis.
* **Environment:** Jupyter Notebook

---

## 🔬 Methodology

The solution was developed following a structured data science workflow:

1.  **Data Cleaning & Preprocessing:** Loaded the training and test datasets. The target variable, `price_value`, was parsed from a string format into a numerical type.
2.  **Log Transformation:** Applied a `log1p` transformation to the target variable (`price_value`) to handle its right-skewed distribution, a common practice that helps stabilize model training.
3.  **Feature Engineering:** This was the most critical phase for improving model accuracy. Key engineered features include:
    * **Consistent Year (`year_production_ext`):** Combined two different year-related columns to create a single, more reliable feature for the car's production year.
    * **Numerical Feature Cleaning:** Converted features like `engine_capacity`, `horse_power`, and `mileage` from string formats to clean numerical types.
    * **Outlier Capping:** Capped numerical features at the 99th percentile to reduce the impact of extreme outliers on the model.
    * **Binary Flags:** Created boolean features like `serviced_at_aso` from categorical text to provide clear signals to the model.
4.  **Modeling with CatBoost:**
    * Chose the **CatBoost Regressor** due to its excellent built-in capabilities for handling categorical features without extensive preprocessing (like one-hot encoding).
    * Employed a **cross-validation** strategy to ensure the model's performance was robust and generalizable.
    * Iteratively tested different feature sets and model hyperparameters to systematically improve the Mean Absolute Error (MAE).
5.  **Final Submission:** Trained the best-performing model on the entire training dataset and generated predictions for the test set to create the final submission file.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn catboost eli5 jupyter
    ```

3.  **Place the data:**
    * Download the `df.train.h5` and `df.test.h5` data files from the Kaggle competition page.
    * Place them in the root directory of the project (or update the file paths in the notebook).

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open the `.ipynb` file and run the cells sequentially to reproduce the results.

---

## 💡 Future Improvements

While this model performed exceptionally well, further improvements could be explored:

* **Advanced Feature Engineering:** Create more interaction features (e.g., `mileage_per_year`) or delve deeper into text-based features like `Wersja` (Version).
* **Alternative Models:** Experiment with other powerful gradient boosting models like `LightGBM` or `XGBoost`, or create an ensemble of multiple models.
* **Hyperparameter Optimization:** Conduct a more extensive, automated hyperparameter search using tools like Optuna or Hyperopt to potentially squeeze out additional performance.
