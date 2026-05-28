# Term-Deposit-Subscription-Prediction
This project uses machine learning models like Logistic Regression and Random Forest to predict whether a bank customer will subscribe to a term deposit based on their background and past marketing calls.

## 1. Project Objective
The goal of this project is to help a banking institution optimize its marketing strategies. Using the UCI Bank Marketing Dataset (`bank-full.csv`), we built a machine learning system that predicts whether a customer will subscribe to a term deposit (`yes` or `no`) based on their demographic data, financial history, and past campaign interactions. 

By accurately identifying potential customers, the bank can direct its resources toward individuals who are most likely to convert, significantly reducing marketing costs and improving campaign efficiency.

## 2. Our Approach

To tackle this classification problem, we followed a structured data science workflow directly inside a Jupyter Notebook:

### Data Cleaning & Preprocessing
* **Handling Missing & Duplicate Data:** Checked for missing variables and removed duplicate records to ensure data integrity.
* **Target Mapping:** Converted the target column `y` from text values (`yes` / `no`) into standard binary numeric values (`1` / `0`).
* **Categorical Encoding:** Applied one-hot encoding (`pd.get_dummies`) to convert non-numeric features like job profiles, education levels, and marital status into a format readable by our models.
* **Feature Scaling:** Applied `StandardScaler` to ensure our numerical features (like account balance and age) were on the same scale, which is essential for linear algorithms.

### Exploratory Data Analysis (EDA)
* Analyzed the distribution of the target variable, noticing a clear data imbalance (significantly more 'no' responses than 'yes').
* Created visualization charts, including distribution histograms for age groups and a correlation matrix heatmap, to spot relationships before modeling.

### Model Training & Evaluation
We split our dataset into an 80% training set and a 20% testing set (using stratification to preserve class balance). We then built and compared two distinct classifiers:
1. **Logistic Regression:** Used as our baseline linear model.
2. **Random Forest Classifier:** Used as our tree-based ensemble method to capture non-linear interactions between client attributes.

Both models were thoroughly evaluated using metrics beyond simple accuracy, including **Confusion Matrices**, **F1-Scores**, and **ROC-AUC Curves** to account for the imbalanced dataset.

### Advanced Interpretability (SHAP)
To avoid treating our best model as a complete "black box," we integrated **SHAP (SHapley Additive exPlanations)**. We specifically extracted and explained 5 individual customer predictions using waterfall plots to show exactly which features pushed the model toward its final decision.

## 3. Key Results and Findings

* **Model Performance:** The **Random Forest Classifier** outperformed Logistic Regression, achieving a higher ROC-AUC score and a more stable F1-score for the positive (`yes`) class. It managed complex feature combinations much better.
* **The Imbalance Challenge:** Because the majority of clients did not subscribe, relying purely on accuracy was misleading. Evaluating the precision-recall trade-off gave us a much truer look at real-world performance.
* **Top Predictive Drivers:** According to our global and individual SHAP analyses, the two most powerful indicators of a subscription are:
  1. **Duration:** The length of the marketing phone call (longer conversations heavily correlate with a 'yes').
  2. **Poutcome (Success):** Whether the customer had accepted an offer in a previous marketing campaign.
* **Actionable Insight:** The bank should focus on retaining clients who have successfully interacted with them in the past and train agents to keep conversations engaging, while avoiding cold calls to demographics the model flags as highly unlikely to subscribe.
