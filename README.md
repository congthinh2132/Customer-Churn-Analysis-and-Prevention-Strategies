# [PYTHON] Customer-Churn-Analysis-and-Prevention-Strategies
## I. Introduction:
### 1. Business questions:
- What factors drive customer churn?
- How can we develop effective strategies to prevent customers from leaving?
### 2. Dataset:
- Churn customers dataset represents the loss of customers who stop using a company's product or service, posing challenges for sustained business growth.
- Understanding churn drivers helps businesses take proactive measures to improve retention and customer satisfaction.
- This dataset contains customer information from a telecommunications network company. The data includes various attributes about each customer and their account details, with a focus on understanding customer churn. Below is a description of the fields in the dataset:

| Field                              | Description                                                                                   | Type of Field  |
|------------------------------------|-----------------------------------------------------------------------------------------------|----------------|
| **Customer ID**                    | Unique identifier for each customer.                                                           | Dimension      |
| **Churn Label**                     | Indicates whether the customer has churned (Yes/No).                                           | Dimension      |
| **Account Length (in months)**     | The duration of the customer's account in months.                                              | Measure        |
| **Local Calls**                     | Number of local calls made by the customer.                                                   | Measure        |
| **Local Mins**                      | Total minutes spent on local calls.                                                           | Measure        |
| **Intl Calls**                      | Number of international calls made by the customer.                                           | Measure        |
| **Intl Mins**                       | Total minutes spent on international calls.                                                   | Measure        |
| **Intl Active**                     | Indicates if the international calling feature is active (Yes/No).                             | Dimension      |
| **Intl Plan**                       | Indicates if the customer is subscribed to an international plan (Yes/No).                    | Dimension      |
| **Extra International Charges**    | Additional charges incurred for international calls.                                          | Measure        |
| **Customer Service Calls**         | Number of calls made to customer service by the customer.                                      | Measure        |
| **Avg Monthly GB Download**        | Average amount of data (in GB) downloaded by the customer per month.                           | Measure        |
| **Unlimited Data Plan**            | Indicates if the customer is subscribed to an unlimited data plan (Yes/No).                   | Dimension      |
| **Extra Data Charges**             | Additional charges incurred for exceeding data limits.                                         | Measure        |
| **State**                           | State where the customer resides.                                                             | Dimension      |
| **Phone Number**                   | Phone number associated with the customer's account.                                          | Dimension      |
| **Gender**                          | Gender of the customer.                                                                        | Dimension      |
| **Age**                             | Age of the customer.                                                                           | Measure        |
| **Under 30**                        | Indicates if the customer is under 30 years old (Yes/No).                                      | Dimension      |
| **Senior**                          | Indicates if the customer is a senior citizen (Yes/No).                                        | Dimension      |
| **Group**                           | Indicates if the customer belongs to a group (Yes/No).                                         | Dimension      |
| **Number of Customers in Group**   | Total number of customers in the same group.                                                   | Measure        |
| **Device Protection & Online Backup** | Indicates if the customer has device protection and online backup services.                | Dimension      |
| **Contract Type**                   | Type of contract the customer is on (e.g., Monthly, Yearly).                                  | Dimension      |
| **Payment Method**                  | Method used by the customer for payments (e.g., Credit Card, Debit Card).                     | Dimension      |
| **Monthly Charge**                  | Monthly charge incurred by the customer.                                                      | Measure        |
| **Total Charges**                   | Total charges incurred by the customer during their account tenure.                            | Measure        |
| **Churn Category**                  | Specific category of churn, if applicable (e.g., Voluntary, Involuntary).                      | Dimension      |
| **Churn Reason**                    | Reason provided for the customer’s churn.                                                     | Dimension      |
## II. Requirments:
- **Jupyter Notebook**: Google Colab
- **Programming Language**: Python 3.x
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Visualization**: Matplotlib, Seaborn
- **Geospatial Data**: GeoPandas
- **Model Saving**: Joblib
## III. Design Think Method and Issue Tree:
### The five steps of desgin thinking:
- **Step 1 - Empathize**:
![image](https://github.com/user-attachments/assets/a9d17f4c-632d-41a8-b9e8-b3f0a597b8ad)
- **Step 2 - Define**:
![image](https://github.com/user-attachments/assets/8e2cdb80-d71e-4781-b98b-e70ba6db7938)
- **Step 3 - Ideate:**:
![image](https://github.com/user-attachments/assets/53c1438d-4a7c-4de8-8b11-7f88f442fbfb)
- **Step 4 - Prototype**:
![image](https://github.com/user-attachments/assets/09380210-9df6-4169-a031-9e87b6080ac1)
- **Step 5 - Review**:
![image](https://github.com/user-attachments/assets/fecfb7f0-c42d-4366-a740-943315714526)
### Issue tree:
![Churn issue tree](https://github.com/user-attachments/assets/3ca8fb54-feee-4475-b9e2-5eb48dd493ee)

## IV. Process:
### 1. Data Preprocessing:
- The columns churn category and churn reason have many nulls, but this is acceptable because those customers are still active users.
- No anomalies detected on the initial review of the dataset.
- The data types appear consistent with no immediate formatting issues.
### 2. Data Visualization (EDA):
All visualizations included in this project, such as bar charts, heatmaps and feature importance plots, were generated using Python's data visualization libraries. Tools like Matplotlib and Seaborn.
#### 2.1 Churn Overview:
![image](https://github.com/user-attachments/assets/c3f9161e-933a-4b46-b077-b60908625fa0)
#### 2.2 Customer Demographics, Charges, Contract and Payment Method:
![image](https://github.com/user-attachments/assets/4e4bf154-ba35-4026-8957-d2abdb5f0684)
#### 2.3 Customer services:
![image](https://github.com/user-attachments/assets/5bf1f47b-5527-499e-a331-d20e4e2eb4c9)
#### 2.4 Extra charges and additional services:
![image](https://github.com/user-attachments/assets/d93828b7-d2db-4371-9e65-7684ac06c809)

### 3. Machine Learning Model Building:
#### 3.1 Features Selection:
- Feature selection is a key step in building effective machine learning models, as it helps to retain only the most important features for prediction.
- **Chi-Square Test** (chi2): Used for selecting **categorical features**. It measures the relationship between each categorical feature and the target variable. Features with higher chi-square scores are considered more relevant to the prediction of churn, while features with lower scores may be discarded.

![Chi-squared-test-in-project](https://github.com/user-attachments/assets/777b50fe-7b7f-43a3-8a22-b0cf8fd379a6)

- **ANOVA F-Test**: Applied to **numerical features** to assess how well each numerical feature can differentiate between different classes of the target variable. Features with higher F-statistics are more significant for predicting churn, while lower F-values may suggest that the feature has little impact on the target.

![Screenshot 2024-12-05 164928](https://github.com/user-attachments/assets/d5c3d433-bd76-4b32-9955-80db5dc35c53)

- **Correlation Heatmap**: A visual tool to analyze the relationships between **numerical features**. By showing the correlation between features, the heatmap helps identify highly correlated features that might lead to multicollinearity, allowing for informed decisions about which features to keep or drop.

![image](https://github.com/user-attachments/assets/5172cbf8-b42f-4238-8213-14ba76222e1a)

#### 3.2 Model Development:
- **Select Input and Output Variables:** Choose the input variables (features) based on the feature selection process above and the output variable (target) from the dataset. The target typically represents the churn status or customer behavior you are predicting.
- **Split the Data:** Use the train_test_split function to divide the data into training (X_train, y_train) and testing (X_test, y_test) sets, maintaining an 80-20% ratio for training and testing, respectively. This ensures the model is trained on a majority of the data and tested on a smaller portion for validation.
- **Build Preprocessing Pipeline:** Identify numerical features (e.g., account length, monthly charges) and categorical features (e.g., gender, payment method) from the dataset.
Use ColumnTransformer: Apply preprocessing steps using the ColumnTransformer. For numerical features, standardize or normalize the data (e.g., using StandardScaler or MinMaxScaler). For categorical features, encode the data using methods like OneHotEncoder or LabelEncoder.
- **Build the Model:** Select a suitable machine learning model, such as the RandomForestRegressor, XGBoost,which is robust and works well for a variety of data types. Predefine the hyperparameters based on domain knowledge or preliminary testing.
- **Create the Model Pipeline:** Combine both preprocessing steps and the machine learning model into a single pipeline using Pipeline. This ensures that the data is preprocessed and transformed consistently during both training and testing phases.
Train and Optimize the Model: Train the model using the training data (X_train, y_train) and apply grid search (GridSearchCV) to find the optimal hyperparameters. Grid search helps systematically test a range of hyperparameter values to improve model performance.
#### 3.2 Model Evaluation:
- **Random Forest**:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| No           | 0.91      | 0.95   | 0.93     | 979     |
| Yes          | 0.84      | 0.74   | 0.78     | 359     |
| Accuracy     | X         | X      | 0.89     | 1338    |
| Macro avg    | 0.87      | 0.84   | 0.86     | 1338    |
| Weighted avg | 0.89      | 0.89   | 0.89     | 1338    |

![image](https://github.com/user-attachments/assets/17d07360-971e-4c79-a67f-01d906afed88)

![image](https://github.com/user-attachments/assets/87d43374-2d8a-4afa-a103-5a4b60458967)

- **XGBoost**:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| No           | 0.92      | 0.95   | 0.93     | 979     |
| Yes          | 0.85      | 0.77   | 0.81     | 359     |
| Accuracy     | X         | X      | 0.90     | 1338    |
| Macro avg    | 0.88      | 0.86   | 0.87     | 1338    |
| Weighted avg | 0.90      | 0.90   | 0.90     | 1338    |

![image](https://github.com/user-attachments/assets/867e3d7d-039b-443a-96fa-c0ce6c5b8bd2)

![image](https://github.com/user-attachments/assets/63a0d55d-aa13-4c89-bca9-8e683a3a894e)

- **Comparison**:
  - Accuracy: XGBoost achieves a slightly higher accuracy (0.90) than Random Forest with 0.89, indicating a small improvement in overall classification performance.
  - Precision: XGBoost shows better precision (0.85) than Random Forest (0.84), meaning XGBoost is slightly more accurate in predicting churn customers (class 1).
  - Recall: XGBoost outperforms Random Forest in recall (0.77 vs. 0.74), demonstrating that XGBoost is better at identifying customers who are likely to churn.
  - F1 Score: XGBoost has a higher F1 score (0.81) compared to Random Forest (0.78), suggesting that XGBoost achieves a better balance between precision and recall.
  - ROC AUC: XGBoost (0.94) performs better than Random Forest (0.93) in ROC AUC, which means XGBoost has a slightly higher ability to distinguish between churn and non-churn customers.

- **Conclusion**: The XGBoost model outperforms the Random Forest model across all major evaluation metrics, including accuracy, precision, recall, F1 score, and ROC AUC. XGBoost is slightly better at identifying churn customers and distinguishing between churn and non-churn classes, making it the preferred model in this case.
## V. Insights:
- The **largest category** of churned customers left due to **competitors (45%)**, indicating customers may be switching to services with better offers or features.
- Other categories include **dissatisfaction (16.2%)**, **attitude issues (16.2%)**, and **pricing concerns (11.3%)**.
- **Churn rates increase with age**, with **older groups (65+)** being more likely to leave.
- Customers using **Paper Check (38.01%)** and **Direct Debit (34.49%)** show higher churn rates compared to **Credit Card users (14.46%)**.
- **Month-to-Month contracts** have the highest churn rate at **46.29%**, especially within the **first 0-3 months**. Although churn decreases over time, it remains relatively high for this group.
- **One-Year** and **Two-Year contracts** experience **slight churn increases at renewal periods** (e.g., 12 months and 24 months) but maintain much lower churn rates overall.
- Customers **not belonging to group** have a high churn rate, nearing **33%**.
- The average **number of customer service calls is high**, with **2-3 calls** per issue reported.
## VI. Recommendations
- **Enhance Competitiveness**:
  - Offer better pricing and improved device quality.
  - Create attractive offers for new customers and loyalty programs for existing customers.
  - Focus on differentiating with unique value propositions to address competitor-driven churn (45%).
- **Encourage Contract Stability**:
  - Provide incentives for one-year and two-year contracts to stabilize churn rates.
  - Offer exclusive benefits for month-to-month contract users, such as data package discounts and top-up incentives.
- **Promote Credit Card Payments**:
  - Encourage credit card payments, as they correlate with lower churn rates compared to paper checks and direct debit.
- **Optimize Customer Service**:
  - Streamline problem-solving processes to address customer issues quickly and efficiently.
  - Avoid situations where customers must make repeated unresolved calls.
  - Introduce online and phone-based consulting for multi-channel support.
- **Retention-Focused Strategies**:
  - Provide ongoing incentives for loyal customers to reduce churn.
  - Tailor offers to meet the needs of specific customer groups, such as older demographics or high-churn segments.
- **Use the model to early identify churn customers**:
  - Leverage machine learning models above to predict potential churn customers based on past behavior and characteristics, enabling proactive retention efforts.













