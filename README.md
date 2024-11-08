# QuantTrade_IET_CIPHER
# Hackathon Challenge: Anomaly Detection in Stock Prices

Welcome to the stock anomaly detection hackathon! Your challenge is to work through a series of tasks involving data preprocessing, model training, visualization, and deployment. This README will guide you through the requirements and deliverables for each task.

## Task Overview

### Task 1: Preprocessing of Train Dataset
- *Objective*: Clean and preprocess the provided training dataset.
- *Details*:
  - The training dataset includes the following columns: date, open, high, low, close, volume, dividends, and stock splits.
  - You are encouraged to explore and handle potential data inconsistencies.
  - You may choose any combination of columns or use feature engineering to create transformed data for training.
- *Output*: A preprocessed training dataset ready for model training.

### Task 2: Training a Model
- *Objective*: Train an anomaly detection model using the preprocessed data.
- *Guidelines*:
  - You can use any machine learning or deep learning technique to develop your model.
  - Document the process, including the model selection rationale, training approach, and hyperparameter tuning.

### Task 3: Visualization of Anomalies
- *Objective*: Visualize detected anomalies in the training data.
- *Details*:
  - Use plots to show anomalies detected by your model.
  - Recommended tools: Matplotlib, Seaborn, or Plotly for dynamic visualizations.
- *Output*: Clear and insightful visual plots that highlight anomalies in the notebook itself(no need for separate picture submission). Try to split the train dataset into multiple parts and show the outputs for the years 2004-2024. Each plot dedicated to a single year.

### Task 4: Applying the Model to the Test Dataset
- *Objective*: Apply your trained model to the test dataset and identify anomalies.
- *Details*:
  - The test dataset includes date, company name, adj close, close, high, low, open, and volume columns. Download dataset from here https://drive.google.com/file/d/1Xm8ejwEbVLOMAv_5sL9mXIWCXd_0zMcq/view?usp=sharing.
  - The meanings of these columns are consistent with financial datasets, representing daily trading data.
- *Output*: A CSV file containing company name and date for each detected anomaly.

## Brownie Task: Model Deployment
- *Objective*: Deploy your anomaly detection model as an application that takes a CSV file as input and outputs dates with identified anomalies.
- *Guidelines*:
  - You can deploy the model locally or on a platform using tools such as *Streamlit, **Flask, or **FastAPI*.
  - Save your trained model and provide code for loading and using it in deployment.
  - Record a demo video showcasing the usage of your deployed model.
  - Provide a link to the demo video and include the saved model file for evaluation.
- *Output*:
  - A working link to the demo video demonstrating the deployment.
  - The saved model file and deployment code in your submission package.

## Submission Guidelines
1. *Clone the provided GitHub repository* and create your work in it.
2. *Upload your work as a Jupyter Notebook* with the filename format anomalyX.ipynb, where X is the number of tasks completed (e.g., anomaly3.ipynb if you completed up to Task 3).
3. *Upload the Brownie Task* as a separate folder within the repository with all relevant deployment code.
4. *Make your repository private* before the submission deadline.
5. *Complete the submission by filling out the [Google Form](#)* with the link to your GitHub repository and details on who to add as collaborators for future assessment.

## Dataset Information
### Training Dataset Columns:
- *date*: The date of the stock data entry.
- *open*: The price at which the stock opened on that date.
- *high*: The highest price reached during the trading day.
- *low*: The lowest price reached during the trading day.
- *close*: The price at which the stock closed at the end of the trading day.
- *volume*: The number of shares traded.
- *dividends*: Dividends paid on that day (if any).
- *stock splits*: Any stock splits occurring on that date.

### Test Dataset Columns:
- *date*: The date of the stock data entry.
- *company name*: The name of the company.
- *adj close*: Adjusted closing price considering factors like dividends and splits.
- *close*: The unadjusted closing price.
- *high, **low, **open, **volume*: As described in the training dataset.

## Evaluation Criteria
- *Preprocessing Quality*: Handling of data inconsistencies, feature engineering.
- *Model Performance*: Effectiveness of the anomaly detection approach.
- *Visualization*: Clarity and insights provided by plots.
- *CSV File*: Correctness and format of the anomaly report.
- *Deployment (Brownie Task)*:
  - Functionality and usability of the deployed app.
  - Completeness of the demo video and deployment code.

Good luck, and we look forward to your innovative solutions!
