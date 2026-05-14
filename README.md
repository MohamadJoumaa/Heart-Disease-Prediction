# Heart Disease Prediction AI/ML Project

## 1. Project Proposal & Problem Definition
**Problem:** Heart disease is a leading cause of mortality globally. Early detection can save lives by allowing for timely medical intervention.
**Approach:** This project builds machine learning models to predict the presence of heart disease based on clinical parameters such as age, sex, cholesterol levels, resting blood pressure, and ECG results. We evaluate Decision Trees, Random Forests, and K-Nearest Neighbors (KNN) algorithms.
**Expected Outcomes:** An accurate machine learning model deployed via a FastAPI backend and a Streamlit dashboard, providing an interactive UI for healthcare professionals or users to predict heart disease risk.

## 2. Project Structure
- `notebook.ipynb`: Jupyter notebook containing EDA, Feature Engineering, Model Training, Evaluation, and Hyperparameter Tuning.
- `api.py`: FastAPI server serving the trained models via a `/predict` REST endpoint.
- `dashboard.py`: Streamlit web interface for interactive predictions without needing to call the API manually.
- `requirements.txt`: Python dependencies.
- `heart.csv`: The dataset.

## 3. Setup Instructions
### Prerequisites
Ensure you have Python 3.8+ installed.

### Installation
1. Clone the repository or navigate to the project directory.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook and Training Models
Before running the API or Dashboard, you must train the models to generate the `.pkl` files.
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
2. Run all cells in the notebook. This will perform the analysis and save `dt_model.pkl`, `rf_model.pkl`, and `knn_model.pkl` to the directory.

### Running the FastAPI Backend
To start the REST API server:
```bash
uvicorn api:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can view the interactive documentation at `http://127.0.0.1:8000/docs`.

### Running the Streamlit Dashboard
To start the interactive web dashboard:
```bash
streamlit run dashboard.py
```
This will open the application in your default web browser where you can input patient data and switch between the trained models.
