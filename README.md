# Heart Disease Prediction

**Live demo:** [https://heartdiseasepredictionss.streamlit.app/](https://heartdiseasepredictionss.streamlit.app/)

Classify heart-disease risk from routine clinical features (age, sex, cholesterol, resting blood pressure, ECG, and related measurements). This is a **demo / portfolio project**, not a medical diagnostic tool — predictions are educational only and are not a substitute for professional care.

## Problem

Heart disease is a leading cause of death worldwide. The goal here is a small, end-to-end ML app: train classifiers on tabular clinical data, compare them, then serve predictions through a REST API and an interactive dashboard.

## Models & results

Three scikit-learn classifiers, trained in `notebook.ipynb` (80/20 stratified split, F1 as the primary metric):

| Model | Test accuracy | Test F1 |
| --- | --- | --- |
| Decision Tree | 0.78 | 0.80 |
| Random Forest | 0.90 | 0.91 |
| K-Nearest Neighbors (KNN) | 0.89 | 0.90 |

Random Forest was also tuned with 5-fold `GridSearchCV` (`n_estimators`, `max_depth`, `min_samples_split`): **CV F1 0.89**, **test F1 0.90**. The dashboard and API load that tuned forest as `rf_model.pkl`.

## Stack

- **Scikit-learn** — preprocessing pipelines, Decision Tree / Random Forest / KNN, evaluation
- **FastAPI** — `/predict` REST endpoint
- **Streamlit** — interactive UI (live demo above)
- **pandas**, **joblib**, **Jupyter** — data, model persistence, notebook workflow

## Project layout

| File | Role |
| --- | --- |
| `notebook.ipynb` | EDA, training, evaluation, hyperparameter tuning |
| `api.py` | FastAPI server |
| `dashboard.py` | Streamlit app |
| `heart.csv` | Training / demo dataset |
| `dt_model.pkl`, `rf_model.pkl`, `knn_model.pkl` | Pretrained pipelines so the demo runs without retraining |
| `requirements.txt` | Python dependencies |

## Setup

Python 3.8+ (3.11 works well). Prefer `python -m …` so the same commands work on Linux, macOS, and Windows. On Windows you can use `py -m` instead of `python -m` if that is how Python is installed.

```bash
git clone https://github.com/MohamadJoumaa/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (cmd): .venv\Scripts\activate.bat
# Windows (PowerShell): .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

### Streamlit dashboard

The committed `.pkl` files are enough to run the UI:

```bash
python -m streamlit run dashboard.py
```

### FastAPI backend

```bash
python -m uvicorn api:app --reload
```

Open interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Retrain models (optional)

To regenerate `dt_model.pkl`, `rf_model.pkl`, and `knn_model.pkl`:

```bash
python -m jupyter notebook notebook.ipynb
```

Run all cells. Retraining is not required for the live demo or a local dashboard/API session.

## License

MIT — see [LICENSE](LICENSE).
