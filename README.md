# Mobile Price Prediction — Multi-Class Classification

**Course:** Introduction to Data Science  
**Institute:** BITS Pilani WILP — Cloud Computing, Semester 2  
**Assignment:** Assignment 1  

---

## Group 13 — Members

| Name | ID |
|------|----|
| Moulik Patra | 2025MT03009 |
| Vedpathak Yuvraj Vijay | 2025MT03141 |
| Ajeet Kumar Yadav | 2025MT03083 |

---

## Objective

A mobile manufacturing company wants to categorise mobile phones into four price ranges based on product specifications. This project builds, trains, and evaluates multiple multi-class classification models to predict the `price_range` of a mobile phone from its technical specifications, and deploys the best model as an interactive Streamlit web application.

**Target variable:** `price_range`

| Class | Label |
|-------|-------|
| 0 | Low Cost |
| 1 | Medium-Low Cost |
| 2 | Medium-High Cost |
| 3 | High Cost |

---

## Dataset

| File | Description |
|------|-------------|
| `Mobile_Price_Prediction_train.csv` | 2000 labelled records used for training and evaluation |
| `Mobile_Price_Prediction_test.csv` | Unlabelled records for inference |

### Features (21 input columns)

| Feature | Description |
|---------|-------------|
| `battery_power` | Battery capacity (mAh) |
| `blue` | Bluetooth support (Yes / No) |
| `clock_speed` | Processor clock speed (GHz) |
| `dual_sim` | Dual SIM support (Yes / No) |
| `fc` | Front camera megapixels |
| `four_g` | 4G support (0 / 1) |
| `int_memory` | Internal storage (GB) |
| `m_dep` | Mobile depth / thickness (cm) |
| `mobile_wt` | Mobile weight category (Low / Med / High) |
| `n_cores` | Number of processor cores |
| `pc` | Primary camera megapixels |
| `px_height` | Screen resolution height (pixels) |
| `px_width` | Screen resolution width (pixels) |
| `ram` | RAM (MB) |
| `sc_h` | Screen height (cm) |
| `sc_w` | Screen width (cm) |
| `talk_time` | Maximum talk time (hours) |
| `three_g` | 3G support (0 / 1) |
| `touch_screen` | Touch screen (0 / 1) |
| `wifi` | WiFi support (0 / 1) |
| `id` | Mobile identifier |

**Engineered features (added during preprocessing):**

| Feature | Formula |
|---------|---------|
| `screen_area` | `px_height × px_width` |
| `battery_per_core` | `battery_power / n_cores` |

---

## Project Structure

```
Assignment 1/
├── .gitignore
├── README.md
├── Mobile_Price_Prediction_train.csv    # Raw training data
├── Mobile_Price_Prediction_test.csv     # Raw test data
└── mobile-price-ml/
    ├── Group 13.ipynb                    # Main analysis notebook
    ├── advanced-ui-prediction.py        # Streamlit app — advanced dashboard
    ├── simple-ui-prediction.py          # Streamlit app — basic prediction UI
    ├── Mobile_Price_Prediction_train.csv
    ├── Mobile_Price_Prediction_test.csv
    └── venv/                            # Python virtual environment (git-ignored)
```

---

## Notebook Workflow (`Group 13.ipynb`)

The notebook is organised into the following sections:

1. **Import Libraries and Dataset** — Load CSV, verify target column
2. **Data and Data Models** — Shape, dtypes, data pipeline design
3. **Data Visualization and Exploration** — Statistical summary, class distribution, univariate and bivariate analysis, correlation heatmap
4. **Data Pre-processing and Cleaning** — Missing value check, outlier handling, skewness, encoding (Label, Ordinal, One-Hot), data balancing discussion, feature engineering, standardisation and normalisation, feature importance
5. **Model Building** — Train/test split, Decision Tree, Random Forest, SVM, Voting Ensemble
6. **Performance Evaluation** — Accuracy, confusion matrix, classification report (precision / recall / F1 per class), model fit analysis
7. **Model Deployment** — Save best model with `joblib`, run Streamlit app

---

## Model Results

| Model | Accuracy |
|-------|----------|
| Decision Tree | ~83% |
| **Random Forest** | **~87%** |
| SVM (with StandardScaler) | ~85% |
| Voting Ensemble (DT + RF) | ~86% |

**Best model: Random Forest Classifier** — highest accuracy, balanced precision/recall across all four classes, robust to overfitting through ensemble averaging.

---

## Setup and Usage

### Prerequisites

- Python 3.10+
- pip

### 1. Create and activate a virtual environment

```bash
cd "Assignment 1/mobile-price-ml"
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit joblib notebook
```

### 3. Run the notebook

```bash
jupyter notebook "Group 13.ipynb"
```

Run all cells top-to-bottom. This will train the models and save the best model as `mobile_price_model.pkl`.

### 4. Launch the Streamlit app

```bash
cd "Assignment 1/mobile-price-ml"
streamlit run advanced-ui-prediction.py
```

The app accepts phone specifications as inputs, computes derived features, and returns the predicted price range class using the saved Random Forest model.

---

## Key Findings

- **RAM** is the single strongest predictor of price range — higher RAM strongly correlates with higher price class.
- **Battery power**, **screen resolution** (`px_height`, `px_width`), and **screen area** (engineered feature) are also highly influential.
- The dataset is **well-balanced** across all four classes (~500 records per class), so no resampling was necessary.
- **Decision Tree** shows slight overfitting; **Random Forest** provides the best bias-variance trade-off.
- **SVM** requires feature scaling but delivers competitive accuracy.
- The **Voting Ensemble** stabilises predictions but does not exceed Random Forest performance.

---

## Submission

Per assignment guidelines, only two files were submitted via Taxila:
- `Group 13.ipynb` — Jupyter notebook with all outputs
- `Group 13.html` / `Group 13.pdf` — Rendered output of the notebook
