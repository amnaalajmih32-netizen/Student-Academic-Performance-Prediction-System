# Student Academic Performance Prediction System (SAPPS)

An Intelligent Academic Decision Support System (DSS) designed to predict student performance levels, detect academic risks early, and offer data-driven administrative recommendations. 

The project bridges the gap between machine learning predictive accuracy and institutional decision-making by combining robust classification with Research-Grade Explainable AI (XAI).

## Key Features
* **Predictive Architecture:** Utilizes a calibrated Random Forest Classifier achieving high experimental accuracy in identifying students at risk.
* **Explainable AI (XAI):** Integrated with **SHAP (SHapley Additive exPlanations)** to extract global feature importance, providing full transparency for pedagogical decisions.
* **Dual-Environment Logic:** Features adaptive administrative logic tailored dynamically to both School and University regulatory protocols.
* **Operational Dashboard:** An interactive Streamlit UI supporting single-student evaluation and batch CSV processing with automated Excel data export.

## Project Structure
* `app.py` - The main interactive Streamlit dashboard and user interface.
* `model.py` - The complete training pipeline, feature engineering, and SHAP evaluation script.
* `student-mat-fixed.csv` - The validated student performance dataset used for experimental training.
* `*.pkl` - Pre-trained model artifacts and calculated metrics ensuring complete reproducibility.

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install streamlit pandas joblib scikit-learn shap matplotlib seaborn xlsxwriter
## 1. Execution
Run the production dashboard locally:
streamlit run app.py




