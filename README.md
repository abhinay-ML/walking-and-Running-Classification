# walking-and-Running-Classification
Create a predictive model to classify whether a person is running or walking based on the given predictor variables.
# Walking vs Running Classification

A compact notebook-based project that classifies wrist sensor data into **Walking** or **Running** using acceleration and gyroscope readings. The notebook (`walking running classification.ipynb`) performs data loading (`walkrun.csv`), EDA (distributions, boxplots, correlation heatmap), feature engineering (acceleration and gyro magnitude), scaling, and trains three models (Random Forest, SVM with RBF kernel, and a small MLP). It evaluates models with accuracy, classification reports, and a confusion matrix, and includes a sample prediction snippet.

## Files
- `walking running classification.ipynb` — main notebook (includes EDA, models, and plots)
- `walkrun.csv` — dataset (place alongside the notebook)
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Quick start
1. Clone or download the repo and place `walkrun.csv` next to the notebook.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the notebook:
   ```bash
   jupyter notebook "walking running classification.ipynb"
   ```

## Notes & suggestions
- The notebook standardizes features with `StandardScaler` and uses a stratified 80/20 train/test split.
- To export the final trained model, save the scikit-learn estimator with `joblib.dump(model, 'model.pkl')` or `pickle`.
- If you want a `requirements.txt` with exact versions, tell me and I can generate one capturing your environment.

## License
MIT — feel free to reuse and modify for educational projects.
