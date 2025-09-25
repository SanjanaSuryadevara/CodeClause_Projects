# Personality Prediction System via CV Analysis

**All artifacts are now saved as flat files** (no `artifacts_bigfive/` folder). Files created by the notebook in the **same directory**:
- `tfidf_b5.pkl`
- `bigfive_ridge.pkl`
- `label_minmax.pkl`
- `meta.json`

## Steps
1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run `personality_prediction.ipynb` to train and create the 4 files above **in the same folder**.
3. Make sure `app.py` and the artifact files are together.
4. Run the app:
   ```bash
   streamlit run app.py
   ```

> Educational demo. Do **not** use for hiring decisions.
