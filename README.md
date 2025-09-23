## University Student Clusters Dashboard (Streamlit)

This app lets university employees explore clustering results from the student dataset (`EncodedWomen_english.csv`). It supports uploading another CSV, choosing the cluster label column, interactive 2D projections (PCA/t‑SNE), feature summaries, filtering, search, and downloading filtered data.

### 1) Requirements

- Python 3.9–3.12
- See `requirements.txt` for Python packages

### 2) Install dependencies (Windows PowerShell)

```powershell
cd C:\Users\PartZ\Desktop\Python\Khu-Dorm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Place your data

- Put `EncodedWomen_english.csv` in the project root (same folder as `app.py`).
- Or upload a CSV via the app sidebar at runtime.

### 4) Run the app

```powershell
streamlit run app.py
```

Your browser will open to the local URL (usually `http://localhost:8501`).

### 5) Notes

- Cluster column detection is heuristic. Use the sidebar to pick the correct label column if needed.
- For large datasets, t‑SNE may take time; try PCA first or reduce selected features.
- Download button exports the currently filtered rows to CSV.

### Dorm room allocation

- In the sidebar, configure Room capacity (default 6), Student identifier, Allocation policy, and Random seed.
- Policies:
  - Pack by cluster: fills rooms with the same cluster first (homogeneous rooms).
  - Round‑robin across clusters: cycles across clusters to increase diversity in each room.
  - Random shuffle: random placement.
- The page shows:
  - Stacked bar chart of cluster composition per room
  - Histogram of room homogeneity (share of majority cluster)
  - Expandable table of assignments and a CSV download button

### Run clustering from the app

- Open the "Run clustering on a CSV and visualize the result" section.
- Upload a CSV or type the input filename, set output filename, Optuna trials, and K settings.
- Click "Run clustering". The app writes a labeled CSV (adds `Cluster` column), previews it, and lets you download it.
- To explore the labeled output in the main dashboard sections, either reload the app and select the new file in the sidebar uploader, or place the output CSV in the project folder and pick it via the sidebar.


