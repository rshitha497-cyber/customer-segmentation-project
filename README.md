# Customer Segmentation Using K-Means Clustering

## Project objective
Group customers with similar age, annual income and spending score so that a business can understand customer behaviour and plan suitable marketing strategies.

## Tools
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn (StandardScaler, KMeans, PCA)

## Dataset
`customers.csv` contains:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

If the CSV file is absent, `main.py` generates a reproducible sample dataset of 250 customers. The sample data is synthetic and must not be presented as real customer data.

## Setup in VS Code
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. (Recommended) Create a virtual environment:
   - Windows: `python -m venv .venv`
   - Activate: `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv`
   - Activate: `source .venv/bin/activate`
5. Install packages: `pip install -r requirements.txt`
6. Run: `python main.py`

## Outputs
The program creates an `output` folder containing:
- `elbow_method.png`
- `customer_segments.png`
- `pca_clusters.png`
- `segmented_customers.csv`
- `customer_summary.csv`

## Methodology
1. Load the CSV dataset or generate sample data.
2. Validate required columns and handle missing numeric values using the median.
3. Standardize age, income and spending score.
4. Plot the elbow curve.
5. Fit K-Means with five clusters for this demonstration dataset.
6. Evaluate separation using the silhouette score.
7. Save cluster labels, summary statistics and visualizations.

## Important notes
- Five clusters are selected because the generated sample data was designed with five broad groups. For a new real dataset, inspect the elbow plot and compare silhouette scores for several values of k before deciding.
- Cluster numbers (0 to 4) are arbitrary labels. They do not automatically mean low, medium or high value.
- Income is represented in thousands in the sample dataset. Confirm the units in any real dataset.
- This is an educational analytics project, not a production customer profiling system.

## Project structure
```
Customer_Segmentation_Project/
  main.py
  customers.csv
  requirements.txt
  README.md
  Project_Report.pdf
  output/
    elbow_method.png
    customer_segments.png
    pca_clusters.png
    segmented_customers.csv
    customer_summary.csv
```
