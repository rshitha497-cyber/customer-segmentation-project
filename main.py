"""
Customer Segmentation Using K-Means Clustering
Run: python main.py
If customers.csv is missing, a reproducible sample dataset is generated.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save charts without requiring a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_FILE = BASE_DIR / "customers.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def create_sample_dataset(path: Path) -> pd.DataFrame:
    """Create 250 sample customers across five illustrative behaviour groups."""
    rng = np.random.default_rng(42)
    centers = [
        (25, 20, 20),
        (30, 25, 80),
        (45, 75, 20),
        (40, 80, 80),
        (55, 50, 50),
    ]
    rows = []
    for age, income, spending in centers:
        for _ in range(50):
            rows.append({
                "CustomerID": len(rows) + 1,
                "Gender": rng.choice(["Male", "Female"]),
                "Age": rng.normal(age, 5),
                "Annual Income (k$)": rng.normal(income, 6),
                "Spending Score (1-100)": rng.normal(spending, 7),
            })

    df = pd.DataFrame(rows)
    df["Age"] = df["Age"].clip(18, 80).round().astype(int)
    df["Annual Income (k$)"] = (
        df["Annual Income (k$)"].clip(10, 150).round().astype(int)
    )
    df["Spending Score (1-100)"] = (
        df["Spending Score (1-100)"].clip(1, 100).round().astype(int)
    )
    df.to_csv(path, index=False)
    return df


def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        print(f"Loading dataset: {DATA_FILE.name}")
        df = pd.read_csv(DATA_FILE)
    else:
        print("customers.csv not found. Creating sample dataset...")
        df = create_sample_dataset(DATA_FILE)

    required = ["CustomerID", "Gender"] + FEATURES
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    # Convert numeric columns safely; invalid values become missing and are imputed.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() == 0:
            raise ValueError(f"Column '{col}' contains no valid numeric values.")
        df[col] = df[col].fillna(df[col].median())

    if len(df) < 5:
        raise ValueError("At least five customer rows are required for five clusters.")
    return df


def main():
    print("\nCUSTOMER SEGMENTATION USING K-MEANS")
    print("=" * 42)

    df = load_data()
    print(f"Dataset loaded: {len(df)} customers")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    # Scale features so age, income and spending contribute comparably.
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    print("\nPreprocessing and feature scaling completed.")

    # Elbow method: compare within-cluster sum of squares for k=1..10.
    max_k = min(10, len(df) - 1)
    inertias = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.title("Elbow Method for Selecting Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, max_k + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=160)
    plt.close()

    # This example uses five clusters, matching the five synthetic groups.
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    # Silhouette score measures cluster separation (-1 to 1).
    score = silhouette_score(X, df["Cluster"])
    print(f"\nK-Means clustering completed with {n_clusters} clusters.")
    print(f"Silhouette score: {score:.3f}")

    summary = df.groupby("Cluster").agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean"),
    ).round(2)

    print("\nCUSTOMER SEGMENT SUMMARY")
    print(summary.to_string())

    # Main customer segmentation chart.
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="Set1",
        s=65,
    )
    plt.title("Customer Segments: Income vs Spending Score")
    plt.xlabel("Annual income (thousands)")
    plt.ylabel("Spending score (1-100)")
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_segments.png", dpi=160)
    plt.close()

    # PCA projects the three scaled features into two dimensions for plotting.
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    pca_df = pd.DataFrame({
        "PC1": reduced[:, 0],
        "PC2": reduced[:, 1],
        "Cluster": df["Cluster"].astype(str),
    })
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2",
        hue="Cluster", palette="Set1", s=65
    )
    plt.title("Customer Segments Visualized Using PCA")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=160)
    plt.close()

    # Save reproducible outputs.
    df.to_csv(OUTPUT_DIR / "segmented_customers.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "customer_summary.csv")

    print("\nPROJECT COMPLETED SUCCESSFULLY!")
    print(f"Total customers: {len(df)}")
    print(f"Number of clusters: {df['Cluster'].nunique()}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Created: elbow_method.png, customer_segments.png, pca_clusters.png")
    print("Created: segmented_customers.csv, customer_summary.csv")


if __name__ == "__main__":
    main()
