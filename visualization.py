import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from scipy import stats

# Load the similarity results
try:
    df = pd.read_csv('Hallucination/clip_similarity_results.csv')
    print(f"Loaded data with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    # Create some sample data for testing if file doesn't exist
    df = pd.DataFrame({
        'similarity_score': np.random.normal(30, 10, 100)
    })
    print("Created sample data for demonstration")

# Extract similarity scores
similarity_scores = df['similarity_score'].values

# 1. Histogram with KDE (Kernel Density Estimation)
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(similarity_scores, kde=True, bins=30)
plt.title('Histogram of CLIP Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')

# 2. Box plot for outlier visualization
plt.subplot(2, 2, 2)
sns.boxplot(y=similarity_scores)
plt.title('Box Plot of Similarity Scores')
plt.ylabel('Similarity Score')

# 3. Strip plot for distribution visualization
plt.subplot(2, 2, 3)
sns.stripplot(y=similarity_scores, jitter=True, alpha=0.4)
plt.title('Strip Plot of Individual Scores')
plt.ylabel('Similarity Score')

# 4. Attempt to identify clusters with KMeans
# Determine optimal number of clusters using silhouette score
max_clusters = min(10, len(similarity_scores) - 1)  # Cap at 10 clusters
if max_clusters > 1:
    from sklearn.metrics import silhouette_score
    
    # Reshape for sklearn
    X = similarity_scores.reshape(-1, 1)
    
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we started from 2
    
    # Apply KMeans with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Plot clusters
    plt.subplot(2, 2, 4)
    for cluster in range(optimal_clusters):
        cluster_data = df[df['cluster'] == cluster]['similarity_score']
        sns.kdeplot(cluster_data, label=f'Cluster {cluster}')
    
    plt.title(f'KDE of {optimal_clusters} Identified Clusters')
    plt.xlabel('Similarity Score')
    plt.legend()
else:
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, 'Not enough data\nfor clustering', 
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')

plt.tight_layout()
plt.savefig('similarity_distribution_no_hallucination.png')

# Create a separate figure for a more detailed 1D scatter plot
plt.figure(figsize=(14, 4))
plt.scatter(range(len(similarity_scores)), np.sort(similarity_scores), alpha=0.6)
plt.title('Sorted Similarity Scores (1D Scatter Plot)')
plt.xlabel('Index (sorted)')
plt.ylabel('Similarity Score')
plt.grid(True, alpha=0.3)

# Add potential threshold lines
q25 = np.percentile(similarity_scores, 25)
q75 = np.percentile(similarity_scores, 75)
median = np.median(similarity_scores)

plt.axhline(y=q25, color='r', linestyle='--', alpha=0.7, label=f'25th Percentile: {q25:.2f}')
plt.axhline(y=median, color='g', linestyle='--', alpha=0.7, label=f'Median: {median:.2f}')
plt.axhline(y=q75, color='b', linestyle='--', alpha=0.7, label=f'75th Percentile: {q75:.2f}')

# Calculate potential outliers using IQR method
iqr = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

outliers = df[(df['similarity_score'] < lower_bound) | (df['similarity_score'] > upper_bound)]
print(f"Number of potential outliers: {len(outliers)}")

# Highlight potential outliers
if not outliers.empty:
    # Find the indices of outliers in the sorted array
    sorted_indices = np.argsort(similarity_scores)
    outlier_indices = [np.where(sorted_indices == i)[0][0] for i in outliers.index if i in sorted_indices]
    
    plt.scatter([outlier_indices], [outliers['similarity_score']], 
                color='red', s=50, alpha=0.8, label='Outliers')

plt.legend()
plt.tight_layout()
plt.savefig('similarity_scores_1d_no_hallucination.png')

# Print summary statistics
print("\nSummary Statistics:")
print(df['similarity_score'].describe())

# If clusters were identified, print cluster information
if max_clusters > 1 and 'cluster' in df.columns:
    print("\nCluster Statistics:")
    cluster_stats = df.groupby('cluster')['similarity_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(cluster_stats)
    
    # Save clusters to CSV for further analysis
    df.to_csv('similarity_with_clusters.csv', index=False)
    print("Saved similarity scores with cluster assignments to 'similarity_with_clusters.csv'")

print("\nVisualization complete. Check 'similarity_distribution.png' and 'similarity_scores_1d.png'")