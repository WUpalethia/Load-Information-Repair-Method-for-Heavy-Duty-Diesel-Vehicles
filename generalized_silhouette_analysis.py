import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
import warnings
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# Set plotting style for research publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ComprehensiveKMeansAnalyzer:
    """Comprehensive K-means clustering analyzer testing full p-value range"""
    
    def __init__(self, folder_path, output_folder=None, n_clusters=2, n_init=10, max_clusters_to_test=6):
        self.folder_path = folder_path
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_clusters_to_test = max_clusters_to_test
        
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"comprehensive_kmeans_analysis_{timestamp}"
        else:
            self.output_folder = output_folder
        
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output directory created: {os.path.abspath(self.output_folder)}")
        print(f"Using K-means clustering, initial clusters: {self.n_clusters}")
        
        self.subfolders = {
            'plots': os.path.join(self.output_folder, 'plots'),
            'tables': os.path.join(self.output_folder, 'tables'),
            'reports': os.path.join(self.output_folder, 'reports')
        }
        
        for folder_name, folder_path in self.subfolders.items():
            os.makedirs(folder_path, exist_ok=True)
        
        self.all_data = []
        self.file_names = []
        
        self.selected_columns_mapping = {
            '发动机燃料流量平均值/Average fuel flow': 'avg_fuel_flow',
            '发动机转速平均值/Average engine speed': 'avg_engine_speed', 
            '摩擦扭矩平均值/Average friction torque': 'avg_friction_torque',
            '车速平均值/Average speed': 'avg_speed'
        }
        
        self.load_related_features = list(self.selected_columns_mapping.values())
        
        # Complete p-value range: from negative infinity to positive infinity
        self.p_values = [-np.inf, -2, -1, 0, 1, 2, np.inf]
        self.p_value_names = {
            -np.inf: 'Min\n(p=-∞)',
            -2: 'Quadratic\nMean\n(p=-2)',
            -1: 'Harmonic\nMean\n(p=-1)',
            0: 'Geometric\nMean\n(p=0)',
            1: 'Arithmetic\nMean\n(p=1)',
            2: 'Square\nMean\n(p=2)',
            np.inf: 'Max\n(p=+∞)'
        }
    
    def save_plot(self, plt, filename, subfolder='plots', dpi=300):
        """Save individual plot with high resolution"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white',
                   transparent=False, pad_inches=0.1)
        plt.close()
        print(f"Plot saved: {filename}")
        return filepath
    
    def save_table(self, df, filename, subfolder='tables'):
        """Save data table to CSV"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Table saved: {filepath}")
        return filepath
    
    def generalized_mean(self, x, p):
        """Calculate generalized mean (Hölder mean) - complete version"""
        if len(x) == 0:
            return 0
        
        x = np.array(x)
        
        # Ensure all values are positive (distances cannot be negative)
        x = np.maximum(x, 1e-8)
        
        if p == -np.inf:
            return np.min(x)
        elif p == np.inf:
            return np.max(x)
        elif p == 0:
            # Geometric mean
            return np.exp(np.mean(np.log(x)))
        else:
            return (np.mean(x**p))**(1/p)
    
    def efficient_generalized_silhouette(self, X, labels, p=1, sample_size=2000):
        """Efficient generalized silhouette coefficient calculation"""
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters <= 1:
            return 0
        
        # Use sampling for large datasets
        if sample_size < n_samples:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
            sample_size = n_samples
        
        silhouette_scores = []
        batch_size = 500
        
        for i in range(0, sample_size, batch_size):
            end_idx = min(i + batch_size, sample_size)
            batch_indices = range(i, end_idx)
            
            for j in batch_indices:
                current_label = labels_sample[j]
                
                # Calculate distance to same cluster
                same_cluster_mask = (labels_sample == current_label) & (np.arange(sample_size) != j)
                if np.sum(same_cluster_mask) > 0:
                    distances_same = np.sqrt(np.sum((X_sample[same_cluster_mask] - X_sample[j])**2, axis=1))
                    a_j = self.generalized_mean(distances_same, p)
                else:
                    a_j = 0
                
                # Calculate distance to other clusters
                b_values = []
                for other_label in unique_labels:
                    if other_label != current_label:
                        other_cluster_mask = labels_sample == other_label
                        if np.sum(other_cluster_mask) > 0:
                            distances_other = np.sqrt(np.sum((X_sample[other_cluster_mask] - X_sample[j])**2, axis=1))
                            b_jk = self.generalized_mean(distances_other, p)
                            b_values.append(b_jk)
                
                if len(b_values) > 0:
                    b_j = min(b_values)
                else:
                    b_j = 0
                
                # Calculate silhouette coefficient
                if max(a_j, b_j) > 0:
                    s_j = (b_j - a_j) / max(a_j, b_j)
                else:
                    s_j = 0
                
                silhouette_scores.append(s_j)
        
        return np.mean(silhouette_scores)
    
    def calculate_negative_silhouette_ratio(self, X, labels, sample_size=1000):
        """Calculate proportion of samples with negative silhouette coefficient"""
        if len(np.unique(labels)) <= 1:
            return 0
        
        # Sampling calculation
        if sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        # Calculate silhouette coefficient for each sample
        silhouette_vals = silhouette_samples(X_sample, labels_sample)
        
        # Negative silhouette coefficient ratio
        negative_ratio = np.sum(silhouette_vals < 0) / len(silhouette_vals)
        
        return negative_ratio
    
    def calculate_cluster_quality_metrics(self, labels):
        """Calculate cluster quality metrics (unsupervised version)"""
        n_samples = len(labels)
        if n_samples == 0:
            return {}
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique_labels)
        
        metrics = {}
        
        # 1. Small cluster ratio (may indicate noise or outliers)
        small_cluster_threshold = n_samples * 0.05  # Less than 5% of samples
        small_cluster_ratio = np.sum(counts[counts < small_cluster_threshold]) / n_samples
        metrics['small_cluster_ratio'] = small_cluster_ratio
        
        # 2. Cluster size balance (closer to 1 is more balanced)
        if n_clusters > 1:
            balance_ratio = np.min(counts) / np.max(counts)
        else:
            balance_ratio = 1.0
        metrics['cluster_balance_ratio'] = balance_ratio
        
        # 3. Cluster size standard deviation (smaller means more uniform sizes)
        cluster_size_std = np.std(counts) / n_samples
        metrics['normalized_cluster_size_std'] = cluster_size_std
        
        return metrics
    
    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        print("Loading pickle files...")
        start_time = time.time()
        
        pkl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        
        if not pkl_files:
            raise ValueError("No pickle files found!")
        
        for file_name in pkl_files:
            file_path = os.path.join(self.folder_path, file_name)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, pd.DataFrame):
                    chinese_columns = list(self.selected_columns_mapping.keys())
                    missing_cols = [col for col in chinese_columns if col not in data.columns]
                    
                    if missing_cols:
                        if all(col in data.columns for col in self.load_related_features):
                            selected_data = data[self.load_related_features].copy()
                        else:
                            continue
                    else:
                        selected_data = data[chinese_columns].copy()
                        selected_data.columns = self.load_related_features
                    
                    selected_data['source_file'] = file_name
                    self.all_data.append(selected_data)
                    self.file_names.append(file_name)
                
            except Exception as e:
                print(f"✗ Error loading file {file_name}: {e}")
        
        if not self.all_data:
            raise ValueError("No data files successfully loaded!")
        
        self.combined_data = pd.concat(self.all_data, ignore_index=True)
        print(f"Total loaded {len(self.all_data)} files, combined shape: {self.combined_data.shape}")
        
        # Preprocessing
        data_for_analysis = self.combined_data[self.load_related_features].copy()
        data_cleaned = data_for_analysis.dropna()
        
        if len(data_cleaned) == 0:
            raise ValueError("No samples remaining after data cleaning!")
        
        print(f"Cleaned data shape: {data_cleaned.shape}")
        
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data_cleaned)
        self.original_data = data_cleaned.copy()
        
        elapsed_time = time.time() - start_time
        print(f"Data preprocessing completed, time: {elapsed_time:.2f}s")
        return True
    
    def find_optimal_clusters(self):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("\nFinding optimal number of clusters...")
        start_time = time.time()
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, self.max_clusters_to_test + 1)
        
        for k in k_range:
            print(f"  Testing K={k}...")
            kmeans = KMeans(n_clusters=k, n_init=self.n_init, random_state=42)
            labels = kmeans.fit_predict(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            
            # Use sampling for large datasets
            if len(self.scaled_data) > 5000:
                sample_size = min(5000, len(self.scaled_data))
                indices = np.random.choice(len(self.scaled_data), sample_size, replace=False)
                score = silhouette_score(self.scaled_data[indices], labels[indices])
            else:
                score = silhouette_score(self.scaled_data, labels)
            
            silhouette_scores.append(score)
        
        # Find best K value (highest silhouette score)
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_range[best_k_idx]
        best_score = silhouette_scores[best_k_idx]
        
        print(f"Optimal number of clusters: K={best_k}, Silhouette Score: {best_score:.4f}")
        
        # Create elbow method and silhouette score plots
        self.create_individual_cluster_selection_plots(k_range, inertias, silhouette_scores, best_k)
        
        elapsed_time = time.time() - start_time
        print(f"Optimal cluster search completed, time: {elapsed_time:.2f}s")
        
        return best_k
    
    def create_individual_cluster_selection_plots(self, k_range, inertias, silhouette_scores, best_k):
        """Create individual cluster selection plots"""
        
        # 1. Elbow Method Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-', linewidth=2.5, markersize=8)
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label=f'Optimal K={best_k}')
        plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12, fontweight='bold')
        plt.title('Elbow Method for Optimal Cluster Selection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_range)
        
        self.save_plot(plt, 'elbow_method_optimal_clusters.png')
        
        # 2. Silhouette Score Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'go-', linewidth=2.5, markersize=8)
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Optimal K={best_k}')
        plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        plt.title('Silhouette Score Analysis for Cluster Selection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_range)
        
        self.save_plot(plt, 'silhouette_score_optimal_clusters.png')
        
        # Save results
        selection_results = pd.DataFrame({
            'K': k_range,
            'Inertia': inertias,
            'Silhouette_Score': silhouette_scores
        })
        self.save_table(selection_results, 'cluster_selection_results.csv')
    
    def perform_pca_analysis(self):
        """Perform PCA analysis"""
        print("\nStarting PCA analysis...")
        start_time = time.time()
        
        self.pca = PCA(n_components=0.85, svd_solver='full')
        self.pca_result = self.pca.fit_transform(self.scaled_data)
        self.n_components = self.pca.n_components_
        
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        print(f"Retained first {self.n_components} principal components (explaining {cumulative_variance[-1]:.2%} variance)")
        
        # Create PCA visualizations
        self.create_individual_pca_plots(explained_variance, cumulative_variance)
        
        elapsed_time = time.time() - start_time
        print(f"PCA analysis completed, time: {elapsed_time:.2f}s")
        return True
    
    def create_individual_pca_plots(self, explained_variance, cumulative_variance):
        """Create individual PCA visualization plots"""
        
        # 1. PCA Variance Explained Plot
        plt.figure(figsize=(10, 6))
        components = range(1, len(explained_variance) + 1)
        
        bars = plt.bar(components, explained_variance, alpha=0.7, color='steelblue', 
                      label='Individual Component')
        plt.plot(components, cumulative_variance, 'ro-', linewidth=2.5, 
                markersize=8, label='Cumulative Variance')
        plt.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label='85% Threshold')
        
        plt.xlabel('Principal Components', fontsize=12, fontweight='bold')
        plt.ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        plt.title('PCA: Explained Variance by Principal Components', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(components)
        
        self.save_plot(plt, 'pca_variance_explained.png')
        
        # 2. PCA Scatter Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                            alpha=0.6, s=40, c='darkblue', edgecolor='white', linewidth=0.5)
        
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.title('PCA: First Two Principal Components', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'pca_scatter_plot.png')
        
        # 3. PCA Component Weights Heatmap
        plt.figure(figsize=(12, 8))
        feature_names = [name.replace('_', ' ').title() for name in self.load_related_features]
        pca_components = self.pca.components_[:4]  # First 4 components
        
        sns.heatmap(pca_components.T, 
                   xticklabels=[f'PC{i+1}' for i in range(pca_components.shape[0])],
                   yticklabels=feature_names,
                   annot=True, cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Component Weight'},
                   fmt='.3f', annot_kws={'size': 10})
        
        plt.title('PCA: Feature Weights in Principal Components', 
                 fontsize=14, fontweight='bold')
        plt.xticks(fontsize=11, fontweight='bold')
        plt.yticks(fontsize=11, fontweight='bold')
        
        self.save_plot(plt, 'pca_component_weights.png')
    
    def perform_kmeans_clustering(self, use_optimal_k=True):
        """Perform K-means clustering"""
        print("\nStarting K-means clustering...")
        start_time = time.time()
        
        # Determine number of clusters
        if use_optimal_k:
            self.n_clusters = self.find_optimal_clusters()
        else:
            print(f"Using preset number of clusters: K={self.n_clusters}")
        
        # K-means clustering in original space
        self.kmeans_original = KMeans(
            n_clusters=self.n_clusters, 
            n_init=self.n_init, 
            random_state=42
        )
        self.labels_original = self.kmeans_original.fit_predict(self.scaled_data)
        
        # K-means clustering in PCA space
        self.kmeans_pca = KMeans(
            n_clusters=self.n_clusters, 
            n_init=self.n_init, 
            random_state=42
        )
        self.labels_pca = self.kmeans_pca.fit_predict(self.pca_result)
        
        elapsed_time = time.time() - start_time
        print(f"K-means clustering completed, time: {elapsed_time:.2f}s")
        print(f"Final number of clusters: {self.n_clusters}")
        
        return True
    
    def evaluate_clustering_comprehensive(self):
        """Comprehensive clustering evaluation - test full p-value range"""
        print("\nStarting comprehensive clustering evaluation...")
        print(f"Testing full p-value range: {self.p_values}")
        start_time = time.time()
        
        evaluation_results = []
        
        # Test different spaces
        spaces = [
            ('Original Feature Space', self.scaled_data, self.labels_original),
            ('PCA Reduced Space', self.pca_result, self.labels_pca)
        ]
        
        for space_name, data, labels in spaces:
            print(f"  Evaluating {space_name}...")
            
            # Calculate traditional evaluation metrics
            silhouette_orig = silhouette_score(data, labels)
            calinski_orig = calinski_harabasz_score(data, labels)
            davies_orig = davies_bouldin_score(data, labels)
            
            # Calculate new evaluation metrics
            negative_silhouette_ratio = self.calculate_negative_silhouette_ratio(data, labels)
            cluster_metrics = self.calculate_cluster_quality_metrics(labels)
            
            # Calculate generalized silhouette scores for full p-value range
            generalized_silhouette_scores = {}
            for p in self.p_values:
                try:
                    print(f"    Calculating {self.p_value_names[p]}...")
                    gs_score = self.efficient_generalized_silhouette(data, labels, p)
                    generalized_silhouette_scores[p] = gs_score
                except Exception as e:
                    print(f"    Error calculating generalized silhouette for p={p}: {e}")
                    generalized_silhouette_scores[p] = np.nan
            
            # Save results
            result = {
                'Clustering_Space': space_name,
                'Traditional_Silhouette': silhouette_orig,
                'Calinski_Harabasz_Index': calinski_orig,
                'Davies_Bouldin_Index': davies_orig,
                'Negative_Silhouette_Ratio': negative_silhouette_ratio,
                'Small_Cluster_Ratio': cluster_metrics['small_cluster_ratio'],
                'Cluster_Balance_Ratio': cluster_metrics['cluster_balance_ratio'],
                'Normalized_Cluster_Size_Std': cluster_metrics['normalized_cluster_size_std'],
                'Number_of_Clusters': self.n_clusters
            }
            
            # Add generalized silhouette results
            for p, score in generalized_silhouette_scores.items():
                p_name = self.p_value_names[p]
                result[f'Generalized_Silhouette_{p_name}'] = score
            
            evaluation_results.append(result)
        
        # Convert to DataFrame
        evaluation_df = pd.DataFrame(evaluation_results)
        
        # Save evaluation results
        self.save_table(evaluation_df, 'comprehensive_clustering_evaluation.csv')
        
        # Create comparison charts
        self.create_individual_comparison_plots(evaluation_df)
        
        elapsed_time = time.time() - start_time
        print(f"Comprehensive evaluation completed, time: {elapsed_time:.2f}s")
        
        return evaluation_df
    
    def create_individual_comparison_plots(self, evaluation_df):
        """Create individual comparison plots"""
        
        # 1. Traditional Metrics Comparison
        self.create_traditional_metrics_comparison(evaluation_df)
        
        # 2. Generalized Silhouette Comparison
        gs_columns = [col for col in evaluation_df.columns if 'Generalized_Silhouette' in col]
        if len(gs_columns) > 0:
            self.create_generalized_silhouette_comparison(evaluation_df, gs_columns)
            self.create_p_value_trend_analysis(evaluation_df, gs_columns)
        
        # 3. Cluster Quality Metrics
        self.create_cluster_quality_metrics_plot(evaluation_df)
    
    def create_traditional_metrics_comparison(self, evaluation_df):
        """Create traditional metrics comparison plot"""
        plt.figure(figsize=(12, 8))
        
        metrics = ['Traditional_Silhouette', 'Calinski_Harabasz_Index', 'Davies_Bouldin_Index']
        metric_names = ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']
        spaces = evaluation_df['Clustering_Space'].values
        
        x = np.arange(len(spaces))
        width = 0.25
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = evaluation_df[metric].values
            offset = (i - 1) * width
            bars = plt.bar(x + offset, values, width, label=metric_name, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Clustering Space', fontsize=12, fontweight='bold')
        plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
        plt.title('Traditional Clustering Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, spaces)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        self.save_plot(plt, 'traditional_metrics_comparison.png')
    
    def create_generalized_silhouette_comparison(self, evaluation_df, gs_columns):
        """Create generalized silhouette comparison plots"""
        
        for i, space in enumerate(evaluation_df['Clustering_Space']):
            space_data = evaluation_df[evaluation_df['Clustering_Space'] == space].iloc[0]
            
            gs_scores = []
            gs_names = []
            for col in gs_columns:
                gs_scores.append(space_data[col])
                # Use the formatted names with line breaks
                gs_names.append(col.replace('Generalized_Silhouette_', ''))
            
            plt.figure(figsize=(14, 8))  # Increased width to accommodate multi-line labels
            bars = plt.bar(gs_names, gs_scores, alpha=0.7, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(gs_scores))))
            plt.axhline(y=space_data['Traditional_Silhouette'], color='red', linestyle='--', 
                       linewidth=2, label=f'Traditional Silhouette: {space_data["Traditional_Silhouette"]:.4f}')
            
            plt.xlabel('Generalized Mean Type', fontsize=12, fontweight='bold')
            plt.ylabel('Silhouette Score Value', fontsize=12, fontweight='bold')
            plt.title(f'{space} - Generalized Silhouette Comparison', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, gs_scores):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=9)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            filename = f'generalized_silhouette_{space.lower().replace(" ", "_")}.png'
            self.save_plot(plt, filename)
    
    def create_p_value_trend_analysis(self, evaluation_df, gs_columns):
        """Create p-value trend analysis plot"""
        # Extract p-value order
        p_order = [-np.inf, -2, -1, 0, 1, 2, np.inf]
        p_names = [self.p_value_names[p] for p in p_order]
        
        plt.figure(figsize=(14, 8))
        
        for i, space in enumerate(evaluation_df['Clustering_Space']):
            space_data = evaluation_df[evaluation_df['Clustering_Space'] == space].iloc[0]
            
            # Extract scores in p-value order
            scores = []
            for p in p_order:
                # Use the exact column name as stored in the DataFrame
                p_name = self.p_value_names[p]
                col_name = f'Generalized_Silhouette_{p_name}'
                # Check if the column exists in the DataFrame
                if col_name in space_data:
                    scores.append(space_data[col_name])
                else:
                    # If column doesn't exist, try to find a similar column
                    matching_cols = [col for col in gs_columns if p_name.replace('\n', ' ') in col.replace('Generalized_Silhouette_', '')]
                    if matching_cols:
                        scores.append(space_data[matching_cols[0]])
                    else:
                        print(f"Warning: Could not find column for {p_name}")
                        scores.append(np.nan)
            
            # Create trend line
            x_positions = range(len(p_names))
            plt.plot(x_positions, scores, 'o-', linewidth=2.5, markersize=8, 
                    label=space, alpha=0.8)
            
            # Mark each point
            for x, y, p_name in zip(x_positions, scores, p_names):
                if not np.isnan(y):
                    plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                                xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        
        plt.xlabel('Generalized Mean Type (p-value)', fontsize=12, fontweight='bold')
        plt.ylabel('Silhouette Score Value', fontsize=12, fontweight='bold')
        plt.title('Generalized Silhouette Score Trend Analysis', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(p_names)), p_names)
        
        # Add traditional silhouette reference lines
        traditional_scores = evaluation_df['Traditional_Silhouette'].values
        for i, (space, score) in enumerate(zip(evaluation_df['Clustering_Space'], traditional_scores)):
            plt.axhline(y=score, color=plt.cm.tab10(i), linestyle='--', alpha=0.5,
                       label=f'{space} Traditional: {score:.4f}')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        self.save_plot(plt, 'p_value_trend_analysis.png')
    
    def create_cluster_quality_metrics_plot(self, evaluation_df):
        """Create cluster quality metrics plot"""
        quality_metrics = ['Negative_Silhouette_Ratio', 'Small_Cluster_Ratio', 
                          'Cluster_Balance_Ratio', 'Normalized_Cluster_Size_Std']
        metric_names = ['Negative Silhouette Ratio', 'Small Cluster Ratio', 
                       'Cluster Balance Ratio', 'Normalized Cluster Size STD']
        
        spaces = evaluation_df['Clustering_Space'].values
        
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(spaces))
        width = 0.2
        
        for i, (metric, metric_name) in enumerate(zip(quality_metrics, metric_names)):
            values = evaluation_df[metric].values
            offset = (i - 1.5) * width
            bars = plt.bar(x + offset, values, width, label=metric_name, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Clustering Space', fontsize=12, fontweight='bold')
        plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
        plt.title('Cluster Quality Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, spaces)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        self.save_plot(plt, 'cluster_quality_metrics.png')
    
    def create_clustering_visualizations(self):
        """Create clustering visualizations"""
        print("\nCreating clustering visualizations...")
        start_time = time.time()
        
        # Create individual clustering visualization plots
        self.create_individual_clustering_plots()
        
        # Create cluster distribution plots
        self.create_cluster_distribution_plots()
        
        # Save clustering results
        self.save_clustering_results()
        
        elapsed_time = time.time() - start_time
        print(f"Visualization completed, time: {elapsed_time:.2f}s")
    
    def create_individual_clustering_plots(self):
        """Create individual clustering visualization plots"""
        
        # 1. Original Space Clustering
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.scaled_data[:, 0], self.scaled_data[:, 1], 
                            c=self.labels_original, cmap='viridis', alpha=0.7, s=40,
                            edgecolor='white', linewidth=0.5)
        plt.xlabel('Standardized Feature 1', fontsize=12, fontweight='bold')
        plt.ylabel('Standardized Feature 2', fontsize=12, fontweight='bold')
        plt.title(f'K-means Clustering in Original Feature Space\n(Silhouette Score: {silhouette_score(self.scaled_data, self.labels_original):.3f})', 
                 fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster Label')
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'clustering_original_space.png')
        
        # 2. PCA Space Clustering
        plt.figure(figsize=(10, 8))
        explained_variance = self.pca.explained_variance_ratio_
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                            c=self.labels_pca, cmap='viridis', alpha=0.7, s=40,
                            edgecolor='white', linewidth=0.5)
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.title(f'K-means Clustering in PCA Space\n(Silhouette Score: {silhouette_score(self.pca_result, self.labels_pca):.3f})', 
                 fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster Label')
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'clustering_pca_space.png')
    
    def create_cluster_distribution_plots(self):
        """Create cluster distribution plots"""
        
        # 1. Original Space Cluster Distribution
        plt.figure(figsize=(10, 6))
        unique_labels, counts = np.unique(self.labels_original, return_counts=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        bars = plt.bar(unique_labels, counts, alpha=0.7, color=colors, edgecolor='black')
        
        plt.xlabel('Cluster Label', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title('Cluster Distribution in Original Feature Space', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        self.save_plot(plt, 'cluster_distribution_original.png')
        
        # 2. PCA Space Cluster Distribution
        plt.figure(figsize=(10, 6))
        unique_labels, counts = np.unique(self.labels_pca, return_counts=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        bars = plt.bar(unique_labels, counts, alpha=0.7, color=colors, edgecolor='black')
        
        plt.xlabel('Cluster Label', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title('Cluster Distribution in PCA Space', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        self.save_plot(plt, 'cluster_distribution_pca.png')
    
    def save_clustering_results(self):
        """Save clustering results"""
        # Cluster statistics
        cluster_stats = []
        for cluster_id in np.unique(self.labels_pca):
            cluster_mask = self.labels_pca == cluster_id
            cluster_data = self.original_data.iloc[cluster_mask]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.original_data) * 100
            }
            
            for i, feature in enumerate(self.load_related_features):
                stats[f'{feature}_mean'] = cluster_data.iloc[:, i].mean()
                stats[f'{feature}_std'] = cluster_data.iloc[:, i].std()
            
            cluster_stats.append(stats)
        
        cluster_stats_df = pd.DataFrame(cluster_stats)
        self.save_table(cluster_stats_df, 'cluster_statistics.csv')
        
        # Complete clustering results
        results_df = self.original_data.copy()
        results_df['cluster_label_original'] = self.labels_original
        results_df['cluster_label_pca'] = self.labels_pca
        self.save_table(results_df, 'clustering_results.csv')
        
        # Save best p-value analysis
        self.save_best_p_analysis()
    
    def save_best_p_analysis(self):
        """Save best p-value analysis results"""
        # Read evaluation results
        eval_path = os.path.join(self.subfolders['tables'], 'comprehensive_clustering_evaluation.csv')
        eval_df = pd.read_csv(eval_path)
        
        # Find best p-value for each space
        best_p_results = []
        gs_columns = [col for col in eval_df.columns if 'Generalized_Silhouette' in col]
        
        for _, row in eval_df.iterrows():
            space = row['Clustering_Space']
            best_score = -1
            best_p_name = ""
            
            for col in gs_columns:
                score = row[col]
                if score > best_score:
                    best_score = score
                    best_p_name = col.replace('Generalized_Silhouette_', '')
            
            best_p_results.append({
                'Clustering_Space': space,
                'Best_Generalized_Silhouette_Type': best_p_name,
                'Best_Generalized_Silhouette_Score': best_score,
                'Traditional_Silhouette_Score': row['Traditional_Silhouette'],
                'Improvement_Magnitude': best_score - row['Traditional_Silhouette']
            })
        
        best_p_df = pd.DataFrame(best_p_results)
        self.save_table(best_p_df, 'best_p_value_analysis.csv')
        
        print("\nBest p-value analysis results:")
        print(best_p_df.to_string(index=False))
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        try:
            print("Starting Comprehensive K-means Clustering Analysis")
            print("=" * 60)
            total_start_time = time.time()
            
            # 1. Data loading and preprocessing
            self.load_and_preprocess_data()
            
            # 2. PCA analysis
            self.perform_pca_analysis()
            
            # 3. K-means clustering (automatically find optimal K)
            self.perform_kmeans_clustering(use_optimal_k=True)
            
            # 4. Comprehensive evaluation (test full p-value range)
            evaluation_results = self.evaluate_clustering_comprehensive()
            
            # 5. Visualization
            self.create_clustering_visualizations()
            
            total_elapsed_time = time.time() - total_start_time
            
            print("\n" + "=" * 60)
            print(f"Analysis Completed! Total time: {total_elapsed_time:.2f}s")
            print(f"All results saved to: {os.path.abspath(self.output_folder)}")
            
            # Display best results
            best_space_idx = evaluation_results['Traditional_Silhouette'].idxmax()
            best_space = evaluation_results.loc[best_space_idx, 'Clustering_Space']
            best_silhouette = evaluation_results.loc[best_space_idx, 'Traditional_Silhouette']
            
            print(f"\nBest clustering results:")
            print(f"- Space: {best_space}")
            print(f"- Silhouette Score: {best_silhouette:.4f}")
            print(f"- Negative Silhouette Ratio: {evaluation_results.loc[best_space_idx, 'Negative_Silhouette_Ratio']:.4f}")
            print(f"- Small Cluster Ratio: {evaluation_results.loc[best_space_idx, 'Small_Cluster_Ratio']:.4f}")
            print(f"- Number of Clusters: {self.n_clusters}")
            
            # Display generalized silhouette results summary
            print(f"\nGeneralized Silhouette Results Summary:")
            gs_columns = [col for col in evaluation_results.columns if 'Generalized_Silhouette' in col]
            for space in evaluation_results['Clustering_Space']:
                print(f"\n{space}:")
                space_data = evaluation_results[evaluation_results['Clustering_Space'] == space].iloc[0]
                for col in gs_columns:
                    p_name = col.replace('Generalized_Silhouette_', '')
                    score = space_data[col]
                    print(f"  {p_name}: {score:.4f}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

# Usage example
if __name__ == "__main__":
    folder_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\trip数据"
    output_folder = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\generalized_silhouette_analysis"
    
    # Use comprehensive K-means clustering analysis
    analyzer = ComprehensiveKMeansAnalyzer(
        folder_path, 
        output_folder, 
        n_clusters=2,  # Initial number of clusters, will be optimized
        n_init=10,     # K-means initialization count
        max_clusters_to_test=6  # Maximum number of clusters to test
    )
    
    analyzer.run_complete_analysis()