import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (KMeans, DBSCAN, SpectralClustering, 
                           AgglomerativeClustering, Birch)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                           davies_bouldin_score)
import warnings
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')

# Set plotting style for research publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AdvancedLoadClusterAnalysis:
    """Advanced clustering analysis for load pattern identification"""
    
    def __init__(self, folder_path, output_folder=None):
        self.folder_path = folder_path
        
        # Create output directory
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"advanced_load_cluster_results_{timestamp}"
        else:
            self.output_folder = output_folder
        
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output directory created: {os.path.abspath(self.output_folder)}")
        
        # Create subdirectories
        self.subfolders = {
            'plots': os.path.join(self.output_folder, 'plots'),
            'tables': os.path.join(self.output_folder, 'tables'),
            'models': os.path.join(self.output_folder, 'models'),
            'reports': os.path.join(self.output_folder, 'reports')
        }
        
        for folder_name, folder_path in self.subfolders.items():
            os.makedirs(folder_path, exist_ok=True)
            print(f"  Subdirectory created: {folder_path}")
        
        self.all_data = []
        self.file_names = []
        
        # Define columns for load-related features
        self.selected_columns_mapping = {
            '发动机燃料流量平均值/Average fuel flow': 'avg_fuel_flow',
            '发动机转速平均值/Average engine speed': 'avg_engine_speed', 
            '摩擦扭矩平均值/Average friction torque': 'avg_friction_torque',
            '车速平均值/Average speed': 'avg_speed'
        }
        
        # Get English column names
        self.load_related_features = list(self.selected_columns_mapping.values())
        
        # Display complete output path structure
        print(f"\nOutput directory structure:")
        print(f"Main output directory: {os.path.abspath(self.output_folder)}")
        for folder_name, folder_path in self.subfolders.items():
            print(f"  {folder_name}: {os.path.abspath(folder_path)}")
    
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
        df.to_csv(filepath, index=True, encoding='utf-8-sig')
        print(f"Table saved: {filepath}")
        return filepath
    
    def load_pkl_files(self):
        """Load all pickle files from directory"""
        print("Loading pickle files...")
        print("=" * 50)
        
        # Get all pickle files
        pkl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        
        if not pkl_files:
            raise ValueError("No pickle files found in the specified directory!")
        
        print(f"Found {len(pkl_files)} pickle files")
        
        for file_name in pkl_files:
            file_path = os.path.join(self.folder_path, file_name)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, pd.DataFrame):
                    # Check if Chinese column names exist
                    chinese_columns = list(self.selected_columns_mapping.keys())
                    missing_cols = [col for col in chinese_columns if col not in data.columns]
                    
                    if missing_cols:
                        print(f"Warning: File {file_name} missing columns: {missing_cols}")
                        # Try using English column names
                        if all(col in data.columns for col in self.load_related_features):
                            print(f"  Using English column names as alternative")
                            selected_data = data[self.load_related_features].copy()
                        else:
                            continue
                    else:
                        # Use Chinese column names and rename to English
                        selected_data = data[chinese_columns].copy()
                        selected_data.columns = self.load_related_features
                    
                    # Add file source information
                    selected_data['source_file'] = file_name
                    
                    self.all_data.append(selected_data)
                    self.file_names.append(file_name)
                    print(f"✓ Successfully loaded: {file_name} (shape: {selected_data.shape})")
                
            except Exception as e:
                print(f"✗ Error loading file {file_name}: {e}")
        
        if not self.all_data:
            raise ValueError("No data files successfully loaded!")
        
        # Combine all data
        self.combined_data = pd.concat(self.all_data, ignore_index=True)
        print(f"\nTotal loaded {len(self.all_data)} files, combined shape: {self.combined_data.shape}")
        
        # Save file list
        file_list = pd.DataFrame({'file_name': self.file_names})
        file_path = self.save_table(file_list, 'loaded_files_list.csv', 'reports')
        print(f"File list saved to: {file_path}")
        
        return True
    
    def preprocess_data(self):
        """Data preprocessing and cleaning"""
        print("\nStarting data preprocessing...")
        print("=" * 50)
        
        # Separate source file information
        source_files = self.combined_data['source_file'].copy()
        data_for_analysis = self.combined_data[self.load_related_features].copy()
        
        # 1. Check for missing values
        missing_stats = data_for_analysis.isnull().sum()
        print("Missing values statistics:")
        for col, count in missing_stats.items():
            if count > 0:
                print(f"  {col}: {count} missing values ({count/len(data_for_analysis):.2%})")
        
        # Save missing values statistics
        missing_df = pd.DataFrame({
            'feature': missing_stats.index,
            'missing_count': missing_stats.values,
            'missing_percentage': (missing_stats.values / len(data_for_analysis)) * 100
        })
        missing_path = self.save_table(missing_df, 'missing_values_statistics.csv', 'tables')
        print(f"Missing values statistics saved to: {missing_path}")
        
        # Remove rows with missing values
        original_shape = data_for_analysis.shape
        data_cleaned = data_for_analysis.dropna()
        print(f"Data shape after removing missing values: {data_cleaned.shape} (removed {original_shape[0] - data_cleaned.shape[0]} rows)")
        
        # 2. Data standardization
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data_cleaned)
        self.scaled_df = pd.DataFrame(self.scaled_data, columns=data_cleaned.columns)
        
        # Save original data indices for later matching
        self.cleaned_indices = data_cleaned.index
        self.original_data = data_cleaned.copy()
        
        # Update combined_data with only complete rows and load-related features
        self.combined_data_cleaned = pd.concat([
            pd.DataFrame(self.scaled_data, columns=data_cleaned.columns),
            source_files.iloc[self.cleaned_indices].reset_index(drop=True)
        ], axis=1)
        
        print("Data standardization completed")
        
        # Save preprocessed data
        preprocessed_path = self.save_table(self.combined_data_cleaned, 'preprocessed_data.csv', 'tables')
        print(f"Preprocessed data saved to: {preprocessed_path}")
        
        return True
    
    def perform_pca_analysis(self):
        """Perform Principal Component Analysis"""
        print("\nStarting PCA analysis...")
        print("=" * 50)
        
        pca_data = self.combined_data_cleaned[self.load_related_features].copy()
        
        self.pca = PCA()
        self.pca_features = self.pca.fit_transform(pca_data)
        
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        print("Principal components explained variance ratio:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            print(f"PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)")
        
        # Select components explaining 85% variance
        self.n_components = np.argmax(cumulative_variance >= 0.85) + 1
        if self.n_components < 2:
            self.n_components = 2
        
        self.pca_final = PCA(n_components=self.n_components)
        self.pca_result = self.pca_final.fit_transform(pca_data)
        
        print(f"Retained first {self.n_components} principal components (explaining {cumulative_variance[self.n_components-1]:.2%} variance)")
        
        # Create PCA visualizations
        self.create_pca_visualizations(explained_variance, cumulative_variance)
        
        # Save PCA results
        pca_df = pd.DataFrame(self.pca_result, columns=[f'PC{i+1}' for i in range(self.n_components)])
        pca_df['source_file'] = self.combined_data_cleaned['source_file'].values
        pca_path = self.save_table(pca_df, 'pca_results.csv', 'tables')
        print(f"PCA results saved to: {pca_path}")
        
        return True
    
    def create_pca_visualizations(self, explained_variance, cumulative_variance):
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
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Principal Components', fontsize=12, fontweight='bold')
        plt.ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        plt.title('PCA: Explained Variance by Principal Components', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(components)
        
        self.save_plot(plt, 'pca_variance_explained.png')
        
        # 2. PCA Scatter Plot (First Two Components)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                            alpha=0.6, s=40, c='darkblue', edgecolor='white', linewidth=0.5)
        
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)', 
                  fontsize=12, fontweight='bold')
        plt.title('PCA: First Two Principal Components', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add density contours
        sns.kdeplot(x=self.pca_result[:, 0], y=self.pca_result[:, 1], 
                   levels=5, color='red', alpha=0.5, linewidths=1)
        
        self.save_plot(plt, 'pca_scatter_plot.png')
        
        # 3. PCA Component Weights Heatmap
        plt.figure(figsize=(12, 8))
        feature_names = [name.replace('_', ' ').title() for name in self.load_related_features]
        pca_components = self.pca_final.components_
        
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
    
    def find_optimal_eps(self, X, k=5):
        """Find optimal eps parameter for DBSCAN"""
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find elbow point
        gradients = np.gradient(distances)
        optimal_idx = np.argmax(gradients)
        optimal_eps = distances[optimal_idx]
        
        return optimal_eps, distances
    
    def try_all_clustering_methods(self):
        """Try multiple clustering methods and compare performance"""
        print("\nTrying multiple clustering methods...")
        print("=" * 60)
        
        results = {}
        X = self.pca_result  # Use PCA-reduced data
        
        # 1. KMeans series
        print("\n1. KMeans Series:")
        for n_clusters in [2, 3, 4]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                results[f'KMeans_{n_clusters}'] = {
                    'labels': labels, 'score': score, 'n_clusters': len(np.unique(labels))
                }
                print(f"  KMeans (k={n_clusters}): Silhouette Score = {score:.3f}")
        
        # 2. DBSCAN
        print("\n2. DBSCAN:")
        try:
            optimal_eps, distances = self.find_optimal_eps(X, k=5)
            print(f"  Recommended eps parameter: {optimal_eps:.3f}")
            for eps in [optimal_eps, optimal_eps*0.8, optimal_eps*1.2]:
                for min_samples in [5, 10, 15]:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X)
                    n_clusters = len(np.unique(labels[labels != -1]))
                    if n_clusters > 1:
                        valid_mask = labels != -1
                        if np.sum(valid_mask) > 10:  # Ensure sufficient samples
                            score = silhouette_score(X[valid_mask], labels[valid_mask])
                            results[f'DBSCAN_eps{eps:.2f}_min{min_samples}'] = {
                                'labels': labels, 'score': score, 'n_clusters': n_clusters,
                                'noise_ratio': np.mean(labels == -1)
                            }
                            print(f"  DBSCAN (eps={eps:.2f}, min_samples={min_samples}): Silhouette = {score:.3f}, Noise Ratio = {np.mean(labels == -1):.2%}")
        except Exception as e:
            print(f"  DBSCAN error: {e}")
        
        # 3. Hierarchical Clustering
        print("\n3. Hierarchical Clustering:")
        for n_clusters in [2, 3, 4]:
            for linkage_method in ['ward', 'complete', 'average']:
                agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                labels = agg.fit_predict(X)
                score = silhouette_score(X, labels)
                results[f'Agglomerative_{linkage_method}_{n_clusters}'] = {
                    'labels': labels, 'score': score, 'n_clusters': n_clusters
                }
                print(f"  Hierarchical ({linkage_method}, k={n_clusters}): Silhouette = {score:.3f}")
        
        # 4. Spectral Clustering
        print("\n4. Spectral Clustering:")
        for n_clusters in [2, 3, 4]:
            spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
            labels = spectral.fit_predict(X)
            score = silhouette_score(X, labels)
            results[f'Spectral_{n_clusters}'] = {
                'labels': labels, 'score': score, 'n_clusters': n_clusters
            }
            print(f"  Spectral (k={n_clusters}): Silhouette = {score:.3f}")
        
        # 5. Gaussian Mixture Models
        print("\n5. Gaussian Mixture Models:")
        for n_components in [2, 3, 4]:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            labels = gmm.fit_predict(X)
            score = silhouette_score(X, labels)
            results[f'GMM_{n_components}'] = {
                'labels': labels, 'score': score, 'n_clusters': n_components
            }
            print(f"  GMM (k={n_components}): Silhouette = {score:.3f}")
        
        # 6. BIRCH
        print("\n6. BIRCH:")
        for n_clusters in [2, 3, 4]:
            birch = Birch(n_clusters=n_clusters)
            labels = birch.fit_predict(X)
            score = silhouette_score(X, labels)
            results[f'BIRCH_{n_clusters}'] = {
                'labels': labels, 'score': score, 'n_clusters': n_clusters
            }
            print(f"  BIRCH (k={n_clusters}): Silhouette = {score:.3f}")
        
        self.clustering_results = results
        return results
    
    def evaluate_clustering_results(self):
        """Evaluate all clustering results using multiple metrics"""
        print("\nEvaluating clustering results...")
        print("=" * 60)
        
        evaluation_results = []
        
        for method_name, result in self.clustering_results.items():
            labels = result['labels']
            
            # Skip if only one cluster
            if len(np.unique(labels)) <= 1:
                continue
            
            # Calculate various evaluation metrics
            silhouette = silhouette_score(self.pca_result, labels)
            calinski = calinski_harabasz_score(self.pca_result, labels)
            davies = davies_bouldin_score(self.pca_result, labels)
            
            # Statistical cluster information
            n_clusters = result['n_clusters']
            cluster_counts = pd.Series(labels).value_counts()
            
            evaluation_results.append({
                'Method': method_name,
                'Silhouette_Score': silhouette,
                'Calinski_Harabasz_Index': calinski,
                'Davies_Bouldin_Index': davies,
                'N_Clusters': n_clusters,
                'Cluster_Distribution': cluster_counts.to_dict()
            })
            
            print(f"{method_name:30} | Silhouette: {silhouette:.3f} | Calinski: {calinski:6.1f} | Davies: {davies:.3f} | Clusters: {n_clusters}")
        
        # Sort by Silhouette Score
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df = evaluation_df.sort_values('Silhouette_Score', ascending=False)
        
        evaluation_path = self.save_table(evaluation_df, 'clustering_evaluation_results.csv', 'tables')
        print(f"Clustering evaluation results saved to: {evaluation_path}")
        
        # Select best result
        best_method = evaluation_df.iloc[0]['Method']
        best_labels = self.clustering_results[best_method]['labels']
        
        print(f"\nBest Method: {best_method}")
        print(f"Best Silhouette Score: {evaluation_df.iloc[0]['Silhouette_Score']:.3f}")
        
        return best_method, best_labels, evaluation_df
    
    def create_clustering_comparison_plots(self, evaluation_df):
        """Create individual research-quality plots for clustering comparison"""
        
        # 1. Silhouette Score Comparison
        plt.figure(figsize=(12, 8))
        methods = evaluation_df['Method']
        scores = evaluation_df['Silhouette_Score']
        
        bars = plt.barh(range(len(methods)), scores, color='steelblue', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(methods)), methods, fontsize=10)
        plt.xlabel('Silhouette Score', fontsize=12, fontweight='bold')
        plt.title('Clustering Methods Comparison: Silhouette Scores', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        self.save_plot(plt, 'silhouette_scores_comparison.png')
        
        # 2. Calinski-Harabasz Index Comparison
        plt.figure(figsize=(12, 8))
        calinski_scores = evaluation_df['Calinski_Harabasz_Index']
        
        bars = plt.barh(range(len(methods)), calinski_scores, color='lightcoral', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(methods)), methods, fontsize=10)
        plt.xlabel('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
        plt.title('Clustering Methods Comparison: Calinski-Harabasz Index', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(calinski_scores)*0.01, bar.get_y() + bar.get_height()/2., 
                    f'{width:.0f}', ha='left', va='center', fontsize=9)
        
        self.save_plot(plt, 'calinski_harabasz_comparison.png')
        
        # 3. Davies-Bouldin Index Comparison (lower is better)
        plt.figure(figsize=(12, 8))
        davies_scores = evaluation_df['Davies_Bouldin_Index']
        
        bars = plt.barh(range(len(methods)), davies_scores, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(methods)), methods, fontsize=10)
        plt.xlabel('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
        plt.title('Clustering Methods Comparison: Davies-Bouldin Index', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        self.save_plot(plt, 'davies_bouldin_comparison.png')
        
        # 4. Cluster Count Distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = evaluation_df['N_Clusters'].value_counts().sort_index()
        
        bars = plt.bar(cluster_counts.index, cluster_counts.values, 
                      color='purple', alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Clusters', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Methods', fontsize=12, fontweight='bold')
        plt.title('Distribution of Cluster Counts Across Methods', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        self.save_plot(plt, 'cluster_count_distribution.png')
    
    def analyze_best_clusters(self, best_labels):
        """Analyze the best clustering results"""
        print("\nAnalyzing best clustering results...")
        print("=" * 60)
        
        # Add labels to data
        self.combined_data_cleaned['best_cluster'] = best_labels
        
        # Calculate original feature means for each cluster
        cluster_means = []
        for cluster_id in np.unique(best_labels):
            if cluster_id != -1:  # Exclude noise points
                cluster_mask = (best_labels == cluster_id)
                cluster_original = self.original_data.iloc[cluster_mask]
                cluster_mean = cluster_original.mean().to_dict()
                cluster_mean['cluster'] = cluster_id
                cluster_mean['count'] = np.sum(cluster_mask)
                cluster_mean['percentage'] = np.sum(cluster_mask) / len(best_labels)
                cluster_means.append(cluster_mean)
        
        cluster_stats = pd.DataFrame(cluster_means)
        cluster_stats = cluster_stats.sort_values('avg_fuel_flow')  # Sort by fuel flow
        
        print("Typical feature values for each load level:")
        print(cluster_stats.round(2))
        
        # Create cluster visualization
        self.create_cluster_visualization(cluster_stats, best_labels)
        
        # Save results
        cluster_path = self.save_table(cluster_stats, 'best_cluster_analysis.csv', 'tables')
        final_data_path = self.save_table(self.combined_data_cleaned, 'final_clustered_data.csv', 'tables')
        
        print(f"Cluster analysis results saved to: {cluster_path}")
        print(f"Final clustered data saved to: {final_data_path}")
        
        return cluster_stats
    
    def create_cluster_visualization(self, cluster_stats, best_labels):
        """Create individual research-quality cluster visualization plots"""
        
        # 1. Cluster Distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = pd.Series(best_labels).value_counts().sort_index()
        valid_clusters = cluster_counts[cluster_counts.index != -1]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(valid_clusters)))
        bars = plt.bar(valid_clusters.index.astype(str), valid_clusters.values, 
                      color=colors, alpha=0.7, edgecolor='black')
        
        plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(valid_clusters.values)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        self.save_plot(plt, 'cluster_size_distribution.png')
        
        # 2. Feature Means by Cluster
        plt.figure(figsize=(12, 8))
        features_to_plot = ['avg_fuel_flow', 'avg_engine_speed', 'avg_friction_torque', 'avg_speed']
        feature_names = [name.replace('_', ' ').title() for name in features_to_plot]
        
        n_features = len(features_to_plot)
        n_clusters = len(cluster_stats)
        
        x = np.arange(n_clusters)
        width = 0.8 / n_features
        
        for i, feature in enumerate(features_to_plot):
            values = cluster_stats[feature].values
            plt.bar(x + i*width, values, width, label=feature_names[i], alpha=0.7)
        
        plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
        plt.ylabel('Normalized Feature Value', fontsize=12, fontweight='bold')
        plt.title('Feature Means by Cluster', fontsize=14, fontweight='bold')
        plt.xticks(x + width*(n_features-1)/2, cluster_stats['cluster'].astype(int))
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'feature_means_by_cluster.png')
        
        # 3. PCA Projection with Cluster Coloring
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                            c=best_labels, cmap='Set3', alpha=0.7, s=40, 
                            edgecolor='white', linewidth=0.5)
        
        plt.xlabel('Principal Component 1', fontsize=12, fontweight='bold')
        plt.ylabel('Principal Component 2', fontsize=12, fontweight='bold')
        plt.title('PCA Projection with Cluster Assignments', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster ID')
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'pca_cluster_projection.png')
    
    def generate_comprehensive_report(self, best_method, evaluation_df, cluster_stats):
        """Generate comprehensive analysis report"""
        report_content = f"""
Advanced Clustering Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
==================================================

Data Overview:
- Analyzed Files: {len(self.file_names)}
- Total Samples: {len(self.combined_data_cleaned)}
- Load-Related Features: {len(self.load_related_features)}

PCA Analysis Results:
- Retained Components: {self.n_components}
- Cumulative Explained Variance: {self.pca_final.explained_variance_ratio_.sum():.2%}

Best Clustering Method:
- Method: {best_method}
- Silhouette Score: {evaluation_df[evaluation_df['Method'] == best_method]['Silhouette_Score'].values[0]:.3f}
- Number of Clusters: {len(cluster_stats)}

Load Level Analysis:
"""
        for _, row in cluster_stats.iterrows():
            report_content += f"\nLevel {int(row['cluster'])}: "
            if row['avg_fuel_flow'] < cluster_stats['avg_fuel_flow'].quantile(0.33):
                report_content += "Low Load"
            elif row['avg_fuel_flow'] < cluster_stats['avg_fuel_flow'].quantile(0.67):
                report_content += "Medium Load"
            else:
                report_content += "High Load"
            report_content += f" (Samples: {int(row['count'])}, Percentage: {row['percentage']:.2%})"
            report_content += f"\n  - Fuel Flow: {row['avg_fuel_flow']:.2f}"
            report_content += f"\n  - Engine Speed: {row['avg_engine_speed']:.2f}"
            report_content += f"\n  - Friction Torque: {row['avg_friction_torque']:.2f}"
            report_content += f"\n  - Vehicle Speed: {row['avg_speed']:.2f}\n"

        report_content += f"\nMethod Ranking (by Silhouette Score):\n"
        for i, (_, row) in enumerate(evaluation_df.iterrows()):
            report_content += f"{i+1}. {row['Method']}: {row['Silhouette_Score']:.3f}\n"

        # Save report
        report_path = os.path.join(self.subfolders['reports'], 'comprehensive_clustering_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Comprehensive report saved to: {report_path}")
        
        return report_path
    
    def run_advanced_analysis(self):
        """Execute complete advanced analysis pipeline"""
        try:
            print("Starting Advanced Clustering Analysis")
            print("=" * 60)
            print(f"Input data path: {os.path.abspath(self.folder_path)}")
            print(f"Output directory: {os.path.abspath(self.output_folder)}")
            print("=" * 60)
            
            self.load_pkl_files()
            self.preprocess_data()
            self.perform_pca_analysis()
            
            # Try all clustering methods
            self.try_all_clustering_methods()
            
            # Evaluate and select best method
            best_method, best_labels, evaluation_df = self.evaluate_clustering_results()
            
            # Create comparison visualizations
            self.create_clustering_comparison_plots(evaluation_df)
            
            # Analyze best clustering results
            cluster_stats = self.analyze_best_clusters(best_labels)
            
            # Generate report
            report_path = self.generate_comprehensive_report(best_method, evaluation_df, cluster_stats)
            
            print("\n" + "=" * 60)
            print("Advanced Clustering Analysis Completed!")
            print(f"All results saved to: {os.path.abspath(self.output_folder)}")
            print(f"Final report location: {report_path}")
            print("=" * 60)
            
            # Display final output summary
            print("\n📁 Output Files Summary:")
            print(f"📊 Plot Files: {os.path.abspath(self.subfolders['plots'])}")
            print(f"📋 Data Tables: {os.path.abspath(self.subfolders['tables'])}")
            print(f"📝 Analysis Reports: {os.path.abspath(self.subfolders['reports'])}")
            print(f"🤖 Model Files: {os.path.abspath(self.subfolders['models'])}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

# Usage example
if __name__ == "__main__":
    # Set your pickle file directory path
    folder_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\trip数据"
    
    # Optional: specify output directory name
    output_folder = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\聚类尝试（秋季）\multiple_methods_clustering_attempt"
    
    # Create analyzer instance and run analysis
    analyzer = AdvancedLoadClusterAnalysis(folder_path, output_folder)
    analyzer.run_advanced_analysis()