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
            'reports': os.path.join(self.output_folder, 'reports')
        }
        
        for folder_name, folder_path in self.subfolders.items():
            os.makedirs(folder_path, exist_ok=True)
        
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
                    print(f"✓ Successfully loaded: {file_name}")
                
            except Exception as e:
                print(f"✗ Error loading file {file_name}: {e}")
        
        if not self.all_data:
            raise ValueError("No data files successfully loaded!")
        
        self.combined_data = pd.concat(self.all_data, ignore_index=True)
        print(f"Total loaded {len(self.all_data)} files, combined shape: {self.combined_data.shape}")
        
        return True
    
    def preprocess_data(self):
        """Data preprocessing and cleaning"""
        print("\nStarting data preprocessing...")
        
        source_files = self.combined_data['source_file'].copy()
        data_for_analysis = self.combined_data[self.load_related_features].copy()
        
        # Remove rows with missing values
        original_shape = data_for_analysis.shape
        data_cleaned = data_for_analysis.dropna()
        print(f"Data shape after removing missing values: {data_cleaned.shape}")
        
        # Data standardization
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data_cleaned)
        
        # Save original data indices for later matching
        self.cleaned_indices = data_cleaned.index
        self.original_data = data_cleaned.copy()
        
        # Update combined_data with only complete rows
        self.combined_data_cleaned = pd.concat([
            pd.DataFrame(self.scaled_data, columns=data_cleaned.columns),
            source_files.iloc[self.cleaned_indices].reset_index(drop=True)
        ], axis=1)
        
        print("Data standardization completed")
        return True
    
    def perform_pca_analysis(self):
        """Perform Principal Component Analysis"""
        print("\nStarting PCA analysis...")
        
        pca_data = self.combined_data_cleaned[self.load_related_features].copy()
        
        self.pca = PCA()
        self.pca_features = self.pca.fit_transform(pca_data)
        
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        # Select components explaining 85% variance
        self.n_components = np.argmax(cumulative_variance >= 0.85) + 1
        if self.n_components < 2:
            self.n_components = 2
        
        self.pca_final = PCA(n_components=self.n_components)
        self.pca_result = self.pca_final.fit_transform(pca_data)
        
        print(f"Retained first {self.n_components} principal components")
        
        # Create PCA visualizations
        self.create_pca_visualizations(explained_variance, cumulative_variance)
        
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
    
    def perform_all_clustering_and_visualize(self):
        """Perform all clustering methods and create PCA visualizations"""
        print("\nPerforming clustering analysis with PCA visualization...")
        
        X = self.pca_result
        clustering_methods = {
            'KMeans': [
                ('KMeans (2 clusters)', KMeans(n_clusters=2, random_state=42, n_init=10)),
                ('KMeans (3 clusters)', KMeans(n_clusters=3, random_state=42, n_init=10)),
                ('KMeans (4 clusters)', KMeans(n_clusters=4, random_state=42, n_init=10))
            ],
            'DBSCAN': [
                ('DBSCAN (eps=0.5)', DBSCAN(eps=0.5, min_samples=5)),
                ('DBSCAN (eps=1.0)', DBSCAN(eps=1.0, min_samples=10)),
                ('DBSCAN (eps=1.5)', DBSCAN(eps=1.5, min_samples=15))
            ],
            'Hierarchical': [
                ('Hierarchical Ward (2 clusters)', AgglomerativeClustering(n_clusters=2, linkage='ward')),
                ('Hierarchical Ward (3 clusters)', AgglomerativeClustering(n_clusters=3, linkage='ward')),
                ('Hierarchical Ward (4 clusters)', AgglomerativeClustering(n_clusters=4, linkage='ward')),
                ('Hierarchical Complete (2 clusters)', AgglomerativeClustering(n_clusters=2, linkage='complete')),
                ('Hierarchical Complete (3 clusters)', AgglomerativeClustering(n_clusters=3, linkage='complete')),
                ('Hierarchical Complete (4 clusters)', AgglomerativeClustering(n_clusters=4, linkage='complete')),
                ('Hierarchical Average (2 clusters)', AgglomerativeClustering(n_clusters=2, linkage='average')),
                ('Hierarchical Average (3 clusters)', AgglomerativeClustering(n_clusters=3, linkage='average')),
                ('Hierarchical Average (4 clusters)', AgglomerativeClustering(n_clusters=4, linkage='average'))
            ],
            'Spectral': [
                ('Spectral (2 clusters)', SpectralClustering(n_clusters=2, random_state=42)),
                ('Spectral (3 clusters)', SpectralClustering(n_clusters=3, random_state=42)),
                ('Spectral (4 clusters)', SpectralClustering(n_clusters=4, random_state=42))
            ],
            'GMM': [
                ('GMM (2 components)', GaussianMixture(n_components=2, random_state=42)),
                ('GMM (3 components)', GaussianMixture(n_components=3, random_state=42)),
                ('GMM (4 components)', GaussianMixture(n_components=4, random_state=42))
            ],
            'BIRCH': [
                ('BIRCH (2 clusters)', Birch(n_clusters=2)),
                ('BIRCH (3 clusters)', Birch(n_clusters=3)),
                ('BIRCH (4 clusters)', Birch(n_clusters=4))
            ]
        }
        
        evaluation_results = []
        
        for method_group, methods in clustering_methods.items():
            print(f"\n{method_group}:")
            for method_name, clusterer in methods:
                try:
                    if hasattr(clusterer, 'fit_predict'):
                        labels = clusterer.fit_predict(X)
                    else:  # For GMM
                        labels = clusterer.fit(X).predict(X)
                    
                    # Calculate metrics
                    n_clusters = len(np.unique(labels))
                    if n_clusters > 1 and n_clusters < len(X):
                        silhouette = silhouette_score(X, labels)
                        calinski = calinski_harabasz_score(X, labels)
                        davies = davies_bouldin_score(X, labels)
                        
                        evaluation_results.append({
                            'Method': method_name,
                            'Silhouette_Score': silhouette,
                            'Calinski_Harabasz_Index': calinski,
                            'Davies_Bouldin_Index': davies,
                            'N_Clusters': n_clusters
                        })
                        
                        print(f"  {method_name}: Silhouette = {silhouette:.3f}, Clusters = {n_clusters}")
                        
                        # Create PCA visualization for this clustering
                        self.create_clustering_pca_plot(method_name, labels, silhouette)
                        
                except Exception as e:
                    print(f"  Error in {method_name}: {e}")
        
        # Save evaluation results
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df = evaluation_df.sort_values('Silhouette_Score', ascending=False)
        self.save_table(evaluation_df, 'clustering_evaluation_results.csv')
        
        # Create comparison plots
        self.create_clustering_comparison_plots(evaluation_df)
        
        return evaluation_df
    
    def create_clustering_pca_plot(self, method_name, labels, silhouette_score):
        """Create individual PCA clustering visualization"""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with cluster colors
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Use a color palette that works well for research
        if n_clusters <= 8:
            colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if len(mask) > 0:  # Check if cluster has points
                cluster_color = colors[i]
                plt.scatter(self.pca_result[mask, 0], self.pca_result[mask, 1],
                           c=[cluster_color], alpha=0.7, s=40, edgecolor='white',
                           linewidth=0.5, label=f'Cluster {label}')
        
        plt.xlabel('Principal Component 1', fontsize=12, fontweight='bold')
        plt.ylabel('Principal Component 2', fontsize=12, fontweight='bold')
        plt.title(f'{method_name}\nSilhouette Score: {silhouette_score:.3f}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Clean filename
        filename = f"pca_clustering_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        self.save_plot(plt, filename)
    
    def create_clustering_comparison_plots(self, evaluation_df):
        """Create comparison plots for clustering methods"""
        
        # 1. Silhouette Score Comparison
        plt.figure(figsize=(14, 10))
        methods = evaluation_df['Method']
        scores = evaluation_df['Silhouette_Score']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        bars = plt.barh(range(len(methods)), scores, color=colors, alpha=0.7, edgecolor='black')
        
        plt.yticks(range(len(methods)), methods, fontsize=9)
        plt.xlabel('Silhouette Score', fontsize=12, fontweight='bold')
        plt.title('Clustering Methods Comparison: Silhouette Scores', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        self.save_plot(plt, 'silhouette_scores_comparison.png')
        
        # 2. Calinski-Harabasz Index Comparison
        plt.figure(figsize=(14, 10))
        calinski_scores = evaluation_df['Calinski_Harabasz_Index']
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(methods)))
        bars = plt.barh(range(len(methods)), calinski_scores, color=colors, alpha=0.7, edgecolor='black')
        
        plt.yticks(range(len(methods)), methods, fontsize=9)
        plt.xlabel('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
        plt.title('Clustering Methods Comparison: Calinski-Harabasz Index', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(calinski_scores)*0.01, bar.get_y() + bar.get_height()/2., 
                    f'{width:.0f}', ha='left', va='center', fontsize=8)
        
        self.save_plot(plt, 'calinski_harabasz_comparison.png')
        
        # 3. Cluster Count Distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = evaluation_df['N_Clusters'].value_counts().sort_index()
        
        bars = plt.bar(cluster_counts.index, cluster_counts.values, 
                      color='steelblue', alpha=0.7, edgecolor='black')
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
    
    def analyze_best_clustering_results(self, evaluation_df):
        """Analyze the best clustering results"""
        print("\nAnalyzing best clustering results...")
        
        # Get best method
        best_method_name = evaluation_df.iloc[0]['Method']
        best_silhouette = evaluation_df.iloc[0]['Silhouette_Score']
        
        print(f"Best Method: {best_method_name}")
        print(f"Best Silhouette Score: {best_silhouette:.3f}")
        
        # Re-run best clustering to get labels
        X = self.pca_result
        best_clusterer = self.get_clusterer_by_name(best_method_name)
        
        if hasattr(best_clusterer, 'fit_predict'):
            best_labels = best_clusterer.fit_predict(X)
        else:
            best_labels = best_clusterer.fit(X).predict(X)
        
        # Add labels to data
        self.combined_data_cleaned['best_cluster'] = best_labels
        
        # Calculate cluster statistics
        cluster_stats = self.calculate_cluster_statistics(best_labels)
        
        # Create best cluster visualization
        self.create_best_cluster_visualization(best_method_name, best_labels, cluster_stats)
        
        return cluster_stats
    
    def get_clusterer_by_name(self, method_name):
        """Get clusterer instance by method name"""
        if 'KMeans' in method_name:
            n_clusters = int(method_name.split('(')[1].split()[0])
            return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif 'DBSCAN' in method_name:
            if 'eps=0.5' in method_name:
                return DBSCAN(eps=0.5, min_samples=5)
            elif 'eps=1.0' in method_name:
                return DBSCAN(eps=1.0, min_samples=10)
            else:
                return DBSCAN(eps=1.5, min_samples=15)
        elif 'Hierarchical' in method_name:
            if 'Ward' in method_name:
                n_clusters = int(method_name.split('(')[1].split()[0])
                return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            elif 'Complete' in method_name:
                n_clusters = int(method_name.split('(')[1].split()[0])
                return AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
            elif 'Average' in method_name:
                n_clusters = int(method_name.split('(')[1].split()[0])
                return AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        elif 'Spectral' in method_name:
            n_clusters = int(method_name.split('(')[1].split()[0])
            return SpectralClustering(n_clusters=n_clusters, random_state=42)
        elif 'GMM' in method_name:
            n_components = int(method_name.split('(')[1].split()[0])
            return GaussianMixture(n_components=n_components, random_state=42)
        elif 'BIRCH' in method_name:
            n_clusters = int(method_name.split('(')[1].split()[0])
            return Birch(n_clusters=n_clusters)
    
    def calculate_cluster_statistics(self, labels):
        """Calculate statistics for each cluster"""
        cluster_means = []
        for cluster_id in np.unique(labels):
            if cluster_id != -1:  # Exclude noise points
                cluster_mask = (labels == cluster_id)
                if np.sum(cluster_mask) > 0:  # Check if cluster has points
                    cluster_original = self.original_data.iloc[cluster_mask]
                    cluster_mean = cluster_original.mean().to_dict()
                    cluster_mean['cluster'] = cluster_id
                    cluster_mean['count'] = np.sum(cluster_mask)
                    cluster_mean['percentage'] = np.sum(cluster_mask) / len(labels)
                    cluster_means.append(cluster_mean)
        
        cluster_stats = pd.DataFrame(cluster_means)
        if not cluster_stats.empty:
            cluster_stats = cluster_stats.sort_values('avg_fuel_flow')
        
        print("\nCluster Statistics:")
        print(cluster_stats.round(2))
        
        self.save_table(cluster_stats, 'best_cluster_analysis.csv')
        
        return cluster_stats
    
    def create_best_cluster_visualization(self, method_name, labels, cluster_stats):
        """Create comprehensive visualization for best clustering"""
        
        # 1. Enhanced PCA with cluster coloring
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        
        for i, label in enumerate(unique_labels):
            if label != -1:  # Skip noise points
                mask = labels == label
                if np.sum(mask) > 0:  # Check if cluster has points
                    cluster_color = colors[i]
                    plt.scatter(self.pca_result[mask, 0], self.pca_result[mask, 1],
                               c=[cluster_color], alpha=0.7, s=50, edgecolor='white',
                               linewidth=0.8, label=f'Cluster {label}')
        
        plt.xlabel('Principal Component 1', fontsize=12, fontweight='bold')
        plt.ylabel('Principal Component 2', fontsize=12, fontweight='bold')
        plt.title(f'Best Clustering: {method_name}\nPCA Projection with Cluster Assignments', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'best_clustering_pca.png')
        
        # 2. Cluster Feature Analysis
        if not cluster_stats.empty:
            plt.figure(figsize=(14, 8))
            
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
            plt.ylabel('Feature Value (Original Scale)', fontsize=12, fontweight='bold')
            plt.title('Feature Distribution Across Clusters', fontsize=14, fontweight='bold')
            plt.xticks(x + width*(n_features-1)/2, cluster_stats['cluster'].astype(int))
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            self.save_plot(plt, 'cluster_feature_analysis.png')
        
        # 3. Cluster Size Distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        valid_clusters = cluster_counts[cluster_counts.index != -1]
        
        if len(valid_clusters) > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, len(valid_clusters)))
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
    
    def generate_final_report(self, evaluation_df, cluster_stats):
        """Generate final analysis report"""
        best_method = evaluation_df.iloc[0]['Method']
        best_score = evaluation_df.iloc[0]['Silhouette_Score']
        
        report_content = f"""
Advanced Clustering Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
==================================================

Data Overview:
- Analyzed Files: {len(self.file_names)}
- Total Samples: {len(self.combined_data_cleaned)}
- Features: {len(self.load_related_features)}

Best Clustering Method:
- Method: {best_method}
- Silhouette Score: {best_score:.3f}
- Number of Clusters: {len(cluster_stats) if not cluster_stats.empty else 0}

Cluster Analysis:
"""
        if not cluster_stats.empty:
            for _, row in cluster_stats.iterrows():
                load_level = "Low" if row['avg_fuel_flow'] < cluster_stats['avg_fuel_flow'].median() else "High"
                report_content += f"\nCluster {int(row['cluster'])} ({load_level} Load):"
                report_content += f"\n  Samples: {int(row['count'])} ({row['percentage']:.1%})"
                report_content += f"\n  Avg Fuel Flow: {row['avg_fuel_flow']:.2f}"
                report_content += f"\n  Avg Engine Speed: {row['avg_engine_speed']:.2f}"
                report_content += f"\n  Avg Friction Torque: {row['avg_friction_torque']:.2f}"
                report_content += f"\n  Avg Speed: {row['avg_speed']:.2f}\n"

        report_content += f"\nTop 5 Methods by Silhouette Score:\n"
        for i, (_, row) in enumerate(evaluation_df.head(5).iterrows()):
            report_content += f"{i+1}. {row['Method']}: {row['Silhouette_Score']:.3f}\n"

        # Save report
        report_path = os.path.join(self.subfolders['reports'], 'comprehensive_clustering_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nComprehensive report saved to: {report_path}")
        return report_path
    
    def run_advanced_analysis(self):
        """Execute complete advanced analysis pipeline"""
        try:
            print("Starting Advanced Clustering Analysis")
            print("=" * 60)
            
            self.load_pkl_files()
            self.preprocess_data()
            self.perform_pca_analysis()
            
            # Perform all clustering methods with visualization
            evaluation_df = self.perform_all_clustering_and_visualize()
            
            # Analyze best results
            cluster_stats = self.analyze_best_clustering_results(evaluation_df)
            
            # Generate final report
            report_path = self.generate_final_report(evaluation_df, cluster_stats)
            
            print("\n" + "=" * 60)
            print("Advanced Clustering Analysis Completed!")
            print(f"All results saved to: {os.path.abspath(self.output_folder)}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

# Usage example
if __name__ == "__main__":
    folder_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\trip数据"
    output_folder = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\clustering_analysis"
    
    analyzer = AdvancedLoadClusterAnalysis(folder_path, output_folder)
    analyzer.run_advanced_analysis()