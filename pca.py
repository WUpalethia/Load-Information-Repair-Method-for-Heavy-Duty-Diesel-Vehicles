import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Set professional fonts and styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class LoadClusterAnalysis:
    def __init__(self, folder_path, output_folder=None, sample_fraction=0.1):
        self.folder_path = folder_path
        self.sample_fraction = sample_fraction  # Sampling ratio
        
        # Create output folder with timestamp
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"load_cluster_results_{timestamp}"
        else:
            self.output_folder = output_folder
        
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"🎯 Results will be saved to: {os.path.abspath(self.output_folder)}")
        
        # Create organized subfolders
        self.subfolders = {
            'plots': os.path.join(self.output_folder, 'visualizations'),
            'tables': os.path.join(self.output_folder, 'data_tables'),
            'models': os.path.join(self.output_folder, 'models'),
            'reports': os.path.join(self.output_folder, 'analysis_reports'),
            'comparison': os.path.join(self.output_folder, 'method_comparison')
        }
        
        for folder_name, folder_path in self.subfolders.items():
            os.makedirs(folder_path, exist_ok=True)
        
        self.all_data = []
        self.file_names = []
        
        # Define feature mapping with human-readable names
        self.selected_columns_mapping = {
            '发动机燃料流量平均值/Average fuel flow': 'Fuel Consumption Rate',
            '发动机转速平均值/Average engine speed': 'Engine Speed', 
            '摩擦扭矩平均值/Average friction torque': 'Friction Torque',
            '车速平均值/Average speed': 'Vehicle Speed'
        }
        
        # Get English feature names
        self.load_related_features = list(self.selected_columns_mapping.values())
        
        # Store comparison results
        self.comparison_results = {}
    
    def save_plot(self, plt, filename, subfolder='plots'):
        """Save high-quality plots for publication"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', 
                   transparent=False, edgecolor='none')
        print(f"📊 Chart saved: {filepath}")
        plt.close()
    
    def save_table(self, df, filename, subfolder='tables'):
        """Save data tables"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        df.to_csv(filepath, index=True, encoding='utf-8-sig')
        print(f"📋 Table saved: {filepath}")
    
    def save_model(self, model, filename, subfolder='models'):
        """Save trained models"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"🤖 Model saved: {filepath}")
    
    def load_pkl_files(self):
        """Load all pkl files with progress tracking"""
        print("🚀 Starting data loading process...")
        print("=" * 60)
        
        # Find all pkl files
        pkl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        
        if not pkl_files:
            raise ValueError("❌ No pkl files found in the specified folder!")
        
        print(f"🔍 Found {len(pkl_files)} data files to process")
        
        for file_name in pkl_files:
            file_path = os.path.join(self.folder_path, file_name)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, pd.DataFrame):
                    # Check for Chinese column names
                    chinese_columns = list(self.selected_columns_mapping.keys())
                    missing_cols = [col for col in chinese_columns if col not in data.columns]
                    
                    if missing_cols:
                        print(f"⚠️  File {file_name} missing columns: {missing_cols}")
                        # Try English column names
                        if all(col in data.columns for col in self.load_related_features):
                            print(f"   Using English column names instead")
                            selected_data = data[self.load_related_features].copy()
                        else:
                            continue
                    else:
                        # Use Chinese columns and rename to human-readable English
                        selected_data = data[chinese_columns].copy()
                        selected_data.columns = self.load_related_features
                    
                    # Add file source information
                    selected_data['source_file'] = file_name
                    
                    self.all_data.append(selected_data)
                    self.file_names.append(file_name)
                    print(f"✅ Successfully loaded: {file_name} (shape: {selected_data.shape})")
                
            except Exception as e:
                print(f"❌ Error loading {file_name}: {e}")
        
        if not self.all_data:
            raise ValueError("❌ No data files were successfully loaded!")
        
        # Combine all data
        self.combined_data = pd.concat(self.all_data, ignore_index=True)
        print(f"\n🎉 Successfully loaded {len(self.all_data)} files")
        print(f"📊 Combined dataset shape: {self.combined_data.shape}")
        
        # Save file list
        file_list = pd.DataFrame({'file_name': self.file_names})
        self.save_table(file_list, 'loaded_files_list.csv', 'reports')
        
        return True
    
    def preprocess_data(self):
        """Data preprocessing with quality checks"""
        print("\n🔧 Starting data preprocessing...")
        print("=" * 60)
        
        # Separate source file information
        source_files = self.combined_data['source_file'].copy()
        data_for_analysis = self.combined_data[self.load_related_features].copy()
        
        # 1. Check for missing values
        missing_stats = data_for_analysis.isnull().sum()
        print("Missing values analysis:")
        for col, count in missing_stats.items():
            if count > 0:
                print(f"  {col}: {count} missing values ({count/len(data_for_analysis):.2%})")
        
        # Save missing value statistics
        missing_df = pd.DataFrame({
            'Feature': missing_stats.index,
            'Missing Count': missing_stats.values,
            'Missing Percentage': (missing_stats.values / len(data_for_analysis)) * 100
        })
        self.save_table(missing_df, 'missing_values_statistics.csv', 'tables')
        
        # Remove rows with missing values
        original_shape = data_for_analysis.shape
        data_cleaned = data_for_analysis.dropna()
        removed_rows = original_shape[0] - data_cleaned.shape[0]
        print(f"🧹 Data cleaning complete: {removed_rows} rows removed")
        print(f"📈 Clean dataset shape: {data_cleaned.shape}")
        
        # 2. Data standardization
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data_cleaned)
        self.scaled_df = pd.DataFrame(self.scaled_data, columns=data_cleaned.columns)
        
        # Save indices for later matching
        self.cleaned_indices = data_cleaned.index
        
        # Update combined data with cleaned version
        self.combined_data_cleaned = pd.concat([
            pd.DataFrame(self.scaled_data, columns=data_cleaned.columns),
            source_files.iloc[self.cleaned_indices].reset_index(drop=True)
        ], axis=1)
        
        # Save original numerical data
        self.original_data = data_cleaned.copy()
        
        print("✅ Data standardization completed")
        
        # Save preprocessed data
        self.save_table(self.combined_data_cleaned, 'preprocessed_data.csv', 'tables')
        
        return True
    
    def perform_pca(self):
        """Perform PCA analysis with visualization"""
        print("\n🔮 Starting PCA analysis...")
        print("=" * 60)
        
        # Get data for PCA
        pca_data = self.combined_data_cleaned[self.load_related_features].copy()
        
        # Perform PCA
        self.pca = PCA()
        self.pca_features = self.pca.fit_transform(pca_data)
        
        # Calculate explained variance
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        print("Principal Components Explained Variance:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            print(f"PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)")
        
        # Create elegant scree plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(1, len(explained_variance) + 1), explained_variance, 
                      alpha=0.7, color='steelblue', label='Individual Variance')
        line = plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                       'ro-', label='Cumulative Variance', linewidth=2, markersize=6)
        plt.axhline(y=0.85, color='forestgreen', linestyle='--', alpha=0.7, 
                   label='85% Variance Threshold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{explained_variance[i]:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Principal Component Number', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        plt.title('PCA Scree Plot: Load Feature Analysis', fontsize=14, pad=20)
        plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        self.save_plot(plt, 'pca_scree_plot.png')
        
        # Select number of components (retain >85% variance)
        self.n_components = np.argmax(cumulative_variance >= 0.85) + 1
        if self.n_components < 2:
            self.n_components = 2
        elif self.n_components > len(pca_data.columns):
            self.n_components = len(pca_data.columns)
            
        print(f"\n💡 Recommended: Keep first {self.n_components} principal components")
        print(f"📊 This explains {cumulative_variance[self.n_components-1]:.2%} of total variance")
        
        # Final PCA with selected components
        self.pca_final = PCA(n_components=self.n_components)
        self.pca_result = self.pca_final.fit_transform(pca_data)
        
        # Save PCA results
        pca_df = pd.DataFrame(self.pca_result, columns=[f'PC{i+1}' for i in range(self.n_components)])
        pca_df['source_file'] = self.combined_data_cleaned['source_file'].values
        self.save_table(pca_df, 'pca_results.csv', 'tables')
        
        # Save PCA model
        self.save_model(self.pca_final, 'pca_model.pkl', 'models')
        
        return True
    
    def analyze_pca_components(self):
        """Analyze principal component meanings"""
        print("\n📐 Principal Component Loadings Analysis:")
        print("=" * 60)
        
        # Create loadings matrix
        loadings = pd.DataFrame(
            self.pca_final.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.load_related_features
        )
        
        print("Component Loadings Matrix:")
        print(loadings.round(4))
        
        # Save loadings matrix
        self.save_table(loadings, 'pca_loadings_matrix.csv', 'tables')
        
        # Create beautiful loadings heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, fmt='.3f', 
                   annot_kws={'size': 10}, cbar_kws={'shrink': 0.8},
                   square=True, linewidths=0.5)
        plt.title('PCA Loadings Heatmap: Feature Contributions', fontsize=14, pad=20)
        plt.xticks(fontsize=10, rotation=0)
        plt.yticks(fontsize=10)
        self.save_plot(plt, 'pca_loadings_heatmap.png')
        
        # Interpret each principal component
        component_interpretations = []
        print("\n🔍 Principal Component Interpretations:")
        for i in range(self.n_components):
            print(f"\nPC{i+1}:")
            pc_loadings = loadings[f'PC{i+1}'].sort_values(key=abs, ascending=False)
            interpretation = []
            
            for feature, loading in pc_loadings.items():
                if abs(loading) > 0.3:
                    direction = "positive" if loading > 0 else "negative"
                    print(f"  {feature}: {loading:.3f} ({direction} influence)")
                    interpretation.append(f"{feature}({direction})")
            
            component_interpretations.append(", ".join(interpretation))
        
        # Save component interpretations
        interpretation_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(self.n_components)],
            'Interpretation': component_interpretations,
            'Variance Explained': self.pca_final.explained_variance_ratio_
        })
        self.save_table(interpretation_df, 'component_interpretations.csv', 'reports')
        
        return True
    
    def perform_load_clustering(self):
        """Perform load level clustering"""
        print("\n🎯 Starting load level clustering...")
        print("=" * 60)
        
        # Use silhouette score to determine optimal clusters
        silhouette_scores = []
        k_range = range(2, 6)
        
        print("Calculating silhouette scores for different cluster counts:")
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.pca_result)
                score = silhouette_score(self.pca_result, labels)
                silhouette_scores.append(score)
                print(f"k={k}: Silhouette Score = {score:.3f}")
            except Exception as e:
                print(f"k={k} error: {e}")
                silhouette_scores.append(-1)
        
        if not silhouette_scores or max(silhouette_scores) < 0:
            print("Using default k=3 due to invalid scores")
            best_k = 3
        else:
            best_k = k_range[np.argmax(silhouette_scores)]
        
        # Create silhouette analysis plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8, 
                markerfacecolor='red', markeredgecolor='darkred')
        plt.axvline(x=best_k, color='crimson', linestyle='--', alpha=0.8, 
                   label=f'Optimal k = {best_k}')
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Silhouette Analysis: Determining Optimal Clusters', fontsize=14)
        plt.legend(fontsize=11, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        self.save_plot(plt, 'silhouette_analysis.png')
        
        print(f"🎯 Optimal number of load levels: {best_k}")
        
        # Perform clustering with optimal k
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.pca_result)
        
        # Add cluster labels to data
        self.combined_data_cleaned['load_cluster'] = self.labels
        
        print(f"📊 Load level distribution:")
        cluster_counts = self.combined_data_cleaned['load_cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.combined_data_cleaned)) * 100
            print(f"  Level {cluster_id}: {count} samples ({percentage:.1f}%)")
        
        # Calculate cluster centers in original feature space
        print("\n📈 Calculating typical feature values for each load level...")
        cluster_means = []
        for cluster_id in range(best_k):
            cluster_mask = (self.labels == cluster_id)
            cluster_original_data = self.original_data.iloc[cluster_mask]
            cluster_mean = cluster_original_data.mean().values
            cluster_means.append(cluster_mean)
        
        load_centers_df = pd.DataFrame(cluster_means, columns=self.load_related_features)
        load_centers_df['Cluster'] = range(best_k)
        
        print("Typical feature values for each load level:")
        print(load_centers_df.round(2))
        
        # Save all results
        self.combined_data_cleaned.to_csv(
            os.path.join(self.subfolders['tables'], 'clustered_data.csv'), 
            index=False, encoding='utf-8-sig'
        )
        self.save_model(self.kmeans, 'kmeans_model.pkl', 'models')
        
        # Save cluster statistics
        cluster_stats = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Sample Count': cluster_counts.values,
            'Percentage': (cluster_counts.values / len(self.combined_data_cleaned)) * 100
        })
        self.save_table(cluster_stats, 'cluster_statistics.csv', 'tables')
        self.save_table(load_centers_df, 'cluster_centers.csv', 'tables')
        
        return True

    def compare_clustering_methods(self):
        """Compare clustering with PCA vs without PCA using multiple metrics - only for k=2"""
        print("\n🔬 Comparing Clustering Methods: PCA vs No-PCA (k=2 only)")
        print("=" * 60)
        
        # Get data for clustering
        pca_data = self.pca_result
        original_data = self.scaled_df.values
        
        # Sample data for faster comparison (1/10 of data)
        n_samples = int(len(pca_data) * self.sample_fraction)
        print(f"📊 Using {n_samples} samples ({self.sample_fraction*100}%) for method comparison")
        
        # Create random indices for sampling
        np.random.seed(42)
        sample_indices = np.random.choice(len(pca_data), n_samples, replace=False)
        
        pca_data_sampled = pca_data[sample_indices]
        original_data_sampled = original_data[sample_indices]
        
        # Only test k=2
        k = 2
        comparison_results = []
        
        print("Evaluating clustering performance for k=2:")
        print(f"{'Method':<10} {'Silhouette':<12} {'Calinski-Harabasz':<18} {'Davies-Bouldin':<15}")
        print("-" * 60)
        
        # Clustering with PCA
        kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_pca = kmeans_pca.fit_predict(pca_data_sampled)
        
        # Clustering without PCA
        kmeans_original = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_original = kmeans_original.fit_predict(original_data_sampled)
        
        # Calculate metrics for PCA clustering
        sil_pca = silhouette_score(pca_data_sampled, labels_pca)
        ch_pca = calinski_harabasz_score(pca_data_sampled, labels_pca)
        db_pca = davies_bouldin_score(pca_data_sampled, labels_pca)
        
        # Calculate metrics for original data clustering
        sil_original = silhouette_score(original_data_sampled, labels_original)
        ch_original = calinski_harabasz_score(original_data_sampled, labels_original)
        db_original = davies_bouldin_score(original_data_sampled, labels_original)
        
        # Store results
        comparison_results.append({
            'k': k,
            'method': 'PCA',
            'silhouette': sil_pca,
            'calinski_harabasz': ch_pca,
            'davies_bouldin': db_pca
        })
        
        comparison_results.append({
            'k': k,
            'method': 'No-PCA',
            'silhouette': sil_original,
            'calinski_harabasz': ch_original,
            'davies_bouldin': db_original
        })
        
        print(f"{'PCA':<10} {sil_pca:<12.4f} {ch_pca:<18.2f} {db_pca:<15.4f}")
        print(f"{'No-PCA':<10} {sil_original:<12.4f} {ch_original:<18.2f} {db_original:<15.4f}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save comparison results
        self.save_table(comparison_df, 'clustering_methods_comparison_k2.csv', 'comparison')
        
        # Create comparison visualizations for k=2 only
        self._create_comparison_visualizations_k2(comparison_df)
        
        # Determine best method
        best_method = self._determine_best_method(comparison_df)
        
        # Store results for reporting
        self.comparison_results = comparison_df
        
        return comparison_df

    def _create_comparison_visualizations_k2(self, comparison_df):
        """Create visualizations comparing clustering methods for k=2 only"""
        
        # Create a single figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for k=2
        pca_data = comparison_df[comparison_df['method'] == 'PCA'].iloc[0]
        no_pca_data = comparison_df[comparison_df['method'] == 'No-PCA'].iloc[0]
        
        # 1. Silhouette Score Comparison
        methods = ['PCA', 'No-PCA']
        silhouette_scores = [pca_data['silhouette'], no_pca_data['silhouette']]
        
        bars1 = axes[0,0].bar(methods, silhouette_scores, color=['steelblue', 'coral'], alpha=0.8)
        axes[0,0].set_ylabel('Silhouette Score')
        axes[0,0].set_title('Silhouette Score Comparison (k=2)\n(Higher is Better)', fontsize=12)
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Calinski-Harabasz Score Comparison
        calinski_scores = [pca_data['calinski_harabasz'], no_pca_data['calinski_harabasz']]
        
        bars2 = axes[0,1].bar(methods, calinski_scores, color=['steelblue', 'coral'], alpha=0.8)
        axes[0,1].set_ylabel('Calinski-Harabasz Score')
        axes[0,1].set_title('Calinski-Harabasz Index Comparison (k=2)\n(Higher is Better)', fontsize=12)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Davies-Bouldin Score Comparison
        davies_scores = [pca_data['davies_bouldin'], no_pca_data['davies_bouldin']]
        
        bars3 = axes[1,0].bar(methods, davies_scores, color=['steelblue', 'coral'], alpha=0.8)
        axes[1,0].set_ylabel('Davies-Bouldin Score')
        axes[1,0].set_title('Davies-Bouldin Index Comparison (k=2)\n(Lower is Better)', fontsize=12)
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Detailed metrics table
        cell_text = [
            [f"{pca_data['silhouette']:.4f}", f"{pca_data['calinski_harabasz']:.2f}", f"{pca_data['davies_bouldin']:.4f}"],
            [f"{no_pca_data['silhouette']:.4f}", f"{no_pca_data['calinski_harabasz']:.2f}", f"{no_pca_data['davies_bouldin']:.4f}"]
        ]
        
        columns = ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin']
        rows = ['PCA', 'No-PCA']
        
        # Create table
        table = axes[1,1].table(cellText=cell_text,
                         rowLabels=rows,
                         colLabels=columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.3, 0.8, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        axes[1,1].set_title('Detailed Metrics Comparison (k=2)', fontsize=12)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        self.save_plot(plt, 'clustering_methods_comparison_k2.png', 'comparison')
    
    def _determine_best_method(self, comparison_df):
        """Determine the best clustering method based on multiple metrics"""
        # Since we're only using k=2, we can simplify this
        pca_data = comparison_df[comparison_df['method'] == 'PCA'].iloc[0]
        no_pca_data = comparison_df[comparison_df['method'] == 'No-PCA'].iloc[0]
        
        # Score calculation (higher is better)
        # For Silhouette and Calinski-Harabasz: higher is better
        # For Davies-Bouldin: lower is better, so we use 1/DB
        pca_score = (pca_data['silhouette'] + 
                    pca_data['calinski_harabasz'] / 1000 +  # Normalize CH score
                    1 / pca_data['davies_bouldin'])
        
        no_pca_score = (no_pca_data['silhouette'] + 
                       no_pca_data['calinski_harabasz'] / 1000 +
                       1 / no_pca_data['davies_bouldin'])
        
        best_method = 'PCA' if pca_score > no_pca_score else 'No-PCA'
        
        print(f"\n🏆 METHOD COMPARISON RESULTS (k=2):")
        print("=" * 50)
        print(f"PCA Method Scores:")
        print(f"  - Silhouette: {pca_data['silhouette']:.4f}")
        print(f"  - Calinski-Harabasz: {pca_data['calinski_harabasz']:.2f}")
        print(f"  - Davies-Bouldin: {pca_data['davies_bouldin']:.4f}")
        print(f"  - Combined Score: {pca_score:.4f}")
        
        print(f"\nNo-PCA Method Scores:")
        print(f"  - Silhouette: {no_pca_data['silhouette']:.4f}")
        print(f"  - Calinski-Harabasz: {no_pca_data['calinski_harabasz']:.2f}")
        print(f"  - Davies-Bouldin: {no_pca_data['davies_bouldin']:.4f}")
        print(f"  - Combined Score: {no_pca_score:.4f}")
        
        print(f"\n🎯 RECOMMENDATION: Use {best_method} method")
        
        if best_method == 'PCA':
            advantage = pca_score - no_pca_score
            print(f"📈 PCA provides {advantage:.4f} better combined score")
            print("💡 PCA helps with: dimensionality reduction, noise reduction, and computational efficiency")
        else:
            advantage = no_pca_score - pca_score
            print(f"📈 No-PCA provides {advantage:.4f} better combined score")
            print("💡 Original features preserve all information but may be affected by noise")
        
        # Save recommendation
        recommendation = {
            'optimal_k': 2,
            'best_method': best_method,
            'pca_combined_score': pca_score,
            'no_pca_combined_score': no_pca_score,
            'advantage': advantage,
            'pca_silhouette': pca_data['silhouette'],
            'no_pca_silhouette': no_pca_data['silhouette'],
            'pca_calinski_harabasz': pca_data['calinski_harabasz'],
            'no_pca_calinski_harabasz': no_pca_data['calinski_harabasz'],
            'pca_davies_bouldin': pca_data['davies_bouldin'],
            'no_pca_davies_bouldin': no_pca_data['davies_bouldin']
        }
        
        recommendation_df = pd.DataFrame([recommendation])
        self.save_table(recommendation_df, 'method_recommendation.csv', 'comparison')
        
        return best_method
    
    def visualize_load_results(self):
        """Create publication-ready visualizations"""
        print("\n🎨 Creating publication-ready visualizations...")
        print("=" * 60)
        
        # 1. PCA clustering scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                             c=self.labels, cmap='viridis', alpha=0.7, s=50, 
                             edgecolor='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, label='Load Level')
        cbar.set_ticks(range(self.kmeans.n_clusters))
        plt.xlabel(f'Principal Component 1 ({self.pca_final.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({self.pca_final.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.title('Load Level Clustering in PCA Space', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        self.save_plot(plt, 'pca_clustering_scatter.png')
        
        # 2. Load level feature comparison
        centers_path = os.path.join(self.subfolders['tables'], 'cluster_centers.csv')
        load_centers_df = pd.read_csv(centers_path)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, feature in enumerate(self.load_related_features):
            if i < len(axes):
                bars = axes[i].bar(range(len(load_centers_df)), load_centers_df[feature], 
                                 color=colors[:len(load_centers_df)], alpha=0.8)
                axes[i].set_title(f'{feature} by Load Level', fontsize=12, pad=10)
                axes[i].set_xlabel('Load Level', fontsize=11)
                axes[i].set_ylabel(feature, fontsize=11)
                axes[i].set_xticks(range(len(load_centers_df)))
                axes[i].grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        self.save_plot(plt, 'load_level_comparison.png')
        
        # 3. Load level distribution (pie chart)
        cluster_counts = self.combined_data_cleaned['load_cluster'].value_counts().sort_index()
        
        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(cluster_counts.values, 
                                          labels=[f'Load Level {i}' for i in cluster_counts.index], 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=colors[:len(cluster_counts)],
                                          shadow=True)
        
        # Improve text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title('Load Level Distribution', fontsize=14, pad=20)
        self.save_plot(plt, 'load_level_distribution.png')
        
        # 4. Feature relationships by load level
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.original_data['Fuel Consumption Rate'], 
                            self.original_data['Engine Speed'],
                            c=self.labels, cmap='viridis', alpha=0.6, s=40)
        plt.colorbar(scatter, label='Load Level')
        plt.xlabel('Fuel Consumption Rate', fontsize=12)
        plt.ylabel('Engine Speed', fontsize=12)
        plt.title('Feature Relationship: Fuel Consumption vs Engine Speed', fontsize=14)
        plt.grid(True, alpha=0.3)
        self.save_plot(plt, 'feature_relationship.png')
        
        return True
    
    def generate_load_report(self):
        """Generate comprehensive analysis report"""
        # Read cluster center data
        centers_path = os.path.join(self.subfolders['tables'], 'cluster_centers.csv')
        cluster_stats = pd.read_csv(centers_path).set_index('Cluster')
        
        # Read cluster statistics
        stats_path = os.path.join(self.subfolders['tables'], 'cluster_statistics.csv')
        cluster_counts = pd.read_csv(stats_path).set_index('Cluster')
        
        # Calculate silhouette score
        sil_score = silhouette_score(self.pca_result, self.labels)
        
        # Add comparison results if available
        comparison_section = ""
        if hasattr(self, 'comparison_results') and not self.comparison_results.empty:
            best_method = self._determine_best_method(self.comparison_results)
            comparison_section = f"""
METHOD COMPARISON RESULTS:
- Best Method: {best_method}
- PCA Silhouette Score: {self.comparison_results[(self.comparison_results['method']=='PCA') & (self.comparison_results['k']==2)]['silhouette'].values[0]:.4f}
- No-PCA Silhouette Score: {self.comparison_results[(self.comparison_results['method']=='No-PCA') & (self.comparison_results['k']==2)]['silhouette'].values[0]:.4f}
- Recommendation: {'PCA method provides better clustering structure' if best_method == 'PCA' else 'Original features provide better clustering structure'}
- Note: Comparison was performed using {self.sample_fraction*100}% of data for efficiency
"""
        
        report_content = f"""
Load Level Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
==================================================================

DATA OVERVIEW:
- Files Analyzed: {len(self.file_names)}
- Total Samples: {len(self.combined_data_cleaned)}
- Load-Related Features: {len(self.load_related_features)}

PRINCIPAL COMPONENT ANALYSIS:
- Components Retained: {self.n_components}
- Total Variance Explained: {self.pca_final.explained_variance_ratio_.sum():.2%}

LOAD LEVEL CLUSTERING RESULTS:
- Number of Load Levels: {self.kmeans.n_clusters}
- Clustering Quality (Silhouette Score): {sil_score:.3f}
- Sample Distribution by Load Level:
{cluster_counts['Sample Count'].to_string()}

{comparison_section}

CHARACTERISTIC FEATURE VALUES BY LOAD LEVEL:
{cluster_stats.to_string()}

LOAD LEVEL PROFILES:
"""
        
        # Generate descriptive names for each load level
        fuel_consumption = cluster_stats['Fuel Consumption Rate']
        engine_speed = cluster_stats['Engine Speed']
        
        for cluster_id in cluster_stats.index:
            report_content += f"\nLevel {cluster_id}: "
            
            # Determine load level characteristics
            if fuel_consumption[cluster_id] < fuel_consumption.quantile(0.33):
                level_desc = "Low Load"
            elif fuel_consumption[cluster_id] < fuel_consumption.quantile(0.67):
                level_desc = "Medium Load"
            else:
                level_desc = "High Load"
                
            # Add specific characteristics
            if engine_speed[cluster_id] > engine_speed.median():
                level_desc += " | High RPM"
            else:
                level_desc += " | Normal RPM"
                
            report_content += f"{level_desc}"
            report_content += f" (Fuel: {cluster_stats.loc[cluster_id, 'Fuel Consumption Rate']:.2f}, "
            report_content += f"Speed: {cluster_stats.loc[cluster_id, 'Engine Speed']:.2f})"
        
        report_content += f"""

INTERPRETATION GUIDELINES:
- Low Load: Typically represents idling, light cruising, or deceleration
- Medium Load: Represents normal urban driving and moderate acceleration
- High Load: Indicates heavy acceleration, climbing, or high-speed operation

DATA QUALITY INDICATORS:
- Silhouette Score > 0.5: Strong cluster structure
- Silhouette Score 0.25-0.5: Reasonable structure
- Silhouette Score < 0.25: Weak structure

Current clustering quality: {'Strong' if sil_score > 0.5 else 'Reasonable' if sil_score > 0.25 else 'Weak'}
"""
        
        # Save comprehensive report
        report_path = os.path.join(self.subfolders['reports'], 'load_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📄 Analysis report saved: {report_path}")
        
        return True
    
    def run_full_analysis(self):
        """Run the complete load analysis journey"""
        try:
            print("🚀 Starting the Load Level Analysis Adventure!")
            print("⭐" * 60)
            
            self.load_pkl_files()
            self.preprocess_data()
            self.perform_pca()
            self.analyze_pca_components()
            self.perform_load_clustering()
            self.compare_clustering_methods()  # New method added
            self.visualize_load_results()
            self.generate_load_report()
            
            print("\n" + "🎉" * 30)
            print("ANALYSIS COMPLETE! 🎊")
            print(f"All results saved to: {os.path.abspath(self.output_folder)}")
            print("⭐" * 60)
            print("\n📁 Your results are organized in:")
            for folder_name, folder_path in self.subfolders.items():
                print(f"   📂 {folder_name}: {os.path.basename(folder_path)}/")
            
        except Exception as e:
            print(f"❌ Analysis encountered an issue: {e}")
            import traceback
            traceback.print_exc()

# Usage example
if __name__ == "__main__":
    # Set your pkl folder path
    folder_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\trip数据"
    
    # Optional: specify output folder name
    output_folder = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\聚类尝试（秋季）\负载pca尝试"
    
    # Create analyzer instance and run the analysis with sampling for comparison
    print("🔧 Initializing Load Level Analysis System...")
    analyzer = LoadClusterAnalysis(folder_path, output_folder, sample_fraction=0.1)  # Use 10% of data for method comparison
    analyzer.run_full_analysis()