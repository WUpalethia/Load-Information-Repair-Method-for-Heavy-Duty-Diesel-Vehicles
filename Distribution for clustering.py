import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import warnings
from datetime import datetime
import time
warnings.filterwarnings('ignore')

# Set professional scientific plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

class ClusteringTendencyAnalyzer:
    """Specialized analyzer for detecting clustering tendency in datasets"""
    
    def __init__(self, folder_path, output_folder=None):
        self.folder_path = folder_path
        
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"clustering_tendency_analysis_{timestamp}"
        else:
            self.output_folder = output_folder
        
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output directory created: {os.path.abspath(self.output_folder)}")
        
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
            '发动机燃料流量平均值/Average fuel flow': 'Fuel Consumption Rate',
            '发动机转速平均值/Average engine speed': 'Engine Speed', 
            '摩擦扭矩平均值/Average friction torque': 'Friction Torque',
            '车速平均值/Average speed': 'Vehicle Speed'
        }
        
        self.load_related_features = list(self.selected_columns_mapping.values())
    
    def save_plot(self, plt, filename, subfolder='plots'):
        """Save publication-quality plots"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', 
                   transparent=False, edgecolor='none')
        print(f"📊 Plot saved: {filepath}")
        plt.close()
        return filepath
    
    def save_table(self, df, filename, subfolder='tables'):
        """Save data tables"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"📋 Table saved: {filepath}")
        return filepath
    
    def load_and_preprocess_data(self):
        """Load and preprocess data from pickle files"""
        print("🚀 Starting data loading process...")
        start_time = time.time()
        
        pkl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        
        if not pkl_files:
            raise ValueError("❌ No pickle files found in the specified directory!")
        
        print(f"🔍 Found {len(pkl_files)} data files to process")
        
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
                    print(f"✅ Successfully loaded: {file_name}")
                
            except Exception as e:
                print(f"❌ Error loading {file_name}: {e}")
        
        if not self.all_data:
            raise ValueError("❌ No data files were successfully loaded!")
        
        self.combined_data = pd.concat(self.all_data, ignore_index=True)
        print(f"📊 Combined dataset: {len(self.all_data)} files, shape: {self.combined_data.shape}")
        
        # Data preprocessing
        data_for_analysis = self.combined_data[self.load_related_features].copy()
        data_cleaned = data_for_analysis.dropna()
        
        if len(data_cleaned) == 0:
            raise ValueError("❌ No samples remaining after data cleaning!")
        
        print(f"🧹 Cleaned dataset shape: {data_cleaned.shape}")
        
        # Standardize the data
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data_cleaned)
        self.original_data = data_cleaned.copy()
        
        elapsed_time = time.time() - start_time
        print(f"✅ Data preprocessing completed in {elapsed_time:.2f} seconds")
        return True
    
    def perform_pca_analysis(self):
        """Perform PCA analysis for dimensionality reduction and visualization"""
        print("\n🔮 Performing Principal Component Analysis...")
        start_time = time.time()
        
        self.pca = PCA(n_components=0.85, svd_solver='full')
        self.pca_result = self.pca.fit_transform(self.scaled_data)
        self.n_components = self.pca.n_components_
        
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        print(f"📈 Retained {self.n_components} principal components (explaining {cumulative_variance[-1]:.2%} of variance)")
        
        # Create individual PCA visualizations
        self.create_individual_pca_plots(explained_variance, cumulative_variance)
        
        elapsed_time = time.time() - start_time
        print(f"✅ PCA analysis completed in {elapsed_time:.2f} seconds")
        return True
    
    def create_individual_pca_plots(self, explained_variance, cumulative_variance):
        """Create individual publication-quality PCA plots"""
        
        # 1. Scree Plot
        plt.figure(figsize=(10, 6))
        components = range(1, len(explained_variance) + 1)
        
        bars = plt.bar(components, explained_variance, alpha=0.7, color='steelblue', 
                      label='Individual Variance')
        plt.plot(components, cumulative_variance, 'ro-', linewidth=2.5, markersize=8,
                label='Cumulative Variance')
        plt.axhline(y=0.85, color='forestgreen', linestyle='--', alpha=0.8, 
                   linewidth=2, label='85% Variance Threshold')
        
        # Add value annotations
        for i, (x, y) in enumerate(zip(components, explained_variance)):
            plt.text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Principal Component', fontsize=14)
        plt.ylabel('Explained Variance Ratio', fontsize=14)
        plt.title('PCA Scree Plot: Variance Explained by Components', fontsize=16, pad=20)
        plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.xticks(components)
        self.save_plot(plt, 'pca_scree_plot.png')
        
        # 2. PCA Scatter Plot (First two components)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                             alpha=0.6, s=40, c='mediumseagreen', edgecolor='white', linewidth=0.5)
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)', fontsize=14)
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)', fontsize=14)
        plt.title('PCA Projection: First Two Principal Components', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        self.save_plot(plt, 'pca_scatter_plot.png')
        
        # 3. Component Loadings Heatmap
        plt.figure(figsize=(12, 8))
        feature_names = self.load_related_features
        pca_components = self.pca.components_[:2]
        
        sns.heatmap(pca_components.T, 
                   xticklabels=['PC1', 'PC2'],
                   yticklabels=feature_names,
                   annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                   annot_kws={'size': 11}, cbar_kws={'label': 'Component Weight', 'shrink': 0.8},
                   square=True, linewidths=0.5)
        plt.title('PCA Component Loadings: Feature Contributions', fontsize=16, pad=20)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12)
        self.save_plot(plt, 'pca_loadings_heatmap.png')
        
        # 4. Cumulative Variance Plot
        plt.figure(figsize=(10, 6))
        plt.plot(components, cumulative_variance, 's-', linewidth=3, markersize=8,
                color='crimson', markerfacecolor='white', markeredgewidth=2)
        plt.axhline(y=0.85, color='darkorange', linestyle='--', alpha=0.8, linewidth=2,
                   label='85% Variance Threshold')
        plt.fill_between(components, cumulative_variance, alpha=0.2, color='crimson')
        
        plt.xlabel('Number of Principal Components', fontsize=14)
        plt.ylabel('Cumulative Explained Variance', fontsize=14)
        plt.title('Cumulative Variance Explained by PCA Components', fontsize=16, pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(components)
        plt.ylim(0, 1.0)
        self.save_plot(plt, 'pca_cumulative_variance.png')
    
    def calculate_hopkins_statistic(self, sample_size=None):
        """Calculate Hopkins statistic to measure clustering tendency"""
        X = self.scaled_data
        
        if sample_size is None:
            sample_size = min(1000, len(X) // 3)
        
        n_samples = min(sample_size, len(X))
        
        # Randomly select actual data points
        np.random.seed(42)
        actual_indices = np.random.choice(len(X), n_samples, replace=False)
        actual_sample = X[actual_indices]
        
        # Generate uniformly distributed random points
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        
        uniform_sample = np.random.uniform(
            low=min_vals, 
            high=max_vals, 
            size=(n_samples, X.shape[1])
        )
        
        # Calculate nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        
        # Distance from actual points to nearest actual neighbors
        actual_distances, _ = nbrs.kneighbors(actual_sample, n_neighbors=2)
        actual_distances = actual_distances[:, 1]  # Exclude self
        
        # Distance from uniform points to nearest actual points
        uniform_distances, _ = nbrs.kneighbors(uniform_sample, n_neighbors=1)
        uniform_distances = uniform_distances[:, 0]
        
        # Calculate Hopkins statistic
        numerator = np.sum(uniform_distances)
        denominator = np.sum(actual_distances) + np.sum(uniform_distances)
        
        hopkins_stat = numerator / denominator if denominator > 0 else 0.5
        
        return hopkins_stat
    
    def calculate_variance_ratio(self):
        """Calculate variance ratio to assess clustering tendency"""
        X = self.scaled_data
        
        # Calculate pairwise distances with sampling for large datasets
        if len(X) > 1000:
            indices = np.random.choice(len(X), 1000, replace=False)
            X_sample = X[indices]
            distances = pairwise_distances(X_sample)
        else:
            distances = pairwise_distances(X)
        
        # Calculate ratio of standard deviation to mean distance
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        mean_distance = np.mean(upper_triangle)
        std_distance = np.std(upper_triangle)
        
        variance_ratio = std_distance / mean_distance if mean_distance > 0 else 0
        
        return variance_ratio
    
    def calculate_distance_distribution_metrics(self):
        """Calculate distance distribution metrics for clustering tendency"""
        X = self.scaled_data
        
        # Sample for computational efficiency
        sample_size = min(1000, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        
        # Calculate nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_sample)
        distances, _ = nbrs.kneighbors(X_sample)
        nearest_distances = distances[:, 1]  # Distance to nearest neighbor (excluding self)
        
        # Calculate statistical metrics
        metrics = {
            'mean_nearest_distance': np.mean(nearest_distances),
            'std_nearest_distance': np.std(nearest_distances),
            'cv_nearest_distance': np.std(nearest_distances) / np.mean(nearest_distances) if np.mean(nearest_distances) > 0 else 0,
            'skewness_nearest_distance': pd.Series(nearest_distances).skew(),
            'kurtosis_nearest_distance': pd.Series(nearest_distances).kurtosis(),
            'distance_entropy': self.calculate_distance_entropy(nearest_distances)
        }
        
        return metrics, nearest_distances
    
    def calculate_distance_entropy(self, distances):
        """Calculate entropy of distance distribution"""
        hist, _ = np.histogram(distances, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero values
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def calculate_spatial_autocorrelation(self):
        """Calculate spatial autocorrelation (simplified Moran's I)"""
        X = self.scaled_data
        
        # Sample for computational efficiency
        sample_size = min(500, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        
        # Calculate distance matrix
        distances = pairwise_distances(X_sample)
        
        # Calculate spatial autocorrelation for each feature
        spatial_correlations = []
        for feature_idx in range(X_sample.shape[1]):
            feature_values = X_sample[:, feature_idx]
            feature_mean = np.mean(feature_values)
            
            numerator = 0
            denominator = 0
            
            for i in range(len(X_sample)):
                for j in range(len(X_sample)):
                    if i != j:
                        weight = 1 / (1 + distances[i, j])  # Distance-based weighting
                        numerator += weight * (feature_values[i] - feature_mean) * (feature_values[j] - feature_mean)
                        denominator += (feature_values[i] - feature_mean) ** 2
            
            if denominator > 0:
                morans_i = (len(X_sample) / np.sum(distances > 0)) * (numerator / denominator)
            else:
                morans_i = 0
            
            spatial_correlations.append(morans_i)
        
        return np.mean(spatial_correlations)
    
    def create_individual_clustering_tendency_plots(self, nearest_distances):
        """Create individual publication-quality clustering tendency plots"""
        X = self.scaled_data
        
        # Sample for computational efficiency
        sample_size = min(1000, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        
        # 1. Nearest Neighbor Distance Distribution
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(nearest_distances, bins=50, alpha=0.7, 
                                   color='lightcoral', density=True, edgecolor='black', linewidth=0.5)
        plt.axvline(np.mean(nearest_distances), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(nearest_distances):.3f}')
        plt.xlabel('Distance to Nearest Neighbor', fontsize=14)
        plt.ylabel('Probability Density', fontsize=14)
        plt.title('Distribution of Nearest Neighbor Distances', fontsize=16, pad=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        self.save_plot(plt, 'nearest_neighbor_distance_distribution.png')
        
        # 2. Cumulative Distribution Function of Distances
        plt.figure(figsize=(10, 6))
        sorted_distances = np.sort(nearest_distances)
        cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        plt.plot(sorted_distances, cdf, linewidth=3, color='royalblue')
        plt.xlabel('Distance', fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.title('Cumulative Distribution Function of Nearest Neighbor Distances', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        self.save_plot(plt, 'distance_cumulative_distribution.png')
        
        # 3. Feature Distribution Boxplot
        plt.figure(figsize=(12, 8))
        feature_data = []
        feature_names = []
        for i, feature in enumerate(self.load_related_features):
            feature_data.append(self.original_data.iloc[:, i])
            feature_names.append(feature)
        
        box_plot = plt.boxplot(feature_data, labels=feature_names, patch_artist=True)
        # Customize box colors
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Standardized Values', fontsize=14)
        plt.title('Feature Distribution Characteristics', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        self.save_plot(plt, 'feature_distribution_boxplot.png')
        
        # 4. Distance Matrix Heatmap (for smaller samples)
        if len(X_sample) <= 300:
            plt.figure(figsize=(10, 8))
            distance_matrix = pairwise_distances(X_sample[:50])  # Further sampling
            
            im = plt.imshow(distance_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Euclidean Distance', shrink=0.8)
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('Sample Index', fontsize=14)
            plt.title('Pairwise Distance Matrix (First 50 Samples)', fontsize=16, pad=20)
            self.save_plot(plt, 'distance_matrix_heatmap.png')
        
        # 5. Q-Q Plot for Normality Assessment
        from scipy import stats
        plt.figure(figsize=(10, 6))
        stats.probplot(nearest_distances, dist="norm", plot=plt)
        plt.grid(True, alpha=0.3)
        plt.title('Q-Q Plot: Normality Assessment of Nearest Neighbor Distances', fontsize=16, pad=20)
        self.save_plot(plt, 'distance_qq_plot.png')
        
        # 6. Hopkins Statistic Visualization
        self.create_hopkins_visualization()
    
    def create_hopkins_visualization(self):
        """Create visualization explaining Hopkins statistic"""
        plt.figure(figsize=(12, 8))
        
        # Create a conceptual visualization
        hopkins_values = np.linspace(0, 1, 100)
        tendency_levels = np.zeros_like(hopkins_values)
        
        # Define tendency levels based on Hopkins values
        tendency_levels[hopkins_values <= 0.5] = 0  # Random
        tendency_levels[(hopkins_values > 0.5) & (hopkins_values <= 0.65)] = 1  # Weak
        tendency_levels[(hopkins_values > 0.65) & (hopkins_values <= 0.75)] = 2  # Moderate
        tendency_levels[hopkins_values > 0.75] = 3  # Strong
        
        colors = ['lightcoral', 'gold', 'lightgreen', 'darkgreen']
        labels = ['Random Distribution', 'Weak Clustering', 'Moderate Clustering', 'Strong Clustering']
        
        for i in range(4):
            mask = tendency_levels == i
            if np.any(mask):
                plt.fill_between(hopkins_values[mask], 0, 1, color=colors[i], alpha=0.6, label=labels[i])
        
        plt.xlabel('Hopkins Statistic Value', fontsize=14)
        plt.ylabel('Clustering Tendency Level', fontsize=14)
        plt.title('Hopkins Statistic: Interpretation Guide for Clustering Tendency', fontsize=16, pad=20)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add interpretation text
        interpretation_text = [
            'H ≤ 0.5: Data is likely randomly distributed',
            '0.5 < H ≤ 0.65: Weak clustering tendency',
            '0.65 < H ≤ 0.75: Moderate clustering tendency', 
            'H > 0.75: Strong clustering tendency'
        ]
        
        for i, text in enumerate(interpretation_text):
            plt.text(0.02, 0.9 - i*0.1, text, transform=plt.gca().transAxes, 
                    fontsize=11, verticalalignment='top')
        
        self.save_plot(plt, 'hopkins_statistic_interpretation.png')
    
    def interpret_clustering_tendency(self, hopkins_stat, variance_ratio, distance_metrics, spatial_autocorr):
        """Interpret and provide recommendations based on clustering tendency metrics"""
        result = {}
        
        # Hopkins statistic interpretation
        if hopkins_stat > 0.75:
            hopkins_assessment = "Strong Clustering Tendency"
            hopkins_confidence = "High"
            hopkins_recommendation = "Data exhibits clear cluster structure, highly suitable for clustering analysis"
        elif hopkins_stat > 0.65:
            hopkins_assessment = "Moderate Clustering Tendency"
            hopkins_confidence = "Medium"
            hopkins_recommendation = "Data shows some cluster structure, suitable for clustering analysis"
        elif hopkins_stat > 0.5:
            hopkins_assessment = "Weak Clustering Tendency"
            hopkins_confidence = "Low"
            hopkins_recommendation = "Data may have subtle cluster patterns, clustering results may be limited"
        else:
            hopkins_assessment = "Random or Uniform Distribution"
            hopkins_confidence = "Very Low"
            hopkins_recommendation = "Data appears randomly distributed, clustering analysis not recommended"
        
        # Variance ratio interpretation
        if variance_ratio > 0.4:
            variance_assessment = "High distance variability suggesting cluster structure"
        elif variance_ratio > 0.2:
            variance_assessment = "Moderate distance variability"
        else:
            variance_assessment = "Low distance variability, possibly uniform distribution"
        
        # Distance distribution coefficient of variation
        cv = distance_metrics['cv_nearest_distance']
        if cv > 0.5:
            cv_assessment = "Uneven distance distribution indicating potential clusters"
        elif cv > 0.3:
            cv_assessment = "Moderate distance distribution variation"
        else:
            cv_assessment = "Relatively uniform distance distribution"
        
        # Spatial autocorrelation assessment
        if spatial_autocorr > 0.3:
            spatial_assessment = "Positive spatial correlation supporting clustering"
        elif spatial_autocorr > 0.1:
            spatial_assessment = "Mild spatial correlation"
        else:
            spatial_assessment = "Weak spatial correlation"
        
        result['assessment'] = f"{hopkins_assessment} | {variance_assessment}"
        result['recommendation'] = hopkins_recommendation
        result['confidence'] = hopkins_confidence
        result['hopkins_statistic'] = hopkins_stat
        result['variance_ratio'] = variance_ratio
        result['distance_cv'] = cv
        result['spatial_autocorrelation'] = spatial_autocorr
        
        # Print detailed results
        print(f"\n📊 CLUSTERING TENDENCY ANALYSIS RESULTS:")
        print("="*50)
        print(f"• Hopkins Statistic: {hopkins_stat:.4f}")
        print(f"• Distance Variance Ratio: {variance_ratio:.4f}")
        print(f"• Nearest Neighbor Distance CV: {cv:.4f}")
        print(f"• Spatial Autocorrelation: {spatial_autocorr:.4f}")
        print(f"• Overall Assessment: {result['assessment']}")
        print(f"• Confidence Level: {result['confidence']}")
        print(f"• Recommendation: {result['recommendation']}")
        
        # Hopkins statistic detailed interpretation
        print(f"\n🔍 HOPKINS STATISTIC INTERPRETATION:")
        print(f"  H = {hopkins_stat:.3f}:")
        if hopkins_stat > 0.75:
            print("  → Strong clustering tendency detected (H > 0.75)")
            print("  → Highly recommended for clustering analysis")
        elif hopkins_stat > 0.65:
            print("  → Moderate clustering tendency (0.65 < H ≤ 0.75)")
            print("  → Suitable for clustering analysis")
        elif hopkins_stat > 0.5:
            print("  → Weak clustering tendency (0.5 < H ≤ 0.65)")
            print("  → Clustering may yield limited results")
        else:
            print("  → Data likely randomly distributed (H ≤ 0.5)")
            print("  → Not recommended for clustering analysis")
        
        return result
    
    def perform_comprehensive_tendency_analysis(self):
        """Perform comprehensive clustering tendency analysis"""
        print("\n🔬 Starting Comprehensive Clustering Tendency Analysis...")
        start_time = time.time()
        
        # Calculate Hopkins statistic
        print("   📈 Calculating Hopkins statistic...")
        hopkins_stat = self.calculate_hopkins_statistic()
        
        # Calculate variance ratio
        print("   📊 Computing distance variance ratio...")
        variance_ratio = self.calculate_variance_ratio()
        
        # Calculate distance distribution metrics
        print("   📐 Analyzing distance distribution patterns...")
        distance_metrics, nearest_distances = self.calculate_distance_distribution_metrics()
        
        # Calculate spatial autocorrelation
        print("   🌐 Assessing spatial autocorrelation...")
        spatial_autocorr = self.calculate_spatial_autocorrelation()
        
        # Create individual visualizations
        print("   🎨 Generating publication-quality visualizations...")
        self.create_individual_clustering_tendency_plots(nearest_distances)
        
        # Interpret results
        print("   💡 Interpreting analysis results...")
        tendency_result = self.interpret_clustering_tendency(
            hopkins_stat, variance_ratio, distance_metrics, spatial_autocorr
        )
        
        # Save detailed results
        detailed_results = {
            'Hopkins_Statistic': hopkins_stat,
            'Variance_Ratio': variance_ratio,
            'Mean_Nearest_Distance': distance_metrics['mean_nearest_distance'],
            'Std_Nearest_Distance': distance_metrics['std_nearest_distance'],
            'CV_Nearest_Distance': distance_metrics['cv_nearest_distance'],
            'Skewness_Nearest_Distance': distance_metrics['skewness_nearest_distance'],
            'Kurtosis_Nearest_Distance': distance_metrics['kurtosis_nearest_distance'],
            'Distance_Entropy': distance_metrics['distance_entropy'],
            'Spatial_Autocorrelation': spatial_autocorr,
            'Clustering_Tendency_Assessment': tendency_result['assessment'],
            'Confidence_Level': tendency_result['confidence'],
            'Recommendation': tendency_result['recommendation'],
            'Sample_Count': len(self.scaled_data),
            'Feature_Count': len(self.load_related_features)
        }
        
        tendency_df = pd.DataFrame([detailed_results])
        self.save_table(tendency_df, 'comprehensive_clustering_tendency_results.csv')
        
        # Create summary report
        self.create_summary_report(tendency_result, detailed_results)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Clustering tendency analysis completed in {elapsed_time:.2f} seconds")
        
        return tendency_df
    
    def create_summary_report(self, tendency_result, detailed_results):
        """Create comprehensive summary report"""
        report_content = f"""
# Clustering Tendency Analysis Report

## Analysis Overview
- **Analysis Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Samples**: {detailed_results['Sample_Count']:,}
- **Features Analyzed**: {detailed_results['Feature_Count']}
- **Output Directory**: {os.path.abspath(self.output_folder)}

## Key Metric Results

### Primary Clustering Tendency Indicators

1. **Hopkins Statistic**: {detailed_results['Hopkins_Statistic']:.4f}
   - **Interpretation**: {tendency_result['recommendation']}

2. **Supporting Metrics**
   - Distance Variance Ratio: {detailed_results['Variance_Ratio']:.4f}
   - Nearest Neighbor Distance CV: {detailed_results['CV_Nearest_Distance']:.4f}
   - Spatial Autocorrelation: {detailed_results['Spatial_Autocorrelation']:.4f}
   - Distance Distribution Skewness: {detailed_results['Skewness_Nearest_Distance']:.4f}
   - Distance Distribution Kurtosis: {detailed_results['Kurtosis_Nearest_Distance']:.4f}

### Comprehensive Assessment
- **Overall Assessment**: {tendency_result['assessment']}
- **Confidence Level**: {tendency_result['confidence']}

## Analytical Recommendation
{tendency_result['recommendation']}

## Hopkins Statistic Reference Ranges
- **H > 0.75**: Strong clustering tendency - Highly suitable for clustering
- **0.65 < H ≤ 0.75**: Moderate clustering tendency - Suitable for clustering  
- **0.5 < H ≤ 0.65**: Weak clustering tendency - Limited clustering utility
- **H ≤ 0.5**: Random/uniform distribution - Not recommended for clustering

**Current Analysis**: Hopkins statistic = {detailed_results['Hopkins_Statistic']:.4f}
- Classification: {['Random Distribution', 'Weak Clustering', 'Moderate Clustering', 'Strong Clustering'][(detailed_results['Hopkins_Statistic'] > 0.5) + (detailed_results['Hopkins_Statistic'] > 0.65) + (detailed_results['Hopkins_Statistic'] > 0.75)]}

## Generated Visualizations
The analysis produced the following publication-quality figures:

1. **PCA Scree Plot** - Variance explained by principal components
2. **PCA Scatter Plot** - Data distribution in reduced dimension space
3. **PCA Loadings Heatmap** - Feature contributions to principal components
4. **Cumulative Variance Plot** - Progressive variance explanation
5. **Nearest Neighbor Distance Distribution** - Distance concentration analysis
6. **Distance Cumulative Distribution** - Distance distribution shape
7. **Feature Distribution Boxplot** - Feature variability assessment
8. **Distance Matrix Heatmap** - Pairwise distance patterns
9. **Distance Q-Q Plot** - Normality assessment of distances
10. **Hopkins Statistic Interpretation** - Statistical reference guide

## Methodological Notes
- All analyses performed on standardized data (z-score normalization)
- Computational efficiency maintained through strategic sampling
- Multiple complementary metrics used for robust assessment
- Visualizations designed for scientific publication standards
"""
        
        report_path = os.path.join(self.subfolders['reports'], 'clustering_tendency_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 Comprehensive report saved: {report_path}")
        
        # Console summary output
        print("\n" + "="*60)
        print("CLUSTERING TENDENCY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Hopkins Statistic: {detailed_results['Hopkins_Statistic']:.4f}")
        print(f"Assessment: {tendency_result['assessment']}")
        print(f"Confidence: {tendency_result['confidence']}")
        print(f"Recommendation: {tendency_result['recommendation']}")
        print("="*60)
    
    def run_comprehensive_analysis(self):
        """Execute complete clustering tendency analysis pipeline"""
        try:
            print("🚀 INITIATING CLUSTERING TENDENCY ANALYSIS")
            print("=" * 60)
            total_start_time = time.time()
            
            # 1. Data loading and preprocessing
            self.load_and_preprocess_data()
            
            # 2. PCA analysis for visualization
            self.perform_pca_analysis()
            
            # 3. Comprehensive clustering tendency analysis
            tendency_results = self.perform_comprehensive_tendency_analysis()
            
            total_elapsed_time = time.time() - total_start_time
            
            print("\n" + "=" * 60)
            print(f"🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total execution time: {total_elapsed_time:.2f} seconds")
            print(f"📁 All results saved to: {os.path.abspath(self.output_folder)}")
            
            # Final recommendation based on Hopkins statistic
            hopkins_stat = tendency_results.iloc[0]['Hopkins_Statistic']
            if hopkins_stat > 0.75:
                print("\n🎯 CONCLUSION: Strong clustering tendency detected - Highly recommended for clustering analysis!")
            elif hopkins_stat > 0.65:
                print("\n✅ CONCLUSION: Moderate clustering tendency - Suitable for clustering analysis.")
            elif hopkins_stat > 0.5:
                print("\n⚠️  CONCLUSION: Weak clustering tendency - Clustering analysis may have limited utility.")
            else:
                print("\n❌ CONCLUSION: Data appears randomly distributed - Clustering analysis not recommended.")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Analysis encountered an error: {e}")
            import traceback
            traceback.print_exc()

# Usage example
if __name__ == "__main__":
    # Set your data directory path
    folder_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\trip数据"
    
    # Optional: specify custom output directory
    output_folder = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\clustering_tendency_analysis"
    
    # Initialize and run the analysis
    print("🔧 Initializing Clustering Tendency Analyzer...")
    analyzer = ClusteringTendencyAnalyzer(folder_path, output_folder)
    
    # Execute comprehensive analysis
    analyzer.run_comprehensive_analysis()