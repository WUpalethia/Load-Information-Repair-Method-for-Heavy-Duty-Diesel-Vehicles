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
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial import ConvexHull
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Set plotting style for research publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SphericalClusterAnalyzer:
    """Analyzer specialized for detecting spherical cluster formations in data"""
    
    def __init__(self, folder_path, output_folder=None):
        self.folder_path = folder_path
        
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = f"spherical_cluster_analysis_{timestamp}"
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
            '发动机燃料流量平均值/Average fuel flow': 'avg_fuel_flow',
            '发动机转速平均值/Average engine speed': 'avg_engine_speed', 
            '摩擦扭矩平均值/Average friction torque': 'avg_friction_torque',
            '车速平均值/Average speed': 'avg_speed'
        }
        
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
        """Save data table"""
        filepath = os.path.join(self.subfolders[subfolder], filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath
    
    def load_and_preprocess_data(self):
        """Load and preprocess data from pickle files"""
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
        print(f"Loaded {len(self.all_data)} files, combined shape: {self.combined_data.shape}")
        
        # Data preprocessing
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
    
    def perform_pca_analysis(self):
        """Perform PCA analysis for visualization"""
        print("\nPerforming PCA analysis for visualization...")
        start_time = time.time()
        
        self.pca = PCA(n_components=0.95, svd_solver='full')
        self.pca_result = self.pca.fit_transform(self.scaled_data)
        self.n_components = self.pca.n_components_
        
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        print(f"Retained {self.n_components} principal components (explaining {cumulative_variance[-1]:.2%} variance)")
        
        # Create PCA visualizations
        self.create_individual_pca_plots(explained_variance, cumulative_variance)
        
        elapsed_time = time.time() - start_time
        print(f"PCA analysis completed, time: {elapsed_time:.2f}s")
        return True
    
    def create_individual_pca_plots(self, explained_variance, cumulative_variance):
        """Create individual PCA plots for research publication"""
        
        # 1. PCA Variance Explained Plot
        plt.figure(figsize=(10, 6))
        components = range(1, len(explained_variance) + 1)
        
        bars = plt.bar(components, explained_variance, alpha=0.7, color='steelblue', 
                      label='Individual Component')
        plt.plot(components, cumulative_variance, 'ro-', linewidth=2.5, 
                markersize=8, label='Cumulative Variance')
        plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label='95% Threshold')
        
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
    
    def calculate_hopkins_statistic(self, sample_size=None):
        """Calculate Hopkins statistic to measure clustering tendency"""
        X = self.scaled_data
        
        if sample_size is None:
            sample_size = min(1000, len(X) // 3)
        
        n_samples = min(sample_size, len(X))
        
        # Random sampling of actual data points
        np.random.seed(42)
        actual_indices = np.random.choice(len(X), n_samples, replace=False)
        actual_sample = X[actual_indices]
        
        # Generate uniform random points
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
        
        # Distance from uniform points to nearest actual neighbors
        uniform_distances, _ = nbrs.kneighbors(uniform_sample, n_neighbors=1)
        uniform_distances = uniform_distances[:, 0]
        
        # Calculate Hopkins statistic
        numerator = np.sum(uniform_distances)
        denominator = np.sum(actual_distances) + np.sum(uniform_distances)
        
        hopkins_stat = numerator / denominator if denominator > 0 else 0.5
        
        return hopkins_stat
    
    def calculate_sphericity_metrics(self):
        """Calculate spherical cluster metrics"""
        X = self.scaled_data
        
        # 1. Eigenvalue ratio of covariance matrix (sphericity measure)
        cov_matrix = np.cov(X.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        # Eigenvalue ratio metrics
        if len(eigenvalues) > 1:
            eigenvalue_ratio = eigenvalues[0] / eigenvalues[-1]
            eigenvalue_cv = np.std(eigenvalues) / np.mean(eigenvalues)
        else:
            eigenvalue_ratio = 1
            eigenvalue_cv = 0
        
        # 2. Isotropic metrics
        centroid = np.mean(X, axis=0)
        distances_to_center = np.linalg.norm(X - centroid, axis=1)
        
        # Coefficient of variation of distances
        distance_cv = np.std(distances_to_center) / np.mean(distances_to_center)
        
        # 3. Direction consistency metrics
        pca = PCA()
        pca.fit(X)
        explained_variance = pca.explained_variance_ratio_
        variance_entropy = -np.sum(explained_variance * np.log(explained_variance + 1e-8))
        
        # 4. Convex hull metrics for 2D/3D visualization
        if X.shape[1] >= 2:
            if hasattr(self, 'pca_result'):
                hull_points = self.pca_result[:, :2]
            else:
                hull_points = X[:, :2]
            
            try:
                hull = ConvexHull(hull_points)
                hull_area = hull.volume
                hull_points_area = hull_points
                
                # Approximate minimum enclosing circle
                from sklearn.preprocessing import StandardScaler
                scaled_hull = StandardScaler().fit_transform(hull_points_area)
                distances_from_origin = np.linalg.norm(scaled_hull, axis=1)
                circularity_ratio = np.std(distances_from_origin) / np.mean(distances_from_origin)
            except:
                hull_area = 0
                circularity_ratio = 1
        else:
            hull_area = 0
            circularity_ratio = 1
        
        metrics = {
            'eigenvalue_ratio': eigenvalue_ratio,
            'eigenvalue_cv': eigenvalue_cv,
            'distance_cv_to_center': distance_cv,
            'variance_entropy': variance_entropy,
            'circularity_ratio': circularity_ratio,
            'isometric_score': 1 / (1 + eigenvalue_ratio),
            'sphericity_score': 1 / (1 + distance_cv)
        }
        
        return metrics, distances_to_center, eigenvalues
    
    def analyze_cluster_shape_distribution(self, n_simulated_clusters=5):
        """Analyze shape distribution of potential clusters"""
        X = self.scaled_data
        
        shape_metrics = []
        
        for k in range(2, min(n_simulated_clusters + 2, 7)):
            try:
                kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
                labels = kmeans.fit_predict(X)
                
                cluster_shapes = []
                for cluster_id in range(k):
                    cluster_points = X[labels == cluster_id]
                    if len(cluster_points) > 10:
                        cluster_cov = np.cov(cluster_points.T)
                        cluster_eigenvalues = np.linalg.eigvals(cluster_cov)
                        cluster_eigenvalues = np.sort(cluster_eigenvalues)[::-1]
                        
                        if len(cluster_eigenvalues) > 1:
                            cluster_eigen_ratio = cluster_eigenvalues[0] / cluster_eigenvalues[-1]
                            cluster_centroid = np.mean(cluster_points, axis=0)
                            cluster_distances = np.linalg.norm(cluster_points - cluster_centroid, axis=1)
                            cluster_distance_cv = np.std(cluster_distances) / np.mean(cluster_distances)
                            
                            cluster_shapes.append({
                                'cluster_size': len(cluster_points),
                                'eigen_ratio': cluster_eigen_ratio,
                                'distance_cv': cluster_distance_cv,
                                'sphericity': 1 / (1 + cluster_eigen_ratio)
                            })
                
                if cluster_shapes:
                    avg_eigen_ratio = np.mean([s['eigen_ratio'] for s in cluster_shapes])
                    avg_sphericity = np.mean([s['sphericity'] for s in cluster_shapes])
                    shape_metrics.append({
                        'k': k,
                        'avg_eigen_ratio': avg_eigen_ratio,
                        'avg_sphericity': avg_sphericity,
                        'n_clusters_analyzed': len(cluster_shapes)
                    })
            except Exception as e:
                print(f"  Error analyzing k={k}: {e}")
                continue
        
        return shape_metrics
    
    def calculate_direction_variance(self):
        """Calculate variance consistency across directions"""
        X = self.scaled_data
        
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        total_variance = np.sum(eigenvalues)
        variance_ratios = eigenvalues / total_variance
        
        isotropy_score = np.min(eigenvalues) / np.max(eigenvalues) if np.max(eigenvalues) > 0 else 0
        
        direction_analysis = {
            'variance_ratios': variance_ratios,
            'isotropy_score': isotropy_score,
            'max_variance_direction': eigenvectors[:, 0],
            'min_variance_direction': eigenvectors[:, -1],
            'variance_range': np.max(eigenvalues) - np.min(eigenvalues)
        }
        
        return direction_analysis
    
    def create_individual_sphericity_plots(self, distances_to_center, eigenvalues, direction_analysis):
        """Create individual research-quality plots for spherical analysis"""
        
        X = self.scaled_data
        
        # 1. Distance to Centroid Distribution
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(distances_to_center, bins=50, alpha=0.7, 
                                   color='steelblue', density=True, edgecolor='black', linewidth=0.5)
        plt.axvline(np.mean(distances_to_center), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(distances_to_center):.3f}')
        plt.axvline(np.median(distances_to_center), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(distances_to_center):.3f}')
        
        plt.xlabel('Distance to Centroid', fontsize=12, fontweight='bold')
        plt.ylabel('Probability Density', fontsize=12, fontweight='bold')
        plt.title('Distribution of Distances to Data Centroid', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'distance_to_centroid_distribution.png')
        
        # 2. Eigenvalue Distribution
        plt.figure(figsize=(10, 6))
        components = range(1, len(eigenvalues) + 1)
        bars = plt.bar(components, eigenvalues, alpha=0.7, color='coral', edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Principal Direction', fontsize=12, fontweight='bold')
        plt.ylabel('Eigenvalue (Variance)', fontsize=12, fontweight='bold')
        plt.title('Covariance Matrix Eigenvalue Distribution\n(Variance by Principal Direction)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(components)
        
        self.save_plot(plt, 'eigenvalue_distribution.png')
        
        # 3. Directional Variance Ratios
        plt.figure(figsize=(10, 6))
        variance_ratios = direction_analysis['variance_ratios']
        directions = range(1, len(variance_ratios) + 1)
        
        bars = plt.bar(directions, variance_ratios, alpha=0.7, color='lightgreen', edgecolor='black')
        uniform_ref = 1/len(variance_ratios)
        plt.axhline(y=uniform_ref, color='red', linestyle='--', linewidth=2,
                   label=f'Uniform Reference: {uniform_ref:.3f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Principal Direction', fontsize=12, fontweight='bold')
        plt.ylabel('Variance Ratio', fontsize=12, fontweight='bold')
        plt.title('Directional Variance Ratios\n(Measure of Isotropy)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(directions)
        
        self.save_plot(plt, 'directional_variance_ratios.png')
        
        # 4. Enhanced PCA Scatter with Distance Coloring
        if hasattr(self, 'pca_result'):
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                                alpha=0.7, s=30, c=distances_to_center, 
                                cmap='viridis', edgecolor='white', linewidth=0.3)
            
            plt.xlabel('Principal Component 1', fontsize=12, fontweight='bold')
            plt.ylabel('Principal Component 2', fontsize=12, fontweight='bold')
            plt.title('PCA Projection Colored by Distance to Centroid', 
                     fontsize=14, fontweight='bold')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Distance to Centroid', fontsize=11, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            self.save_plot(plt, 'pca_distance_colored.png')
        
        # 5. Q-Q Plot for Multivariate Normality
        plt.figure(figsize=(8, 8))
        stats.probplot(distances_to_center, dist="norm", plot=plt)
        plt.xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
        plt.ylabel('Ordered Distances to Centroid', fontsize=12, fontweight='bold')
        plt.title('Q-Q Plot: Testing Multivariate Normality\n(Spherical Clusters Approach Normal Distribution)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        self.save_plot(plt, 'qq_plot_multivariate_normality.png')
        
        # 6. Polar Plot of Directional Variance
        if len(variance_ratios) >= 2:
            plt.figure(figsize=(8, 8))
            theta = np.linspace(0, 2*np.pi, len(variance_ratios), endpoint=False)
            radii = variance_ratios / np.max(variance_ratios)
            
            theta = np.concatenate([theta, [theta[0]]])
            radii = np.concatenate([radii, [radii[0]]])
            
            ax = plt.subplot(111, projection='polar')
            ax.plot(theta, radii, 'o-', linewidth=2.5, markersize=8, color='purple')
            ax.fill(theta, radii, alpha=0.3, color='purple')
            ax.set_title('Directional Variance Polar Plot\n(Circular Shape Indicates Spherical Distribution)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
            
            self.save_plot(plt, 'variance_polar_plot.png')
    
    def interpret_spherical_tendency(self, hopkins_stat, sphericity_metrics, shape_metrics, direction_analysis):
        """Interpret spherical clustering tendency"""
        # Basic clustering assessment
        if hopkins_stat > 0.75:
            base_assessment = "Strong Clustering Tendency"
        elif hopkins_stat > 0.65:
            base_assessment = "Moderate Clustering Tendency"
        elif hopkins_stat > 0.5:
            base_assessment = "Weak Clustering Tendency"
        else:
            base_assessment = "No Significant Clustering"
        
        # Sphericity assessment
        eigenvalue_ratio = sphericity_metrics['eigenvalue_ratio']
        distance_cv = sphericity_metrics['distance_cv_to_center']
        isotropy_score = direction_analysis['isotropy_score']
        
        # Sphericity judgment
        if eigenvalue_ratio < 2 and distance_cv < 0.3 and isotropy_score > 0.5:
            spherical_assessment = "High Sphericity"
            spherical_confidence = "High"
            spherical_recommendation = "Data exhibits clear spherical cluster structure, highly suitable for K-means and other distance-based clustering algorithms"
        elif eigenvalue_ratio < 3 and distance_cv < 0.5 and isotropy_score > 0.3:
            spherical_assessment = "Moderate Sphericity"
            spherical_confidence = "Medium"
            spherical_recommendation = "Data shows some spherical characteristics, K-means may be effective"
        elif eigenvalue_ratio < 5 and distance_cv < 0.7:
            spherical_assessment = "Low Sphericity"
            spherical_confidence = "Low"
            spherical_recommendation = "Data shows weak spherical features, consider DBSCAN, spectral clustering, or other non-spherical algorithms"
        else:
            spherical_assessment = "Non-Spherical"
            spherical_confidence = "High"
            spherical_recommendation = "Data clearly exhibits non-spherical distribution, avoid K-means and other spherical-assuming algorithms"
        
        # Shape distribution analysis
        if shape_metrics:
            avg_sphericity = np.mean([m['avg_sphericity'] for m in shape_metrics])
            if avg_sphericity > 0.7:
                shape_assessment = "Potential clusters are mostly spherical"
            elif avg_sphericity > 0.5:
                shape_assessment = "Mixed cluster shapes"
            else:
                shape_assessment = "Potential clusters are mostly non-spherical"
        else:
            shape_assessment = "Unable to analyze potential cluster shapes"
            avg_sphericity = 0
        
        result = {
            'base_assessment': base_assessment,
            'spherical_assessment': spherical_assessment,
            'shape_assessment': shape_assessment,
            'confidence': spherical_confidence,
            'recommendation': spherical_recommendation,
            'hopkins_statistic': hopkins_stat,
            'eigenvalue_ratio': eigenvalue_ratio,
            'distance_cv': distance_cv,
            'isotropy_score': isotropy_score,
            'avg_potential_sphericity': avg_sphericity,
            'sphericity_score': sphericity_metrics['sphericity_score']
        }
        
        # Print detailed results
        print(f"\nSpherical Cluster Analysis Detailed Results:")
        print(f"- Hopkins Statistic: {hopkins_stat:.4f} ({base_assessment})")
        print(f"- Eigenvalue Ratio: {eigenvalue_ratio:.4f} (measures anisotropy)")
        print(f"- Distance Coefficient of Variation: {distance_cv:.4f} (measures sphericity)")
        print(f"- Isotropy Score: {isotropy_score:.4f}")
        print(f"- Comprehensive Sphericity Score: {sphericity_metrics['sphericity_score']:.4f}")
        print(f"- Sphericity Assessment: {spherical_assessment}")
        print(f"- Potential Cluster Shapes: {shape_assessment}")
        print(f"- Confidence Level: {spherical_confidence}")
        print(f"- Recommendation: {spherical_recommendation}")
        
        # Sphericity interpretation
        print(f"\nSphericity Metrics Interpretation:")
        print(f"  Eigenvalue Ratio = {eigenvalue_ratio:.3f}: ", end="")
        if eigenvalue_ratio < 2:
            print("Relatively uniform variance across directions")
        elif eigenvalue_ratio < 5:
            print("Some directional preference exists")
        else:
            print("Strong dominant direction present")
            
        print(f"  Distance CV = {distance_cv:.3f}: ", end="")
        if distance_cv < 0.3:
            print("Concentrated distance distribution, near-spherical")
        elif distance_cv < 0.5:
            print("Moderate distance variation")
        else:
            print("High distance variation, non-spherical")
        
        return result
    
    def perform_spherical_analysis(self):
        """Perform comprehensive spherical cluster analysis"""
        print("\nStarting spherical cluster analysis...")
        start_time = time.time()
        
        # Calculate Hopkins statistic
        print("  Calculating clustering tendency...")
        hopkins_stat = self.calculate_hopkins_statistic()
        
        # Calculate sphericity metrics
        print("  Calculating sphericity metrics...")
        sphericity_metrics, distances_to_center, eigenvalues = self.calculate_sphericity_metrics()
        
        # Analyze directional variance
        print("  Analyzing directional variance...")
        direction_analysis = self.calculate_direction_variance()
        
        # Analyze potential cluster shapes
        print("  Analyzing potential cluster shapes...")
        shape_metrics = self.analyze_cluster_shape_distribution()
        
        # Create individual visualizations
        print("  Generating research-quality visualizations...")
        self.create_individual_sphericity_plots(distances_to_center, eigenvalues, direction_analysis)
        
        # Interpret results
        print("  Interpreting analysis results...")
        analysis_result = self.interpret_spherical_tendency(
            hopkins_stat, sphericity_metrics, shape_metrics, direction_analysis
        )
        
        # Save detailed results
        detailed_results = {
            'hopkins_statistic': hopkins_stat,
            'eigenvalue_ratio': sphericity_metrics['eigenvalue_ratio'],
            'eigenvalue_cv': sphericity_metrics['eigenvalue_cv'],
            'distance_cv_to_center': sphericity_metrics['distance_cv_to_center'],
            'variance_entropy': sphericity_metrics['variance_entropy'],
            'isotropy_score': direction_analysis['isotropy_score'],
            'sphericity_score': sphericity_metrics['sphericity_score'],
            'clustering_tendency_assessment': analysis_result['base_assessment'],
            'sphericity_assessment': analysis_result['spherical_assessment'],
            'potential_cluster_shapes': analysis_result['shape_assessment'],
            'confidence_level': analysis_result['confidence'],
            'algorithm_recommendation': analysis_result['recommendation'],
            'sample_count': len(self.scaled_data),
            'feature_count': len(self.load_related_features)
        }
        
        # Add shape analysis results
        if shape_metrics:
            for i, metric in enumerate(shape_metrics):
                detailed_results[f'potential_clusters_{i+1}_avg_sphericity'] = metric['avg_sphericity']
                detailed_results[f'potential_clusters_{i+1}_eigen_ratio'] = metric['avg_eigen_ratio']
        
        results_df = pd.DataFrame([detailed_results])
        self.save_table(results_df, 'spherical_cluster_analysis_results.csv')
        
        # Create summary report
        self.create_spherical_summary_report(analysis_result, detailed_results)
        
        elapsed_time = time.time() - start_time
        print(f"Spherical cluster analysis completed, time: {elapsed_time:.2f}s")
        
        return results_df
    
    def create_spherical_summary_report(self, analysis_result, detailed_results):
        """Create comprehensive spherical analysis report"""
        report_content = f"""
# Spherical Cluster Analysis Report

## Basic Information
- Analysis Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Data Samples: {detailed_results['sample_count']}
- Data Features: {detailed_results['feature_count']}
- Output Directory: {os.path.abspath(self.output_folder)}

## Key Metric Results

### 1. Clustering Tendency Detection
- **Hopkins Statistic**: {detailed_results['hopkins_statistic']:.4f}
- **Assessment**: {analysis_result['base_assessment']}

### 2. Sphericity Metrics
- **Eigenvalue Ratio**: {detailed_results['eigenvalue_ratio']:.4f} (measures anisotropy, lower is better)
- **Distance Coefficient of Variation**: {detailed_results['distance_cv_to_center']:.4f} (measures sphericity, lower is better)
- **Isotropy Score**: {detailed_results['isotropy_score']:.4f} (measures directional uniformity, higher is better)
- **Comprehensive Sphericity Score**: {detailed_results['sphericity_score']:.4f}

### 3. Comprehensive Assessment
**Sphericity Assessment**: {analysis_result['spherical_assessment']}
**Confidence Level**: {analysis_result['confidence']}
**Potential Cluster Shapes**: {analysis_result['shape_assessment']}

## Algorithm Recommendation
{analysis_result['recommendation']}

## Metric Interpretation Guidelines

### Eigenvalue Ratio
- < 2: Uniform variance across directions, near-spherical
- 2-5: Some directional preference
- > 5: Strong dominant direction, non-spherical

### Distance Coefficient of Variation  
- < 0.3: Concentrated distance distribution, near-spherical
- 0.3-0.5: Moderate distance variation
- > 0.5: High distance variation, non-spherical

### Isotropy Score
- > 0.5: Relatively uniform variance across directions
- 0.3-0.5: Some directional preference
- < 0.3: Significant directional variance differences

## Current Data Assessment
Based on the analysis results, your data exhibits:

**Hopkins Statistic** {detailed_results['hopkins_statistic']:.4f} indicates {analysis_result['base_assessment'].lower()}
**Eigenvalue Ratio** {detailed_results['eigenvalue_ratio']:.4f} indicates {['relatively uniform variance across directions', 'some directional preference', 'strong dominant direction'][(detailed_results['eigenvalue_ratio'] > 2) + (detailed_results['eigenvalue_ratio'] > 5)]}
**Distance Coefficient of Variation** {detailed_results['distance_cv_to_center']:.4f} indicates {['concentrated distance distribution, near-spherical', 'moderate distance variation', 'high distance variation, non-spherical'][(detailed_results['distance_cv_to_center'] > 0.3) + (detailed_results['distance_cv_to_center'] > 0.5)]}

**Overall Conclusion**: Data {['exhibits clear spherical cluster characteristics', 'shows some spherical characteristics', 'shows weak spherical features', 'clearly exhibits non-spherical distribution'][(detailed_results['hopkins_statistic'] < 0.5) + (detailed_results['eigenvalue_ratio'] > 3) + (detailed_results['distance_cv_to_center'] > 0.5)]}

## Generated Visualizations
The analysis generated the following research-quality plots:
1. PCA Variance Explained - Shows variance captured by principal components
2. PCA Scatter Plot - Visualizes data distribution in reduced space
3. PCA Component Weights - Displays feature contributions to components
4. Distance to Centroid Distribution - Examines concentration around center
5. Eigenvalue Distribution - Shows variance by principal direction
6. Directional Variance Ratios - Displays variance distribution across directions
7. PCA Distance-Colored - Colors PCA plot by distance to centroid
8. Q-Q Plot - Tests multivariate normality assumption
9. Variance Polar Plot - Circular visualization of directional variance
"""
        
        report_path = os.path.join(self.subfolders['reports'], 'spherical_cluster_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nDetailed report saved: {report_path}")
        
        # Print key information to console
        print("\n" + "="*60)
        print("SPHERICAL CLUSTER ANALYSIS SUMMARY")
        print("="*60)
        print(f"Clustering Tendency: {analysis_result['base_assessment']}")
        print(f"Sphericity Assessment: {analysis_result['spherical_assessment']}")
        print(f"Confidence Level: {analysis_result['confidence']}")
        print(f"Recommendation: {analysis_result['recommendation']}")
        print("="*60)
    
    def run_spherical_analysis(self):
        """Execute complete spherical cluster analysis pipeline"""
        try:
            print("Starting Spherical Cluster Analysis")
            print("=" * 60)
            total_start_time = time.time()
            
            # 1. Data loading and preprocessing
            self.load_and_preprocess_data()
            
            # 2. PCA analysis for visualization
            self.perform_pca_analysis()
            
            # 3. Spherical cluster analysis
            analysis_results = self.perform_spherical_analysis()
            
            total_elapsed_time = time.time() - total_start_time
            
            print("\n" + "=" * 60)
            print(f"Spherical Cluster Analysis Completed! Total time: {total_elapsed_time:.2f}s")
            print(f"All results saved to: {os.path.abspath(self.output_folder)}")
            
            # Final recommendation based on results
            spherical_score = analysis_results.iloc[0]['sphericity_score']
            if spherical_score > 0.7:
                print("\n🎯 CONCLUSION: Data exhibits strong spherical characteristics, highly recommend K-means!")
            elif spherical_score > 0.5:
                print("\n✅ CONCLUSION: Data shows moderate spherical features, K-means may be suitable.")
            elif spherical_score > 0.3:
                print("\n⚠️  CONCLUSION: Data shows weak spherical features, consider alternative clustering algorithms.")
            else:
                print("\n❌ CONCLUSION: Data clearly exhibits non-spherical distribution, avoid K-means.")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()

# Usage example
if __name__ == "__main__":
    folder_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\trip数据"
    output_folder = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\spherical_cluster_analysis"
    
    # Initialize spherical cluster analyzer
    analyzer = SphericalClusterAnalyzer(folder_path, output_folder)
    
    # Execute spherical cluster analysis
    analyzer.run_spherical_analysis()