import pandas as pd
import numpy as np
import os
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TripDataProcessor:
    def __init__(self, folder1_path, folder2_path, folder3_path):
        """
        🚀 Initialize the Data Processing Adventure!
        
        Parameters:
        folder1_path: Path to Folder 1 (contains license plates, trip IDs, load status)
        folder2_path: Path to Folder 2 (contains detailed operational data)
        folder3_path: Path to Folder 3 (output destination for processed results)
        """
        self.folder1_path = folder1_path
        self.folder2_path = folder2_path
        self.folder3_path = folder3_path
        
        # Create the output playground
        os.makedirs(folder3_path, exist_ok=True)
        
        # Our magical column transformation dictionary
        self.selected_columns_mapping = {
            '发动机燃料流量平均值/Average fuel flow': 'Fuel Consumption Rate',
            '发动机转速平均值/Average engine speed': 'Engine Speed', 
            '摩擦扭矩平均值/Average friction torque': 'Friction Torque',
            '车速平均值/Average speed': 'Vehicle Speed'
        }
        
        # File matching treasure map
        self.file_mapping = {}

    def load_and_combine_folder2_data(self):
        """
        🎯 Mission: Gather all data treasures from Folder 2
        """
        print("🔍 Scanning Folder 2 for data treasures...")
        folder2_files = glob.glob(os.path.join(self.folder2_path, "*.pkl"))
        print(f"🎁 Found {len(folder2_files)} potential data treasures!")
        
        all_data = []
        file_info = []
        
        for file_path in folder2_files:
            try:
                # Extract file fingerprint (first 7 characters)
                file_id = os.path.basename(file_path)[:7]
                df = pd.read_pickle(file_path)
                
                # Check if this treasure has the right jewels
                available_columns = []
                for col in self.selected_columns_mapping.keys():
                    if col in df.columns:
                        available_columns.append(col)
                
                if len(available_columns) < 2:
                    print(f"⚠️  File {file_id} missing essential jewels, skipping...")
                    continue
                
                # Select our precious data gems
                selected_data = df[available_columns].copy()
                selected_data['file_id'] = file_id
                selected_data['original_index'] = selected_data.index
                
                all_data.append(selected_data)
                file_info.append({
                    'file_id': file_id,
                    'file_path': file_path,
                    'row_count': len(selected_data),
                    'available_columns': available_columns
                })
                
                # Update our treasure map
                self.file_mapping[file_id] = {
                    'folder2_path': file_path,
                    'folder1_path': None  # To be discovered later
                }
                
            except Exception as e:
                print(f"💥 Oops! Trouble loading {file_path}: {e}")
        
        if not all_data:
            raise ValueError("😞 No data treasures found! Mission aborted.")
        
        # Combine all our treasures into one big chest
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined treasure chest size: {combined_data.shape}")
        
        return combined_data, file_info

    def perform_pca_kmeans(self, data):
        """
        🧙‍♂️ Perform the Magic: PCA & K-Means Sorcery
        """
        print("\n✨ Beginning the magical transformation...")
        
        # Prepare our magical ingredients
        feature_columns = [col for col in self.selected_columns_mapping.keys() 
                          if col in data.columns]
        
        X = data[feature_columns].copy()
        
        # Cleanse our data of any impurities
        X = X.dropna()
        
        if len(X) == 0:
            raise ValueError("😱 Not enough magical energy for the spell!")
        
        print(f"🧪 Magical ingredients prepared: {X.shape}")
        
        # Standardize our magical components
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction magic (PCA)
        print("🌀 Casting PCA spell...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"🌈 PCA magical power distribution: {pca.explained_variance_ratio_}")
        print(f"⚡ Total magical power captured: {sum(pca.explained_variance_ratio_):.3f}")
        
        # Cluster discovery magic (K-Means)
        print("🔮 Summoning K-Means clusters...")
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        
        # Evaluate our magical creation
        silhouette_avg = silhouette_score(X_pca, clusters)
        print(f"🎭 Cluster harmony score (Silhouette): {silhouette_avg:.3f}")
        
        # Annotate our original data with magical insights
        result_data = data.loc[X.index].copy()
        result_data['cluster'] = clusters
        result_data['cluster_pca'] = clusters
        
        # Determine load status based on fuel consumption patterns
        fuel_col = '发动机燃料流量平均值/Average fuel flow'
        if fuel_col in result_data.columns:
            cluster_fuel_means = result_data.groupby('cluster')[fuel_col].mean()
            print(f"⛽ Fuel consumption by cluster:\n{cluster_fuel_means}")
            
            # Higher fuel consumption = loaded vehicles
            full_load_cluster = cluster_fuel_means.idxmax()
            result_data['load_status'] = result_data['cluster'].apply(
                lambda x: 'Loaded' if x == full_load_cluster else 'Empty'
            )
            print(f"🚛 Cluster {full_load_cluster} identified as Loaded vehicles")
        
        return result_data, pca, kmeans, scaler, silhouette_avg

    def save_cluster_results_to_folder2(self, clustered_data, file_info):
        """
        💾 Mission: Return enriched data treasures to their homes
        """
        print("\n💾 Returning enriched treasures to Folder 2...")
        
        current_index = 0
        for info in file_info:
            file_id = info['file_id']
            row_count = info['row_count']
            file_path = info['file_path']
            
            # Extract the enchanted data for this file
            file_cluster_data = clustered_data.iloc[current_index:current_index + row_count].copy()
            current_index += row_count
            
            if len(file_cluster_data) == 0:
                continue
                
            try:
                # Read the original treasure
                original_df = pd.read_pickle(file_path)
                
                # Add our magical enhancements
                original_df['cluster'] = file_cluster_data['cluster'].values
                original_df['load_status'] = file_cluster_data['load_status'].values
                
                # Save the enriched treasure
                output_path = file_path.replace('.pkl', '_clustered.pkl')
                original_df.to_pickle(output_path)
                print(f"💎 Enriched treasure saved: {output_path}")
                
            except Exception as e:
                print(f"💥 Trouble saving {file_path}: {e}")

    def process_folder1_data(self):
        """
        🕵️‍♂️ Mission: Find matching treasures in Folder 1
        """
        print("\n🕵️‍♂️ Exploring Folder 1 for matching treasures...")
        
        folder1_files = glob.glob(os.path.join(self.folder1_path, "*.pkl"))
        print(f"🎯 Found {len(folder1_files)} potential matches in Folder 1")
        
        # Build our matching network
        for file_path in folder1_files:
            file_id = os.path.basename(file_path)[:7]
            if file_id in self.file_mapping:
                self.file_mapping[file_id]['folder1_path'] = file_path
        
        all_folder1_results = []
        
        for file_id, mapping in self.file_mapping.items():
            if mapping['folder1_path'] is None:
                print(f"⚠️  No match found for {file_id} in Folder 1")
                continue
                
            try:
                # Uncover the Folder 1 treasure
                df1 = pd.read_pickle(mapping['folder1_path'])
                
                # Focus only on the essential information
                required_cols = ['车牌号', '变化次数', '满载/空载']
                available_cols = [col for col in required_cols if col in df1.columns]
                
                if len(available_cols) < 3:
                    print(f"⚠️  File {file_id} missing essential info, skipping")
                    continue
                
                df1_filtered = df1[available_cols].copy()
                
                # Find the matching enriched treasure from Folder 2
                cluster_file_path = mapping['folder2_path'].replace('.pkl', '_clustered.pkl')
                if not os.path.exists(cluster_file_path):
                    print(f"⚠️  No enriched data found for {file_id}")
                    continue
                
                df2_clustered = pd.read_pickle(cluster_file_path)
                
                # Find the trip ID column (it might be hiding under different names)
                trip_id_col = None
                for col in ['Trip编号/Trip ID', 'trip_id', 'TripID', '行程编号']:
                    if col in df2_clustered.columns:
                        trip_id_col = col
                        break
                
                if trip_id_col is None:
                    print(f"⚠️  Couldn't find trip ID column in {file_id}")
                    continue
                
                # Create our magical matching dictionaries
                trip_cluster_map = df2_clustered.set_index(trip_id_col)['cluster'].to_dict()
                trip_load_map = df2_clustered.set_index(trip_id_col)['load_status'].to_dict()
                
                # Apply our magical insights to Folder 1 data
                df1_filtered['cluster'] = df1_filtered['变化次数'].map(trip_cluster_map)
                df1_filtered['load_status_cluster'] = df1_filtered['变化次数'].map(trip_load_map)
                
                # Store our unified treasure
                output_filename = f"{file_id}_result.pkl"
                output_path = os.path.join(self.folder3_path, output_filename)
                df1_filtered.to_pickle(output_path)
                
                print(f"🎉 Unified treasure created: {output_path}")
                all_folder1_results.append(df1_filtered)
                
            except Exception as e:
                print(f"💥 Trouble processing {file_id}: {e}")
        
        return all_folder1_results

    def generate_summary_report(self, silhouette_score, all_folder1_results):
        """
        📊 Mission: Create our adventure summary report
        """
        print("\n" + "="*50)
        print("📊 ADVENTURE SUMMARY REPORT")
        print("="*50)
        print(f"🎭 Cluster Harmony Score: {silhouette_score:.3f}")
        print(f"🤝 Successful File Matches: {len([x for x in self.file_mapping.values() if x['folder1_path'] is not None])}")
        
        if all_folder1_results:
            combined_results = pd.concat(all_folder1_results, ignore_index=True)
            cluster_counts = combined_results['cluster'].value_counts().sort_index()
            load_status_counts = combined_results['load_status_cluster'].value_counts()
            
            print(f"\n📈 Cluster Distribution:")
            for cluster, count in cluster_counts.items():
                print(f"   Cluster {cluster}: {count} records")
            
            print(f"\n🚛 Load Status Distribution:")
            for status, count in load_status_counts.items():
                print(f"   {status}: {count} records")

    def run_full_processing(self):
        """
        🚀 Launch the Complete Data Processing Adventure!
        """
        print("🚀 Launching the Data Processing Adventure!")
        
        try:
            # Phase 1: Gather Folder 2 treasures
            combined_data, file_info = self.load_and_combine_folder2_data()
            
            # Phase 2: Perform magical transformations
            clustered_data, pca, kmeans, scaler, silhouette_avg = self.perform_pca_kmeans(combined_data)
            
            # Phase 3: Return enriched treasures
            self.save_cluster_results_to_folder2(clustered_data, file_info)
            
            # Phase 4: Find and enrich Folder 1 matches
            all_folder1_results = self.process_folder1_data()
            
            # Phase 5: Create our adventure report
            self.generate_summary_report(silhouette_avg, all_folder1_results)
            
            print("\n🎉 Adventure Completed Successfully!")
            
            return {
                'pca': pca,
                'kmeans': kmeans,
                'scaler': scaler,
                'silhouette_score': silhouette_avg,
                'processed_files': len(file_info)
            }
            
        except Exception as e:
            print(f"💥 Adventure Failed: {e}")
            return None

# 🎮 Let the adventure begin!
if __name__ == "__main__":
    # Set your adventure coordinates
    folder1_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\分trip\清洗后数据重置分trip(删去trip不足5km版本)"
    folder2_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\trip数据" 
    folder3_path = r"D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\对比结果"
    
    # Create your adventure guide and begin!
    processor = TripDataProcessor(folder1_path, folder2_path, folder3_path)
    result = processor.run_full_processing()