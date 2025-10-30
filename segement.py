import pandas as pd
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging with fun ASCII art
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("""
🚛 HEAVY-DUTY TRUCK DATA SEGMENTATION WIZARD 🚛
============================================
Starting the magical journey of transforming raw data into clean trips...
✨ Let the data segmentation magic begin! ✨
""")

def process_pkl(file_path, time_threshold=15):
    try:
        # 🎯 STEP 0: LOAD THE DATA TREASURE CHEST
        logging.info("🔓 Unlocking the data treasure chest...")
        df = pd.read_pickle(file_path)
        
        df.columns = df.columns.str.strip()
        logging.info(f"🧭 Navigating through file: {file_path}")
        
        if '上报时间' not in df.columns or '车速' not in df.columns:
            raise ValueError(f"❌ Critical columns missing in {file_path}")

        # 🕒 STEP 1: TIME-BASED SEGMENTATION - The Chronomancer's Spell
        logging.info("🕒 Casting time-based segmentation spell...")
        df['上报时间'] = pd.to_datetime(df['上报时间'], format='%Y/%m/%d %H:%M:%S')

        df['时间间隔'] = df['上报时间'].diff().dt.total_seconds().fillna(0)
        df['分段编号'] = (df['时间间隔'] > time_threshold*60).cumsum() + 1

        # 📊 VISUALIZATION 1: Time Gaps Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['时间间隔']/60, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(time_threshold, color='red', linestyle='--', label=f'Threshold: {time_threshold} min')
        plt.xlabel('Time Gap Between Records (minutes)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Time Intervals Between Consecutive Data Points')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{file_path.replace(".pkl", "_time_gaps.png")}', dpi=300, bbox_inches='tight')
        plt.close()

        # 🐌 STEP 2: LOW-SPEED DURATION FILTERING - The Snail Hunter
        logging.info("🐌 Hunting for extended low-speed periods...")
        df['is_speed_under_1'] = (df['车速'] < 1).astype(int)
        df['速度小于1的时间'] = df['时间间隔'] * df['is_speed_under_1']
        df['低速段分组'] = (df['is_speed_under_1'] == 0).cumsum()

        segment_sums = df[df['is_speed_under_1'] == 1].groupby('低速段分组')['速度小于1的时间'].sum()
        valid_segments = segment_sums[segment_sums > 1800].index
        df['标记列'] = np.where((df['is_speed_under_1'] == 1) & 
                            (df['低速段分组'].isin(valid_segments)), 'xxx', '')

        df.drop(['低速段分组'], axis=1, inplace=True)

        # 📊 VISUALIZATION 2: Speed Profile with Low-Speed Annotations
        plt.figure(figsize=(12, 6))
        plt.plot(df['上报时间'], df['车速'], 'b-', alpha=0.7, linewidth=0.8, label='Vehicle Speed')
        
        # Mark low-speed segments
        low_speed_mask = df['is_speed_under_1'] == 1
        plt.scatter(df[low_speed_mask]['上报时间'], df[low_speed_mask]['车速'], 
                   c='orange', s=10, alpha=0.6, label='Low Speed (<1 km/h)')
        
        # Mark extended low-speed segments for removal
        extended_low_mask = df['标记列'] == 'xxx'
        plt.scatter(df[extended_low_mask]['上报时间'], df[extended_low_mask]['车速'], 
                   c='red', s=20, alpha=0.8, label='Extended Low Speed (To Remove)')
        
        plt.xlabel('Timestamp')
        plt.ylabel('Vehicle Speed (km/h)')
        plt.title('Vehicle Speed Profile with Low-Speed Annotations')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{file_path.replace(".pkl", "_speed_profile.png")}', dpi=300, bbox_inches='tight')
        plt.close()

        # 🎪 STEP 3: SUB-SEGMENTATION MASTERPIECE - The Data Sculptor
        logging.info("🎪 Sculpting fine-grained sub-segments...")
        df['子编号'] = None

        for segment in df['分段编号'].unique():
            segment_mask = df['分段编号'] == segment
            segment_data = df[segment_mask]
            
            is_xxx = segment_data['标记列'] == 'xxx'
            is_boundary = is_xxx | (is_xxx.shift(1, fill_value=False))
            is_boundary.iloc[0] = True
        
            sub_labels = is_boundary.cumsum()
            df.loc[segment_mask, '子编号'] = [f"{segment}.{label}" for label in sub_labels]
        
            xxx_mask = segment_mask & is_xxx
            df.loc[xxx_mask, '子编号'] = df.loc[xxx_mask, '子编号'] + '.xxx'

        # 🧹 STEP 4: DATA PURIFICATION - The Cleanliness Crusader
        logging.info("🧹 Wielding the broom of data purification...")
        df_clean = df[~df['子编号'].str.contains('xxx', na=False)].copy()
        df_clean = df_clean.sort_index()  
        df_clean['最终编号'] = (df_clean['子编号'] != df_clean['子编号'].shift(1)).cumsum()

        # 📊 VISUALIZATION 3: Final Trip Segmentation
        plt.figure(figsize=(14, 8))
        
        # Color map for different trips
        unique_trips = df_clean['最终编号'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_trips)))
        
        for i, trip_id in enumerate(unique_trips[:10]):  # Show first 10 trips for clarity
            trip_data = df_clean[df_clean['最终编号'] == trip_id]
            plt.plot(trip_data['上报时间'], trip_data['车速'], 
                    color=colors[i], linewidth=2, label=f'Trip {trip_id}')
        
        plt.xlabel('Timestamp')
        plt.ylabel('Vehicle Speed (km/h)')
        plt.title('Final Trip Segmentation Results\n(Different Colors Represent Different Trips)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{file_path.replace(".pkl", "_final_trips.png")}', dpi=300, bbox_inches='tight')
        plt.close()

        # 📊 VISUALIZATION 4: Trip Statistics Overview
        trip_stats = df_clean.groupby('最终编号').agg({
            '上报时间': ['count', lambda x: (x.max() - x.min()).total_seconds()/60],
            '车速': ['mean', 'max', 'std']
        }).round(2)
        
        trip_stats.columns = ['Data Points', 'Duration (min)', 'Avg Speed', 'Max Speed', 'Speed Std']
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(trip_stats['Duration (min)'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Trip Duration (minutes)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Trip Durations')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(trip_stats['Avg Speed'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Average Speed (km/h)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Average Speeds')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(trip_stats['Data Points'], bins=20, alpha=0.7, color='gold', edgecolor='black')
        plt.xlabel('Number of Data Points per Trip')
        plt.ylabel('Frequency')
        plt.title('Distribution of Data Points per Trip')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.scatter(trip_stats['Duration (min)'], trip_stats['Avg Speed'], alpha=0.6, c='purple')
        plt.xlabel('Trip Duration (minutes)')
        plt.ylabel('Average Speed (km/h)')
        plt.title('Trip Duration vs Average Speed')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{file_path.replace(".pkl", "_trip_statistics.png")}', dpi=300, bbox_inches='tight')
        plt.close()

        # 💾 STEP 5: SAVE THE POLISHED GEM
        output_path = file_path.replace('.pkl', '_spilt_trip.pkl')
        df_clean.to_pickle(output_path)
        logging.info(f"💎 Polished data gem saved to: {output_path}")
        
        # 🎉 SUCCESS CELEBRATION
        logging.info(f"🎉 Successfully processed! Created {len(unique_trips)} clean trips from raw data!")
        
        return df_clean
    
    except Exception as e:
        logging.error(f"💥 Catastrophic failure processing {file_path}: {e}")
        return None

# 🗂️ MAIN EXECUTION - The Grand Data Expedition
file_directory = r'D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\分trip\清洗后数据重置分trip预备数据'

processed_count = 0
for filename in os.listdir(file_directory):
    if filename.endswith('.pkl'):  
        file_path = os.path.join(file_directory, filename)
        logging.info(f"\n{'='*60}")
        logging.info(f"🚀 Launching expedition for: {filename}")
        logging.info(f"{'='*60}")
        
        result = process_pkl(file_path)
        if result is not None:
            processed_count += 1

logging.info(f"\n{'🎊'*20}")
logging.info(f"GRAND EXPEDITION COMPLETE!")
logging.info(f"Successfully processed {processed_count} files!")
logging.info(f"Generated beautiful scientific visualizations for analysis!")
logging.info(f"{'🎊'*20}")