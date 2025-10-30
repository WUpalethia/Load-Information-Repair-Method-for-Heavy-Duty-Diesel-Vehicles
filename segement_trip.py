import pandas as pd
import os
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#time_threshold表示设置的间隔时间大于15分钟将会造成分段
def process_pkl(file_path, time_threshold=15):
    try:
        df = pd.read_pickle(file_path)
        
        df.columns = df.columns.str.strip()
        logging.info(f"Processing file: {file_path}")
        
        if '上报时间' not in df.columns or '车速' not in df.columns:
            raise ValueError(f"Required columns not found in {file_path}")
        
        # step1：时间间隔大于15分钟的分段标签
        # 将上报时间转换为时间的形式
        df['上报时间'] = pd.to_datetime(df['上报时间'], format='%Y/%m/%d %H:%M:%S')

        df['时间间隔'] = df['上报时间'].diff().dt.total_seconds().fillna(0)
        df['分段编号'] = (df['时间间隔'] > time_threshold*60).cumsum() + 1 #使用cumsum纵向求和进行优化

        # step2:开始处理连续低速时间超过1800秒的分段
        # 标记低速时段
        df['is_speed_under_1'] = (df['车速'] < 1).astype(int)
        # 计算低速持续时间
        df['速度小于1的时间'] = df['时间间隔'] * df['is_speed_under_1']
        # 识别连续低速段
        df['低速段分组'] = (df['is_speed_under_1'] == 0).cumsum()

        # 计算每个连续低速段的总持续时间
        segment_sums = df[df['is_speed_under_1'] == 1].groupby('低速段分组')['速度小于1的时间'].sum()

        # 标记总时长>1800秒的段
        valid_segments = segment_sums[segment_sums > 1800].index
        df['标记列'] = np.where((df['is_speed_under_1'] == 1) & 
                            (df['低速段分组'].isin(valid_segments)), 'xxx', '')

        # 清理临时列
        df.drop(['低速段分组'], axis=1, inplace=True)

        #step3:
        # 初始化子编号列
        df['子编号'] = None

        # 记得要遍历每个分段编号
        for segment in df['分段编号'].unique():
            segment_mask = df['分段编号'] == segment
            segment_data = df[segment_mask]
            
            # 在分段内根据xxx标记进行再分段
            # 当遇到xxx或从xxx变为非xxx时，都视为分段点
            is_xxx = segment_data['标记列'] == 'xxx'
            is_boundary = is_xxx | (is_xxx.shift(1, fill_value=False))  # 当前或前一行是xxx
        
            # 确保每个分段的开始也是一个边界
            is_boundary.iloc[0] = True
        
            # 生成子编号
            sub_labels = is_boundary.cumsum()
            df.loc[segment_mask, '子编号'] = [f"{segment}.{label}" for label in sub_labels]
        
            # 对于xxx标记的行，子编号添加'xxx'后缀
            xxx_mask = segment_mask & is_xxx
            df.loc[xxx_mask, '子编号'] = df.loc[xxx_mask, '子编号'] + '.xxx'



        # step3:生成最终编号
        # 删除所有子编号包含'xxx'的行
        df_clean = df[~df['子编号'].str.contains('xxx', na=False)].copy()

        # 按子编号变化生成最终编号
        df_clean = df_clean.sort_index()  
        df_clean['最终编号'] = (df_clean['子编号'] != df_clean['子编号'].shift(1)).cumsum()

       # 输出名字最后记得改一下，改为pkl格式
        output_path = file_path.replace('.pkl', '_spilt_trip.pkl')
        df_clean.to_pickle(output_path)
        logging.info(f"Processed file saved to {output_path}")
    
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

#文件名填这
file_directory = r'D:\seadrive\Yuxuan.W\共享资料库\ShanghaiTruck\1_intermediate\清洗后数据重置分trip预备数据' 


for filename in os.listdir(file_directory):
    if filename.endswith('.pkl'):  
        file_path = os.path.join(file_directory, filename)
        process_pkl(file_path)

logging.info("处理完成！")