import sklearn
import results_handler as rh
from sklearn import metrics
import numpy as np

class DatasetBuilder():
    
    def __init__(self):

    def convert_pcaps_to_csv():

    def shorten_attack_duration(df, timestamp_col, start_time):

        random_duration = pd.Timedelta(seconds=random.uniform(10, 30))
        if start_time == 0:
            start_time = df[timestamp_col].iloc[0]
            end_time = start_time + random_duration 
        if start_time != 0:
            df = df[df[timestamp_col] > start_time]
            start_time = df[timestamp_col].iloc[0]
            end_time = start_time + random_duration
            
        return end_time

    def select_insertion_point(df, timestamp_column, last_timestamp):
    
        start_time = last_timestamp
        end_time = start_time +  pd.Timedelta(minutes=random.uniform(1, 10))
    
        from_last_data = df[(df[timestamp_column] >= start_time) & (df[timestamp_column] < end_time)]
        
        if not from_last_data.empty:
            selected_row = from_last_data.sample(n=1)
            selected_index = selected_row.index[0]
            selected_timestamp = selected_row[timestamp_column].iloc[0]
            return selected_index, selected_timestamp
        else:
            raise ValueError("No data points found within the first hour.")
            
    def adjust_attack_timestamps_relative(attack_df, timestamp_column, insertion_timestamp):
        
        attack_df[timestamp_column] = pd.to_datetime(attack_df[timestamp_column], errors='coerce')
    
        if attack_df[timestamp_column].isnull().all():
            raise ValueError("No valid datetime values in the attack DataFrame.")
        
        time_deltas = attack_df[timestamp_column].diff().fillna(pd.Timedelta(seconds=0))
        new_timestamps = [insertion_timestamp]
        
        for delta in time_deltas[1:]:
            new_timestamps.append(new_timestamps[-1] + delta)
        
        attack_df[timestamp_column] = new_timestamps
        
        return attack_df, attack_df['layers.frame.frame.time'].max()

    def merge(normal_df, attack_df, timestamp_column, features):
    
        normal_df.reset_index(drop=True, inplace=True)
        attack_df.reset_index(drop=True, inplace=True)
        selected_columns = [col for col in features if col in attack_df.columns]
        
        merged_df = pd.concat([normal_df, attack_df[selected_columns]], ignore_index=True)
        merged_df = merged_df.sort_values(by=timestamp_column, ascending=True)
        merged_df.reset_index(drop=True, inplace=True)
    
        return merged_df


    def create_attack_sequences():
        
        out_path = './data/datasets/dos_mqtt_iot/train_set_new.csv'
        normal_df_train['label'] = 'normal'
        normal_df_train[timestamp] = normal_df_train[timestamp].str.replace(" CEST", "", regex=False)
        normal_df_train[timestamp] = pd.to_datetime(normal_df_train[timestamp], errors='coerce')
        
        last_timestamp = normal_df_train[timestamp].min()
        
        for i, file in enumerate(attack_files_train):
            print(i)
            #load attack csv, label, and convert to datetime
            attack_csv = attacks_root_train + file
            attack_df = pd.read_csv(attack_csv)
            print(attack_df.shape)  
            
            attack_df[timestamp] = attack_df[timestamp].str.replace(" CEST", "", regex=False)
            attack_df[timestamp] = pd.to_datetime(attack_df[timestamp], errors='coerce')
        
            # shorten attack csv to have a sequence of 30 to 180 seconds
            end_time_1 = shorten_attack_duration(attack_df, timestamp, 0)
            end_time_2 = shorten_attack_duration(attack_df, timestamp, end_time_1)
            attack_df_1 = attack_df[attack_df[timestamp] <= end_time_1]
            attack_df_2 = attack_df[(attack_df[timestamp] > end_time_1) & (attack_df[timestamp] <= end_time_2)]
            attack_df_1['label'] = file[:-6]
            attack_df_1['sequence'] = i
            attack_df_2['label'] = file[:-6]
            attack_df_2['sequence'] = i + 0.5
            print(attack_df.shape)
        
            #select a timestamp to merge attack sequence in normal and save last attack package
            insert_index, insert_timestamp = select_insertion_point(normal_df_train, timestamp, last_timestamp)
            
            old_last_timestamp = last_timestamp 
        
            print(old_last_timestamp)
            print(insert_timestamp)
            
            #merge and return last_timestamp of sequence
            attack_df_1, last_timestamp = adjust_attack_timestamps_relative(attack_df_1, timestamp, insert_timestamp)
            print(last_timestamp)
            normal_df_temp = normal_df_train[(normal_df_train[timestamp] > old_last_timestamp) & (normal_df_train[timestamp] < last_timestamp)]
            
            merged_df_train = merge(normal_df_temp, attack_df_1, timestamp, features)
        
            merged_df_train[features].to_csv(out_path, mode='a', index=False, header=not pd.io.common.file_exists(out_path))
        
            #select a timestamp to merge attack sequence in normal and save last attack package
            insert_index, insert_timestamp = select_insertion_point(normal_df_train, timestamp, last_timestamp)
        
            old_last_timestamp = last_timestamp 
            
            print(old_last_timestamp)
            print(insert_timestamp)
            
            #merge and return last_timestamp of sequence
            attack_df_2, last_timestamp = adjust_attack_timestamps_relative(attack_df_2, timestamp, insert_timestamp)
            print(last_timestamp)
            normal_df_temp = normal_df_train[(normal_df_train[timestamp] > old_last_timestamp) & 
            (normal_df_train[timestamp] < last_timestamp)]
        
            merged_df_train = merge(normal_df_temp, attack_df_2, timestamp, features)
        
            merged_df_train[features].to_csv(out_path, mode='a', index=False, header=not pd.io.common.file_exists(out_path))
    

    