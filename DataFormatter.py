import pandas as pd
import dask.dataframe as dd
import re
import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from utils import time_to_seconds

class DataFormatter:
    @staticmethod
    def from_videocc(csv_path):
        # 讀取所有行
        with open(csv_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 使用正則表達式分割前三個逗號，保留剩餘部分不變
        pattern = re.compile(r'^([^,]+),([^,]+),([^,]+),(.+)$')
        data = [re.match(pattern, line).groups() for line in lines if re.match(pattern, line)]

        # 轉換成DataFrame
        df = pd.DataFrame(data, columns=['Youtube_ID', 'Start_time', 'End_time', 'Description'])
        return df

    @staticmethod
    def from_vast27m(json_path, columns_to_load:list):
        # 讀取parquet
        df = pd.read_json(json_path)
        # df['clip_span_seconds'] = df['clip_span'].apply(lambda x: [time_to_seconds(t) for t in x])
        return df[columns_to_load]
        
    @staticmethod
    def from_msrvtt(json_path):
        df = pd.read_json(json_path)
        print(df.head())
        return df
    
    def from_msvd(json_path):
        df = pd.read_json(json_path)
        print(df.head())
        return df
    
    def from_didemo(json_path):
        df = pd.read_json(json_path)
        print(df.head())
        return df


# DataFormatter.from_videocc('pretrain_dataset/VideoCC/video_cc_public.csv')
# df = DataFormatter.from_vast27m('pretrain_dataset/VAST27M/split_annotations/split_0.json')
# print(df)
