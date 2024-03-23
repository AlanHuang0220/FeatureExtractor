import json
import os
import ijson
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def time_to_seconds(time_str):
    """
    將時間字串 (HH:MM:SS.xxx) 轉換為秒
    
    参数:
    time_str (str): 時間字串
    
    返回:
    (float) 秒數 
    """
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(float(s))

def split_large_json(input_file, output_prefix, record_per_file):
    file_size = os.path.getsize(input_file) # 獲取文件大小，用於進度條

    # 打開原始JSON檔案
    with open(input_file, 'rb') as file:
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True) # 創建一個基於文件大小的進度條
        # 創建一個生成器來逐步讀取檔案
        objects = ijson.items(file, 'item')
        current_chunk = []
        file_index = 0
        
        for obj in objects:
            current_chunk.append(obj)
            progress_bar.update(file.tell() - progress_bar.n)  # 更新進度條

            if len(current_chunk) == record_per_file:
                output_file = f"{output_prefix}_{file_index}.json"
                with open(output_file, 'w', encoding='utf-8') as out_file:
                    json.dump(current_chunk, out_file, ensure_ascii=False, indent=4)
                current_chunk = []
                file_index += 1
        
        # 處理剩餘的記錄
        if current_chunk:
            output_file = f"{output_prefix}_{file_index}.json"
            with open(output_file, 'w', encoding='utf-8') as out_file:
                json.dump(current_chunk, out_file, ensure_ascii=False, indent=4)
        
        progress_bar.close()

def check_file_existence(file_path):
    """
    檢查給的路徑是否存在文件。

    Parameters:
    file_path (str): 要檢查的Path

    retuen:
    如果文件存在return True,如果不存在return False
    """
    return os.path.exists(file_path)    

def convert_mp4_to_wav(source_file_path, output_file_path):
    """
    將mp4 file 轉換為 wav file。
    
    Parameters:
        source_file_path (str): mp4 file 的路徑
        output_file_path (str): wav 儲存路徑
    """
    # Load the video file
    video_clip = VideoFileClip(source_file_path)
    
    # Extract the audio part
    audio_clip = video_clip.audio
    
    # Write the audio to a WAV file
    audio_clip.write_audiofile(output_file_path, codec='pcm_s16le')
    
    # Close the clips to free up system resources
    audio_clip.close()
    video_clip.close()
    

def align_tensors_by_first_dim(tensor1, tensor2):
    """
    輸入兩個 Tensor 並根據 shape[0] 的長度對他們進行裁剪，
    使他們長度相等。如果兩個tensor長度已經相等則直接 return

    Parameters:
        tensor1 (Tensor): 第一個 Tensor
        tensor2 (Tensor): 第二個 Tensor

    return:
        Tuple[Tensor, Tensor]: 在dim 0 上長度相等的兩個 Tensor
    """
    len1 = tensor1.shape[0]
    len2 = tensor2.shape[0]

    if len1 != len2: # 確認兩個tensor在dim=0 的長度是否相等
        if len1 > len2:
            tensor1 = tensor1[:len2, :]  # 裁剪tensor1
        else:
            tensor2 = tensor2[:len1, :]  # 裁剪tensor2

    return tensor1, tensor2
# split_large_json('F:/dataset/pretrain_dataset/VAST27M/annotations.json', 'F:/dataset/pretrain_dataset/VAST27M/split_annotations1/split', 100000)
# 轉換.mp4
# video_folder = 'F:/dataset/pretrain_dataset/VAST27M/video'
# audio_folder = 'F:/dataset/pretrain_dataset/VAST27M/audio'
# for file_name in os.listdir(video_folder):
#     video_path = os.path.join(video_folder, file_name)
#     audio_path = os.path.join(audio_folder, file_name.replace('.mp4', '.wav'))
#     convert_mp4_to_wav(video_path, audio_path)