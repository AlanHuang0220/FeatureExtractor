import os
import sys
import json
import pickle
from tqdm import tqdm
from DataFormatter import DataFormatter
from AudioFeatureExtractor import AudioFeatureExtractor
from VisionFeatureExtractor import VisionFeatureExtractor
from utils import check_file_existence, convert_mp4_to_wav, align_tensors_by_first_dim


class Config:
    def __init__(self, json_file_name):
        self.dataset_base_path = 'F:/dataset/pretrain_dataset/VAST27M'
        self.audio_folder = os.path.join(self.dataset_base_path, 'audio')
        self.video_folder = os.path.join(self.dataset_base_path, 'video')
        self.data_json_path = os.path.join(self.dataset_base_path, 'split_annotations1', json_file_name)
        self.target_folder = os.path.join(self.dataset_base_path, 'video_feature')
        self.processed_clips_file = f'{json_file_name.split(".")[0]}_processed_clips.txt'
        self.columns_to_load = ['clip_id', 'vision_cap', 'audio_cap', 'subtitle', 'vast_cap']
        self.vision_model = 'CLIP'
        self.audio_model = 'CLAP'

def record_processed_clip_id(processed_clips_file, clip_id):
    with open(processed_clips_file, 'a') as file:
        file.write(f"{clip_id}\n")

def load_processed_clips(processed_clips_file):
    if not os.path.exists(processed_clips_file):
        return set()
    with open(processed_clips_file, 'r') as file:
        return set(file.read().splitlines())
    
def ensure_audio_exists(video_path, audio_path):
    if not check_file_existence(audio_path):
        try:
            convert_mp4_to_wav(video_path, audio_path)
            return True
        except Exception as e:
            tqdm.write(f"Error converting video to audio: {e}")
            return False
    return True

def process_clip(row, config, vision_feature_extractor, audio_feature_extractor):
    clip_id = row['clip_id']
    video_path = os.path.join(config.video_folder, f"clip_{clip_id}.mp4")
    audio_path = os.path.join(config.audio_folder, f"clip_{clip_id}.wav")
    clip_target_folder = os.path.join(config.target_folder, clip_id)
    
    if check_file_existence(video_path) and ensure_audio_exists(video_path, audio_path):
        tqdm.write(f"{clip_id} Encoding...")
        os.makedirs(clip_target_folder, exist_ok=True)
        vision_feature = vision_feature_extractor.process_video(video_path)
        audio_feature = audio_feature_extractor.process_audio(audio_path)
        
        vision_feature, audio_feature = align_tensors_by_first_dim(vision_feature, audio_feature)
        
        with open(os.path.join(clip_target_folder, f"{config.vision_model}.pkl"), 'wb') as vf, \
             open(os.path.join(clip_target_folder, f"{config.audio_model}.pkl"), 'wb') as af:
            pickle.dump(vision_feature.detach().cpu().numpy(), vf)
            pickle.dump(audio_feature.detach().cpu().numpy(), af)
        
        with open(os.path.join(clip_target_folder, "caption.json"), 'w') as f:
            json.dump(row.to_dict(), f, indent=4)

    else:
        tqdm.write(f"{clip_id} not exist.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract.py <JSON_FILE_NAME>")
        sys.exit(1)
    json_file_name = sys.argv[1]  # 從 command-line 讀取json檔名稱
    config = Config(json_file_name)

    # Instantiate feature extractor
    vision_feature_extractor = VisionFeatureExtractor(config.vision_model)
    audio_feature_extractor = AudioFeatureExtractor(config.audio_model)

    df = DataFormatter.from_vast27m(config.data_json_path, config.columns_to_load)
    processed_clips = load_processed_clips(config.processed_clips_file)

    tqdm_desc = f"Processing Clips [{json_file_name}]"
    for index, row in tqdm(df.iterrows(), total=len(df), desc=tqdm_desc):
        clip_id = row['clip_id']
        if clip_id in processed_clips:
            tqdm.write(f"{clip_id} already processed. Skipping...")
            continue
        
        process_clip(row, config, vision_feature_extractor, audio_feature_extractor)
        record_processed_clip_id(config.processed_clips_file, clip_id)