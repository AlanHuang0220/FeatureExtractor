from AudioFeatureExtractor import AudioFeatureExtractor
from VisionFeatureExtractor import VisionFeatureExtractor
from DataFormatter import DataFormatter
from utils import check_file_existence, convert_mp4_to_wav, align_tensors_by_first_dim
import os
import pickle
import json

if __name__ == "__main__":
    audio_folder = 'F:/dataset/pretrain_dataset/VAST27M/audio'
    video_folder = 'F:/dataset/pretrain_dataset/VAST27M/video' 
    data_json_path = 'F:/dataset/pretrain_dataset/VAST27M/split_annotations1/split_0.json'
    columns_to_load = ['clip_id', 'vision_cap', 'audio_cap', 'subtitle', 'vast_cap']
    target_folder = 'F:/dataset/pretrain_dataset/VAST27M/video_feature'

    vision_model = 'CLIP'
    audio_model = 'CLAP'
    vision_feature_extractor = VisionFeatureExtractor(vision_model)
    audio_feature_extractor = AudioFeatureExtractor(audio_model)

    df = DataFormatter.from_vast27m(data_json_path, columns_to_load)
    for index, row in df.iterrows():
        video_path = f"{video_folder}/clip_{row['clip_id']}.mp4"
        audio_path = f"{audio_folder}/clip_{row['clip_id']}.wav"
        if check_file_existence(video_path):
            if not check_file_existence(audio_path):
                convert_mp4_to_wav(video_path, audio_path)
            
            print(f"{row['clip_id']} Encoding ...")
            os.makedirs(f"{target_folder}/{row['clip_id']}")
            vision_feature = vision_feature_extractor.process_video(video_path)
            audio_feature = audio_feature_extractor.process_audio(audio_path)

            vision_feature, audio_feature = align_tensors_by_first_dim(vision_feature, audio_feature)
            with open(f"{target_folder}/{row['clip_id']}/{vision_model}.pkl", 'wb') as f:
                pickle.dump(vision_feature, f)

            with open(f"{target_folder}/{row['clip_id']}/{audio_model}.pkl", 'wb') as f:
                pickle.dump(audio_feature, f)

            with open(f"{target_folder}/{row['clip_id']}/caption.json", 'w') as f:
                f.write(json.dumps(row.to_dict(), indent=4))

        else:
            print(f"{row['clip_id'] } 不存在")
        
