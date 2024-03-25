from transformers import AutoProcessor, AutoImageProcessor, MobileNetV1Model, SwinModel, ViTModel, CLIPModel
from transformers import logging
from PIL import Image
import torch
import cv2
from tqdm import tqdm

logging.set_verbosity_error()

class VisionFeatureExtractor:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == "MobileNet":
            self.processor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
            self.model = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")
        elif model_name == "Swin":
            self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        elif model_name == "ViT":
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif model_name == "CLIP":
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError("Unsupported model.")
        self.model.to(0).eval()
        
    def _load_image(self, image_path):
        image = Image.open(image_path)
        return image
    
    def _load_video_frames(self, video_path, start_time=None, end_time=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_time * fps) if start_time is not None else 1
        end_frame = int(end_time * fps) if end_time is not None else total_frames

        frames = []
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret or current_frame > end_frame:
                break
            
            if current_frame >= start_frame and ((current_frame - start_frame) % int(fps) == int(fps/2)):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))

            current_frame += 1

        cap.release()
        return frames
    
    def process_video(self, video_path, start_time=None, end_time=None):
        frames = self._load_video_frames(video_path, start_time, end_time)
        hidden_states = []

        if self.model_name == "CLIP":
                inputs = self.processor(images=frames, return_tensors="pt").to(0)
                hidden_states = self.model.get_image_features(**inputs)
        else:
            with torch.no_grad():
                inputs = self.processor(images=frames, return_tensors="pt").to(0)
                hidden_states = self.model(**inputs).pooler_output
        tqdm.write(f"{self.model_name}: {hidden_states.shape}")
        return hidden_states
    
    @classmethod
    def process_video_directly(cls, model_name, video_path, start_time=None, end_time=None):
        """
        直接提取影片的特徵

        參數:
        model_name:str,指定要使用的model
        video_path:str,影片的路徑
        start_time:可選,int,指定影片開始幀的時間(秒)。如果不指定默認從影片一開始
        end_time:可選,int,指定影片結束幀的時間(秒)。如果不指定默認到影片結束

        return:
        Tensor,從影片中提取的特徵
        """
        video_processor = cls(model_name)
        return video_processor.process_video(video_path, start_time, end_time)
    
# model_name = "MobileNet"
# video_path = "1.mp4"
# start_time = 10
# end_time = 15
# hidden_states = VisionFeatureExtractor.process_video_directly(model_name, video_path, start_time, end_time)
# print(hidden_states.shape)
# model_name = "Swin"
# hidden_states = VisionFeatureExtractor.process_video_directly(model_name, video_path, start_time, end_time)
# print(hidden_states.shape)
# model_name = "ViT"
# hidden_states = VisionFeatureExtractor.process_video_directly(model_name, video_path, start_time, end_time)
# print(hidden_states.shape)
# model_name = "CLIP"
# hidden_states = VisionFeatureExtractor.process_video_directly(model_name, video_path, start_time, end_time)
# print(hidden_states.shape)


