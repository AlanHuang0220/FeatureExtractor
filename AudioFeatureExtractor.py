from transformers import AutoProcessor, ClapModel, ASTModel, HubertModel, Wav2Vec2Model
import librosa
from transformers import logging
from tqdm import tqdm


logging.set_verbosity_error()


class AudioFeatureExtractor:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == "CLAP":
            self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.target_sampling_rate = 48000
        elif model_name == "AST":
            self.processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            self.target_sampling_rate = 16000
        elif model_name == "Hubert":
            self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
            self.model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
            self.target_sampling_rate = 16000
        elif model_name == "Wav2Vec2":
            self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.target_sampling_rate = 16000
        else:
            raise ValueError("Unsupported model.")
        self.model.to(0).eval()

    def _load_audio(self, audio_path, start_time=None, end_time=None):
        if start_time is None or end_time is None:
            audio_input, sample_rate = librosa.load(audio_path, sr=self.target_sampling_rate, mono=True)
            duration_seconds = int(len(audio_input) / sample_rate)#計算音頻時長
        else:
            audio_input, sample_rate = librosa.load(audio_path, sr=self.target_sampling_rate, mono=True, offset=start_time, duration=end_time-start_time)
            duration_seconds = int(end_time - start_time)#計算音頻時長

        if self.model_name == "CLAP":
            audio_input = audio_input[:duration_seconds*sample_rate]
            audio_input = audio_input.reshape(duration_seconds, -1)
            return audio_input, sample_rate
        else:
            return audio_input, sample_rate

    def process_audio(self, audio_path, start_time=None, end_time=None):
        audio_input, sample_rate = self._load_audio(audio_path, start_time, end_time)
        if self.model_name == "CLAP":
            inputs = self.processor.feature_extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to(0)
            hidden_states = self.model.get_audio_features(**inputs)
        else:
            inputs = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to(0).input_values
            hidden_states = self.model(inputs)['last_hidden_state'].squeeze(0)
        tqdm.write(f"{self.model_name}: {hidden_states.shape}")
        return hidden_states
    
    @classmethod
    def process_audio_directly(cls, model_name, audio_path, start_time=None, end_time=None):
        """
        直接提取聲音的特徵

        參數:
        model_name:str,指定要使用的model
        audio_path:str,音檔的路徑
        start_time:可選,int,指定音檔開始的時間(秒)。如果不指定默認從音檔一開始
        end_time:可選,int,指定音檔結束的時間(秒)。如果不指定默認到音檔結束

        return:
        Tensor,從聲音中提取的特徵
        """
        audio_processor = cls(model_name)
        return audio_processor.process_audio(audio_path, start_time, end_time)
    
# Example usage without creating an instance
# model_name = "Wav2Vec2"
# hidden_states = AudioFeatureExtractor.process_audio_directly(model_name, '1.mp3', 5, 10)
# print(hidden_states.shape)
# model_name = "AST"
# hidden_states = AudioFeatureExtractor.process_audio_directly(model_name, '1.mp3', 5, 10)
# print(hidden_states.shape)
# model_name = "Hubert"
# hidden_states = AudioFeatureExtractor.process_audio_directly(model_name, '1.mp3', 5, 10)
# print(hidden_states.shape)
# model_name = "Clap"
# hidden_states = AudioFeatureExtractor.process_audio_directly(model_name, '1.mp3', 5, 10)
# print(hidden_states.shape)
    
