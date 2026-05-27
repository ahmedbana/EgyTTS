import sys
import types
import importlib.machinery

# Mock torchcodec to prevent its buggy C-extensions from loading,
# while still satisfying the Coqui TTS PyTorch 2.9+ validation check.
mock_torchcodec = types.ModuleType("torchcodec")
mock_torchcodec.__path__ = []  # Make it look like a package
mock_torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
sys.modules["torchcodec"] = mock_torchcodec

# Cause sub-module imports to raise ImportError, forcing torchaudio to fall back to soundfile
sys.modules["torchcodec.decoders"] = None
sys.modules["torchcodec.encoders"] = None
sys.modules["torchcodec.samplers"] = None
sys.modules["torchcodec.transforms"] = None
sys.modules["torchcodec._core"] = None
sys.modules["torchcodec._core.ops"] = None

import torch
import torchaudio
import soundfile as sf

def monkeypatched_load(
    uri,
    frame_offset=0,
    num_frames=-1,
    normalize=True,
    channels_first=True,
    format=None,
    buffer_size=4096,
    backend=None,
):
    start = frame_offset
    stop = start + num_frames if num_frames > 0 else None
    data, samplerate = sf.read(uri, start=start, stop=stop, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif channels_first:
        tensor = tensor.T
    return tensor, samplerate

def monkeypatched_save(
    uri,
    src,
    sample_rate,
    channels_first=True,
    format=None,
    encoding=None,
    bits_per_sample=None,
    buffer_size=4096,
    backend=None,
    compression=None,
):
    data = src.cpu().numpy()
    if channels_first and data.ndim > 1:
        data = data.T
    sf.write(uri, data, sample_rate)

torchaudio.load = monkeypatched_load
torchaudio.save = monkeypatched_save

import os
import shutil
import tempfile
import wave
import functools
import folder_paths
from huggingface_hub import hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Define the models directory for TTS
tts_models_dir = os.path.join(folder_paths.models_dir, "TTS")
try:
    os.makedirs(tts_models_dir, exist_ok=True)
except Exception as e:
    print(f"[EgyTTS] Warning: Could not create directory {tts_models_dir}: {e}")

# Register the directory in ComfyUI
if "tts" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tts"] = ([tts_models_dir], {".pth"})

def download_model_files(model_dir):
    repo_id = "OmarSamir/EGTTS-V0.1"
    try:
        os.makedirs(model_dir, exist_ok=True)
    except Exception as e:
        print(f"[EgyTTS] Warning: Could not create directory {model_dir}: {e}")
    
    for file in ["model.pth", "config.json", "vocab.json", "speaker_reference.wav"]:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            print(f"[EgyTTS] Downloading {file} from HuggingFace repository '{repo_id}'...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download {file} from HF: {e}")

class EgyTTSGenerator:
    """
    ComfyUI Node for Colloquial Egyptian Arabic Text-To-Speech (EGTTS-V0.1 / XTTS v2).
    Generates high-quality speech in Egyptian Arabic dialect from a text prompt and reference audio.
    """
    
    _cached_model = None
    _cached_model_path = None
    _cached_device = None
    
    _cached_speaker_key = None
    _cached_gpt_cond_latent = None
    _cached_speaker_embedding = None
    _cached_speaker_audio_tensor = None

    @classmethod
    def INPUT_TYPES(cls):
        # Retrieve registered models (.pth files) in models/TTS
        models = folder_paths.get_filename_list("tts")
        # Ensure we always have an auto-download option
        auto_download_option = "OmarSamir/EGTTS-V0.1 (Auto-Download)"
        if not models:
            models = [auto_download_option]
        else:
            models = [m for m in models if m.endswith(".pth")]
            models.append(auto_download_option)
            
        return {
            "required": {
                "model_name": (models,),
                "prompt": ("STRING", {
                    "default": "صباح الخير، إزيك عامل إيه؟",
                    "multiline": True,
                    "placeholder": "اكتب الكلام بالعامية المصرية هنا..."
                }),
                "language": (["ar", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ja", "ko", "hu", "sv"], {"default": "ar"}),
                "device": (["auto", "cpu", "cuda", "mps"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 2.0, "step": 0.01}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "EgyTTS"

    def generate(self, model_name, prompt, language, device, temperature, speed, 
                 reference_audio=None, repetition_penalty=2.0, length_penalty=1.0, 
                 top_k=50, top_p=0.85):
        
        # 1. Resolve model directories and download if necessary
        if model_name == "OmarSamir/EGTTS-V0.1 (Auto-Download)":
            # Check if user already has model.pth in the parent tts_models_dir
            parent_checkpoint = os.path.join(tts_models_dir, "model.pth")
            if os.path.exists(parent_checkpoint):
                model_dir = tts_models_dir
            else:
                model_dir = os.path.join(tts_models_dir, "EGTTS-V0.1")
            download_model_files(model_dir)
            checkpoint_path = os.path.join(model_dir, "model.pth")
            config_path = os.path.join(model_dir, "config.json")
            vocab_path = os.path.join(model_dir, "vocab.json")
        else:
            checkpoint_path = folder_paths.get_full_path("tts", model_name)
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {model_name}")
            model_dir = os.path.dirname(checkpoint_path)
            config_path = os.path.join(model_dir, "config.json")
            vocab_path = os.path.join(model_dir, "vocab.json")
            
            # Double check config and vocab exist, otherwise download them
            if not os.path.exists(config_path) or not os.path.exists(vocab_path):
                print(f"[EgyTTS] Config or Vocab missing in model dir. Downloading...")
                download_model_files(model_dir)

        # 2. Resolve Device
        if device == "auto":
            if torch.cuda.is_available():
                actual_device = "cuda"
            else:
                actual_device = "cpu"
        else:
            actual_device = device
            
        # 3. Load / Cache Model
        if (EgyTTSGenerator._cached_model is None or 
            EgyTTSGenerator._cached_model_path != checkpoint_path or 
            EgyTTSGenerator._cached_device != actual_device):
            
            print(f"[EgyTTS] Loading XTTS model from {checkpoint_path} on {actual_device}...")
            
            config = XttsConfig()
            config.load_json(config_path)
            
            model = Xtts.init_from_config(config)
            
            # Load checkpoint with patched torch.load to bypass weights_only safety on PyTorch 2.6+
            original_load = torch.load
            try:
                torch.load = functools.partial(original_load, weights_only=False)
                model.load_checkpoint(
                    config, 
                    checkpoint_path=checkpoint_path, 
                    vocab_path=vocab_path, 
                    use_deepspeed=False
                )
            finally:
                torch.load = original_load
                
            model.to(actual_device)
            
            EgyTTSGenerator._cached_model = model
            EgyTTSGenerator._cached_model_path = checkpoint_path
            EgyTTSGenerator._cached_device = actual_device
            
            # Reset speaker latents cache since model changed
            EgyTTSGenerator._cached_speaker_key = None
            EgyTTSGenerator._cached_gpt_cond_latent = None
            EgyTTSGenerator._cached_speaker_embedding = None
            EgyTTSGenerator._cached_speaker_audio_tensor = None
        else:
            model = EgyTTSGenerator._cached_model

        # 4. Resolve Speaker Reference Audio
        speaker_wav = None
        is_same_audio = False
        
        if reference_audio is not None:
            # reference_audio format: {"waveform": tensor [batch, channels, samples], "sample_rate": int}
            waveform = reference_audio["waveform"]
            sample_rate = reference_audio["sample_rate"]
            
            # Check if reference_audio tensor is identical to cached one
            if (EgyTTSGenerator._cached_speaker_audio_tensor is not None and
                EgyTTSGenerator._cached_speaker_audio_tensor.shape == waveform.shape and
                torch.equal(EgyTTSGenerator._cached_speaker_audio_tensor, waveform)):
                is_same_audio = True
            
            # Use a fixed temporary path to optimize file system caching
            temp_wav_path = os.path.join(tempfile.gettempdir(), "egytts_speaker_ref.wav")
            
            if not is_same_audio:
                print("[EgyTTS] New reference audio detected. Writing temporary wave file...")
                # Extract first batch item and convert to 16-bit PCM WAV bytes
                wav_to_save = waveform[0]
                wav_tensor = wav_to_save.cpu().clamp(-1.0, 1.0)
                wav_pcm = (wav_tensor * 32767.0).to(torch.int16)
                
                # Interleave channels if stereo/multi-channel
                if wav_pcm.ndim > 1:
                    wav_pcm_interleaved = wav_pcm.T
                else:
                    wav_pcm_interleaved = wav_pcm
                
                wav_bytes = wav_pcm_interleaved.numpy().tobytes()
                num_channels = wav_pcm.shape[0] if wav_pcm.ndim > 1 else 1
                
                # Write to PCM WAV file using built-in wave module (no ffmpeg/torchcodec dependency)
                with wave.open(temp_wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(num_channels)
                    wav_file.setsampwidth(2) # 2 bytes for 16-bit PCM
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(wav_bytes)
                
                # Cache the tensor
                EgyTTSGenerator._cached_speaker_audio_tensor = waveform
                
            speaker_wav = temp_wav_path
            
        else:
            # Fallback to default speaker reference
            default_ref = os.path.join(model_dir, "speaker_reference.wav")
            if os.path.exists(default_ref):
                speaker_wav = default_ref
            else:
                # Try to download default reference if missing
                print("[EgyTTS] Downloading default speaker_reference.wav...")
                try:
                    speaker_wav = hf_hub_download(
                        repo_id="OmarSamir/EGTTS-V0.1",
                        filename="speaker_reference.wav",
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    raise RuntimeError(f"No reference audio provided and failed to download default: {e}")
            
            # Compare default reference usage
            if EgyTTSGenerator._cached_speaker_audio_tensor is None:
                is_same_audio = True
            else:
                is_same_audio = False
                EgyTTSGenerator._cached_speaker_audio_tensor = None

        # 5. Compute or retrieve cached conditioning latents
        speaker_key = (
            checkpoint_path, 
            speaker_wav, 
            is_same_audio
        )
        
        if (EgyTTSGenerator._cached_speaker_key == speaker_key and 
            EgyTTSGenerator._cached_gpt_cond_latent is not None):
            print("[EgyTTS] Using cached speaker conditioning latents.")
            gpt_cond_latent = EgyTTSGenerator._cached_gpt_cond_latent
            speaker_embedding = EgyTTSGenerator._cached_speaker_embedding
        else:
            print("[EgyTTS] Computing speaker conditioning latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[speaker_wav]
            )
            EgyTTSGenerator._cached_speaker_key = speaker_key
            EgyTTSGenerator._cached_gpt_cond_latent = gpt_cond_latent
            EgyTTSGenerator._cached_speaker_embedding = speaker_embedding

        # 6. Run Inference
        print(f"[EgyTTS] Running inference for prompt (length: {len(prompt)} chars)...")
        out = model.inference(
            text=prompt,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            speed=speed,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            top_k=top_k,
            top_p=top_p
        )
        
        # 7. Format output audio for ComfyUI
        # out["wav"] is usually a list or numpy array
        wav = out["wav"]
        if isinstance(wav, list):
            wav_tensor = torch.tensor(wav, dtype=torch.float32)
        elif isinstance(wav, torch.Tensor):
            wav_tensor = wav.cpu().float()
        else:
            wav_tensor = torch.from_numpy(wav).float()
            
        # ComfyUI audio shape: [batch, channels, samples] -> [1, 1, samples]
        waveform_out = wav_tensor.unsqueeze(0).unsqueeze(0)
        
        audio_out = {
            "waveform": waveform_out,
            "sample_rate": 24000
        }
        
        print("[EgyTTS] Speech generation complete.")
        return (audio_out,)
