#!/usr/bin/env python3
"""
Seed-VC Real-Time Inference Benchmark

Auto-detects system specs, benchmarks all model configurations, and validates
real-time feasibility by measuring per-component latency and computing RTF.

Usage:
    python tools/benchmark_realtime.py [--model MODEL_NAME] [--steps STEPS] [--block-time SECONDS]

Examples:
    python tools/benchmark_realtime.py                          # Benchmark all configs
    python tools/benchmark_realtime.py --model xlsr_tiny         # Benchmark specific model
    python tools/benchmark_realtime.py --model xlsr_tiny --steps 4,6,8,10
    python tools/benchmark_realtime.py --block-time 0.20         # Use custom block time
"""

import os
import sys
import json
import time
import argparse
import platform
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import numpy as np
import librosa

# ============================================================================
# System Detection
# ============================================================================

def detect_system() -> Dict:
    """Detect system specifications for Apple Silicon Macs."""
    
    # Basic info
    specs = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": None,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "chip": None,
        "cpu_count_physical": None,
        "cpu_count_logical": None,
        "memory_total_gb": None,
        "memory_available_gb": None,
    }
    
    # Device detection
    if specs["mps_available"]:
        specs["device"] = "mps"
        # Try to get MPS device name
        try:
            specs["mps_device"] = torch.backends.mps.current_device()
        except:
            specs["mps_device"] = "unknown"
    elif specs["cuda_available"]:
        specs["device"] = "cuda"
        specs["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        specs["device"] = "cpu"
    
    # Apple Silicon detection
    if platform.system() == "Darwin":
        try:
            # Get hardware info using system_profiler
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                hw_info = json.loads(result.stdout)
                if hw_info and "SPHardwareDataType" in hw_info:
                    hw = hw_info["SPHardwareDataType"][0]
                    specs["chip"] = hw.get("chip_name", "Unknown Apple Silicon")
                    specs["cpu_count_physical"] = hw.get("physical_processor_count")
                    specs["cpu_count_logical"] = hw.get("thread_count")
                    specs["memory_total_gb"] = hw.get("physical_memory") / 1024
        except Exception as e:
            specs["chip_detection_error"] = str(e)
        
        # Fallback: try to parse from /proc or other sources
        if specs["chip"] is None or specs["chip"] == "Unknown Apple Silicon":
            # Check for M-series chips
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    specs["chip"] = result.stdout.strip()
            except:
                pass
    
    # Memory info (cross-platform)
    try:
        import psutil
        memory = psutil.virtual_memory()
        specs["memory_total_gb"] = memory.total / (1024**3)
        specs["memory_available_gb"] = memory.available / (1024**3)
        specs["memory_percent_used"] = memory.percent
        specs["cpu_count_logical"] = psutil.cpu_count(logical=True)
        specs["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        # psutil not installed, use rough estimates
        specs["psutil_available"] = False
    
    return specs


# ============================================================================
# Model Auto-Detection
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model benchmark."""
    name: str
    yaml_file: str
    checkpoint: str
    speech_tokenizer: str
    vocoder: str
    estimated_params_m: float
    description: str


def load_available_models() -> Dict:
    """
    Auto-detect available models from checkpoints directory.
    Returns a dict of model_name -> ModelConfig
    """
    # Import detection module
    tools_dir = Path(__file__).parent
    sys.path.insert(0, str(tools_dir))
    
    try:
        from detect_models import detect_available_models, KNOWN_MODELS
        import json
        
        # Detect available models
        detected = detect_available_models()
        
        # Build model configs from detected models
        available_models = {}
        for model_info in detected:
            if model_info.ready_for_benchmark and not model_info.is_v2:
                # Create config from detected info - use FULL path, not just filename
                from benchmark_realtime import ModelConfig
                available_models[model_info.name] = ModelConfig(
                    name=model_info.name,
                    yaml_file=f"configs/presets/{model_info.config}",
                    checkpoint=model_info.path,  # ← Use full path, not filename
                    speech_tokenizer=model_info.tokenizer,
                    vocoder=model_info.vocoder,
                    estimated_params_m=model_info.estimated_params_m,
                    description=model_info.description
                )
        
        # If no models detected, provide defaults that will download from HF
        if not available_models:
            print("⚠️  No models found locally. Will attempt to download from HuggingFace.")
            print("   First benchmark run will take longer due to downloads.\n")
            
            # Provide default configs that trigger HF downloads
            from benchmark_realtime import ModelConfig
            available_models = {
                "xlsr_tiny": ModelConfig(
                    name="xlsr_tiny",
                    yaml_file="configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml",
                    checkpoint="DiT_uvit_tat_xlsr_ema.pth",
                    speech_tokenizer="xlsr",
                    vocoder="hifigan",
                    estimated_params_m=25,
                    description="25M DiT + XLSR-large (300M frozen) + HiFi-GAN"
                ),
            }
        
        return available_models
        
    except Exception as e:
        print(f"⚠️  Model detection failed: {e}")
        print("   Falling back to default model configs.\n")
        
        # Fallback to hardcoded configs
        from benchmark_realtime import ModelConfig
        return {
            "xlsr_tiny": ModelConfig(
                name="xlsr_tiny",
                yaml_file="configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml",
                checkpoint="DiT_uvit_tat_xlsr_ema.pth",
                speech_tokenizer="xlsr",
                vocoder="hifigan",
                estimated_params_m=25,
                description="25M DiT + XLSR-large (300M frozen) + HiFi-GAN"
            ),
        }


# Load available models at startup
AVAILABLE_MODELS = load_available_models()

# For backward compatibility
MODEL_CONFIGS = AVAILABLE_MODELS


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class ComponentLatency:
    """Latency breakdown for individual components."""
    vad_ms: float = 0.0
    resample_ms: float = 0.0
    content_encoder_ms: float = 0.0
    style_encoder_ms: float = 0.0
    length_regulator_ms: float = 0.0
    dit_inference_ms: float = 0.0
    vocoder_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one configuration."""
    timestamp: str
    model_name: str
    diffusion_steps: int
    block_time: float
    num_runs: int
    latencies_ms: List[float] = field(default_factory=list)
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    rtf: float = 0.0  # Real-Time Factor = latency / block_time
    memory_peak_mb: float = 0.0
    component_latencies: Dict = field(default_factory=dict)
    feasible: bool = False
    feasible_mean: bool = False
    feasible_p99: bool = False
    system_specs: Dict = field(default_factory=dict)
    error: Optional[str] = None


# ============================================================================
# Model Loading and Inference
# ============================================================================

class BenchmarkRunner:
    """Handles model loading and benchmark execution."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model_set = None
        self.config = None
        self.model_params = None
        
    def load_model(self, model_config: ModelConfig) -> bool:
        """Load a model configuration for benchmarking."""
        try:
            from modules.commons import recursive_munch, build_model, load_checkpoint
            from hf_utils import load_custom_model_from_hf
            
            # Load config
            yaml_path = Path(model_config.yaml_file)
            if not yaml_path.exists():
                print(f"❌ Config file not found: {yaml_path}")
                return False
            
            self.config = yaml.safe_load(open(yaml_path, "r"))
            self.model_params = recursive_munch(self.config["model_params"])
            self.model_params.dit_type = 'DiT'
            
            # Build model
            self.model = build_model(self.model_params, stage="DiT")
            
            # Load checkpoint
            checkpoint_path = model_config.checkpoint
            if not Path(checkpoint_path).exists():
                # Try to load from HuggingFace
                try:
                    checkpoint_path, _ = load_custom_model_from_hf(
                        "Plachta/Seed-VC",
                        model_config.checkpoint,
                        None
                    )
                except:
                    print(f"⚠️  Checkpoint not found: {model_config.checkpoint}")
                    print(f"   Will attempt to use default or skip this model")
                    return False
            
            self.model, _, _, _ = load_checkpoint(
                self.model, None, checkpoint_path,
                load_only_params=True, ignore_modules=[], is_distributed=False
            )
            
            # Move to device and setup
            for key in self.model:
                self.model[key].eval()
                self.model[key].to(self.device)
            
            self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            
            # Load CAMPPlus
            from modules.campplus.DTDNN import CAMPPlus
            campplus_ckpt_path = load_custom_model_from_hf(
                "funasr/campplus", "campplus_cn_common.bin", config_filename=None
            )
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(
                torch.load(campplus_ckpt_path, map_location="cpu")
            )
            self.campplus_model.eval()
            self.campplus_model.to(self.device)
            
            # Load vocoder
            vocoder_type = self.model_params.vocoder.type
            if vocoder_type == 'bigvgan':
                from modules.bigvgan import bigvgan
                bigvgan_name = self.model_params.vocoder.name
                self.vocoder = bigvgan.BigVGAN.from_pretrained(
                    bigvgan_name, use_cuda_kernel=False
                )
                self.vocoder.remove_weight_norm()
                self.vocoder = self.vocoder.eval().to(self.device)
            elif vocoder_type == 'hifigan':
                from modules.hifigan.generator import HiFTGenerator
                from modules.hifigan.f0_predictor import ConvRNNF0Predictor
                hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
                hift_gen = HiFTGenerator(
                    **hift_config['hift'],
                    f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor'])
                )
                hift_path = load_custom_model_from_hf(
                    "FunAudioLLM/CosyVoice-300M", 'hift.pt', None
                )
                hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
                hift_gen.eval()
                hift_gen.to(self.device)
                self.vocoder = hift_gen
            else:
                print(f"❌ Unknown vocoder type: {vocoder_type}")
                return False
            
            # Load speech tokenizer
            tokenizer_type = self.model_params.speech_tokenizer.type
            if tokenizer_type == 'whisper':
                self.semantic_fn = self._load_whisper_tokenizer()
            elif tokenizer_type == 'xlsr':
                self.semantic_fn = self._load_xlsr_tokenizer()
            else:
                print(f"❌ Unknown tokenizer type: {tokenizer_type}")
                return False
            
            # Setup mel spectrogram
            from modules.audio import mel_spectrogram
            mel_fn_args = {
                "n_fft": self.config['preprocess_params']['spect_params']['n_fft'],
                "win_size": self.config['preprocess_params']['spect_params']['win_length'],
                "hop_size": self.config['preprocess_params']['spect_params']['hop_length'],
                "num_mels": self.config['preprocess_params']['spect_params']['n_mels'],
                "sampling_rate": self.config['preprocess_params']['sr'],
                "fmin": self.config['preprocess_params']['spect_params'].get('fmin', 0),
                "fmax": None if self.config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
                "center": False
            }
            self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
            self.mel_fn_args = mel_fn_args
            
            # Load VAD model
            from funasr import AutoModel
            self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
            
            print(f"✅ Model loaded: {model_config.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_whisper_tokenizer(self):
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = self.model_params.speech_tokenizer.name
        
        # Load with appropriate dtype for device
        if self.device.type == "mps":
            whisper_model = WhisperModel.from_pretrained(
                whisper_name, torch_dtype=torch.float32
            ).to(self.device)
        else:
            whisper_model = WhisperModel.from_pretrained(
                whisper_name, torch_dtype=torch.float16
            ).to(self.device)
        
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
        
        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(self.device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
        
        return semantic_fn
    
    def _load_xlsr_tokenizer(self):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        model_name = self.config['model_params']['speech_tokenizer']['name']
        output_layer = self.config['model_params']['speech_tokenizer']['output_layer']
        
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(self.device)
        wav2vec_model = wav2vec_model.eval()
        
        # Use float32 for MPS compatibility
        if self.device.type == "mps":
            wav2vec_model = wav2vec_model.to(torch.float32)
        else:
            wav2vec_model = wav2vec_model.half()
        
        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000
            ).to(self.device)
            with torch.no_grad():
                if self.device.type == "mps":
                    ori_outputs = wav2vec_model(ori_inputs.input_values)
                else:
                    ori_outputs = wav2vec_model(ori_inputs.input_values.half())
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
        
        return semantic_fn
    
    def benchmark_inference(self, audio_16k: torch.Tensor, 
                            reference_wav: np.ndarray,
                            diffusion_steps: int,
                            num_runs: int = 5) -> Tuple[List[float], Dict]:
        """
        Benchmark inference with detailed timing.
        
        Returns:
            Tuple of (list of total latencies, dict of component latencies)
        """
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        import torch.nn.functional as F
        
        sr = self.mel_fn_args["sampling_rate"]
        hop_length = self.mel_fn_args["hop_size"]
        
        # Prepare reference
        prompt_len = 3.0  # seconds
        reference_wav_trimmed = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav_trimmed).to(self.device)
        
        # Pre-compute reference features
        ori_waves_16k_ref = torchaudio.functional.resample(
            reference_wav_tensor, sr, 16000
        )
        S_ref = self.semantic_fn(ori_waves_16k_ref.unsqueeze(0))
        feat_ref = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k_ref.unsqueeze(0),
            num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat_ref = feat_ref - feat_ref.mean(dim=0, keepdim=True)
        style_ref = self.campplus_model(feat_ref.unsqueeze(0))
        mel_ref = self.to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel_ref.size(2)]).to(mel_ref.device)
        prompt_condition = self.model.length_regulator(
            S_ref, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]
        
        # Warmup run
        self._single_inference(audio_16k, prompt_condition, mel_ref, style_ref, 
                              diffusion_steps, warmup=True)
        
        # Benchmark runs
        latencies = []
        component_times = {
            "vad": [], "resample": [], "content_encoder": [],
            "style_encoder": [], "length_regulator": [], "dit": [],
            "vocoder": [], "postprocess": []
        }
        
        for run in range(num_runs):
            if self.device.type == "mps":
                start_event = torch.mps.event.Event(enable_timing=True)
                end_event = torch.mps.event.Event(enable_timing=True)
            else:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
            
            run_start = time.perf_counter()
            
            # Time each component
            comp_times = {}
            
            # VAD
            if self.device.type == "mps":
                torch.mps.synchronize()
            start_event.record()
            audio_16k_np = audio_16k.cpu().numpy()
            vad_result = self.vad_model.generate(
                input=audio_16k_np, cache={}, is_final=False, chunk_size=500
            )
            end_event.record()
            if self.device.type == "mps":
                torch.mps.synchronize()
            comp_times["vad"] = start_event.elapsed_time(end_event)
            
            # Content encoder
            if self.device.type == "mps":
                torch.mps.synchronize()
            start_event.record()
            S_alt = self.semantic_fn(audio_16k.unsqueeze(0))
            end_event.record()
            if self.device.type == "mps":
                torch.mps.synchronize()
            comp_times["content_encoder"] = start_event.elapsed_time(end_event)
            
            # Length regulator
            if self.device.type == "mps":
                torch.mps.synchronize()
            start_event.record()
            ce_dit_frame_difference = int(2.0 * 50)  # 2 seconds
            S_alt_trimmed = S_alt[:, ce_dit_frame_difference:]
            target_lengths = torch.LongTensor([S_alt_trimmed.size(1)]).to(S_alt.device)
            cond = self.model.length_regulator(
                S_alt_trimmed, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            end_event.record()
            if self.device.type == "mps":
                torch.mps.synchronize()
            comp_times["length_regulator"] = start_event.elapsed_time(end_event)
            
            # DiT inference
            if self.device.type == "mps":
                torch.mps.synchronize()
            start_event.record()
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16 if self.device.type != "mps" else torch.float32
                ):
                    vc_target = self.model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(mel_ref.device),
                        mel_ref,
                        style_ref,
                        None,
                        n_timesteps=diffusion_steps,
                        inference_cfg_rate=0.7,
                    )
                    vc_target = vc_target[:, :, mel_ref.size(-1):]
            end_event.record()
            if self.device.type == "mps":
                torch.mps.synchronize()
            comp_times["dit"] = start_event.elapsed_time(end_event)
            
            # Vocoder
            if self.device.type == "mps":
                torch.mps.synchronize()
            start_event.record()
            with torch.no_grad():
                vc_wave = self.vocoder(vc_target).squeeze()
            end_event.record()
            if self.device.type == "mps":
                torch.mps.synchronize()
            comp_times["vocoder"] = start_event.elapsed_time(end_event)
            
            run_end = time.perf_counter()
            total_time_ms = (run_end - run_start) * 1000
            
            latencies.append(total_time_ms)
            for key, value in comp_times.items():
                component_times[key].append(value)
        
        # Average component times
        avg_component_times = {
            key: np.mean(values) for key, values in component_times.items()
        }
        avg_component_times["total"] = np.mean(latencies)
        
        return latencies, avg_component_times
    
    def _single_inference(self, audio_16k, prompt_condition, mel_ref, style_ref,
                         diffusion_steps, warmup=False):
        """Single inference pass (used for warmup)."""
        ce_dit_frame_difference = int(2.0 * 50)
        S_alt = self.semantic_fn(audio_16k.unsqueeze(0))
        S_alt_trimmed = S_alt[:, ce_dit_frame_difference:]
        target_lengths = torch.LongTensor([S_alt_trimmed.size(1)]).to(S_alt.device)
        cond = self.model.length_regulator(
            S_alt_trimmed, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.device.type != "mps" else torch.float32
            ):
                vc_target = self.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel_ref.device),
                    mel_ref,
                    style_ref,
                    None,
                    n_timesteps=diffusion_steps,
                    inference_cfg_rate=0.7,
                )


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_benchmark(model_config: ModelConfig, diffusion_steps: int,
                  block_time: float, test_audio: Tuple[np.ndarray, float],
                  reference_audio: Tuple[np.ndarray, float],
                  system_specs: Dict, num_runs: int = 5) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    
    sr_model = test_audio[1]
    test_wav, _ = test_audio
    ref_wav, _ = reference_audio
    
    # Create result object
    result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model_name=model_config.name,
        diffusion_steps=diffusion_steps,
        block_time=block_time,
        num_runs=num_runs,
        system_specs=system_specs
    )
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load model
    runner = BenchmarkRunner(device)
    if not runner.load_model(model_config):
        result.error = f"Failed to load model {model_config.name}"
        return result
    
    # Prepare test audio (resample to 16k for VAD/content encoder)
    test_wav_16k = librosa.resample(test_wav, orig_sr=sr_model, target_sr=16000)
    test_wav_16k_tensor = torch.from_numpy(test_wav_16k).to(device)
    
    # Run benchmark
    try:
        latencies, component_times = runner.benchmark_inference(
            test_wav_16k_tensor, ref_wav, diffusion_steps, num_runs
        )
        
        # Compute statistics
        result.latencies_ms = latencies
        result.mean_latency_ms = np.mean(latencies)
        result.std_latency_ms = np.std(latencies)
        result.min_latency_ms = np.min(latencies)
        result.max_latency_ms = np.max(latencies)
        result.p50_latency_ms = np.percentile(latencies, 50)
        result.p90_latency_ms = np.percentile(latencies, 90)
        result.p99_latency_ms = np.percentile(latencies, 99)
        result.rtf = result.mean_latency_ms / (block_time * 1000)
        result.component_latencies = component_times
        
        # Feasibility check (with 20% headroom for mean, strict for P99)
        result.feasible_mean = result.mean_latency_ms < block_time * 0.8 * 1000
        result.feasible_p99 = result.p99_latency_ms < block_time * 1000
        result.feasible = result.feasible_mean and result.feasible_p99
        
    except Exception as e:
        result.error = f"Benchmark error: {str(e)}"
        import traceback
        traceback.print_exc()
    
    return result


def print_system_specs(specs: Dict):
    """Print system specifications."""
    print("\n" + "=" * 80)
    print("SYSTEM SPECIFICATIONS")
    print("=" * 80)
    print(f"OS: {specs.get('os', 'N/A')} {specs.get('os_version', 'N/A')}")
    print(f"Chip: {specs.get('chip', 'Unknown')}")
    print(f"CPU: {specs.get('cpu_count_physical', 'N/A')} physical / {specs.get('cpu_count_logical', 'N/A')} logical")
    print(f"Memory: {specs.get('memory_total_gb', 0):.1f}GB total, {specs.get('memory_available_gb', 0):.1f}GB available")
    print(f"PyTorch: {specs.get('torch_version', 'N/A')}")
    print(f"Device: {specs.get('device', 'N/A')}")
    if specs.get('mps_available'):
        print(f"MPS: Available ✅")
    if specs.get('cuda_available'):
        print(f"CUDA: Available ✅ ({specs.get('cuda_device', 'N/A')})")
    print("=" * 80 + "\n")


def print_result(result: BenchmarkResult, verbose=False):
    """Print benchmark result."""
    status = "✅" if result.feasible else ("⚠️" if result.feasible_mean else "❌")
    
    print(f"\n{status} {result.model_name} ({result.diffusion_steps} steps)")
    print("-" * 60)
    print(f"Block time: {result.block_time:.2f}s ({result.block_time * 1000:.0f}ms)")
    print(f"Latency - Mean: {result.mean_latency_ms:.0f}ms, P50: {result.p50_latency_ms:.0f}ms, "
          f"P90: {result.p90_latency_ms:.0f}ms, P99: {result.p99_latency_ms:.0f}ms")
    print(f"RTF: {result.rtf:.3f} (Real-Time Factor)")
    
    if result.component_latencies:
        print(f"Component breakdown:")
        print(f"  - Content Encoder: {result.component_latencies.get('content_encoder', 0):.0f}ms")
        print(f"  - DiT Inference: {result.component_latencies.get('dit', 0):.0f}ms")
        print(f"  - Vocoder: {result.component_latencies.get('vocoder', 0):.0f}ms")
        print(f"  - Length Regulator: {result.component_latencies.get('length_regulator', 0):.0f}ms")
        print(f"  - VAD: {result.component_latencies.get('vad', 0):.0f}ms")
    
    print(f"Feasibility - Mean < 80% block: {'✅' if result.feasible_mean else '❌'}, "
          f"P99 < 100% block: {'✅' if result.feasible_p99 else '❌'}")
    
    if result.error:
        print(f"Error: {result.error}")


def print_summary(results: List[BenchmarkResult]):
    """Print summary table of all results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Steps':>6} {'Block':>8} {'Mean(ms)':>10} {'P99(ms)':>10} "
          f"{'RTF':>8} {'Feasible':>10}")
    print("-" * 80)
    
    for r in results:
        if r.error:
            print(f"{r.model_name:<20} {r.diffusion_steps:>6} {'N/A':>8} "
                  f"{'ERROR':>10} {'-':>10} {'-':>8} {'❌':>10}")
        else:
            feasible = "✅" if r.feasible else "❌"
            print(f"{r.model_name:<20} {r.diffusion_steps:>6} "
                  f"{r.block_time:>7.2f}s {r.mean_latency_ms:>9.0f} "
                  f"{r.p99_latency_ms:>9.0f} {r.rtf:>7.3f} {feasible:>10}")
    
    print("=" * 80)
    
    # Count feasible configs
    feasible_count = sum(1 for r in results if r.feasible)
    print(f"\nFeasible configurations: {feasible_count}/{len(results)}")
    
    # Best configuration
    feasible_results = [r for r in results if r.feasible]
    if feasible_results:
        best = min(feasible_results, key=lambda x: x.rtf)
        print(f"Best configuration: {best.model_name} with {best.diffusion_steps} steps "
              f"(RTF: {best.rtf:.3f}, Latency: {best.mean_latency_ms:.0f}ms)")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Seed-VC Real-Time Inference Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Model to benchmark (default: all models)"
    )
    
    parser.add_argument(
        "--steps", "-s",
        type=str,
        default="4,6,8,10",
        help="Diffusion steps to test (comma-separated, default: 4,6,8,10)"
    )
    
    parser.add_argument(
        "--block-time", "-b",
        type=float,
        default=0.20,
        help="Block time in seconds (default: 0.20)"
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=5,
        help="Number of benchmark runs per config (default: 5)"
    )
    
    parser.add_argument(
        "--test-audio",
        type=str,
        default="examples/source/source_1.wav",
        help="Path to test audio file (default: examples/source/source_1.wav)"
    )
    
    parser.add_argument(
        "--reference-audio",
        type=str,
        default="examples/reference/ref_1.wav",
        help="Path to reference audio file (default: examples/reference/ref_1.wav)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file (default: benchmark_results_YYYYMMDD_HHMMSS.json)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse steps
    steps = [int(s.strip()) for s in args.steps.split(",")]
    
    # Detect system
    print("Detecting system specifications...")
    system_specs = detect_system()
    print_system_specs(system_specs)
    
    # Check device availability
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("❌ Error: No MPS or CUDA device available. Benchmark requires GPU.")
        sys.exit(1)
    
    # Load test audio
    print(f"Loading test audio: {args.test_audio}")
    print(f"Loading reference audio: {args.reference_audio}")
    
    if not Path(args.test_audio).exists():
        print(f"❌ Test audio not found: {args.test_audio}")
        sys.exit(1)
    
    if not Path(args.reference_audio).exists():
        print(f"❌ Reference audio not found: {args.reference_audio}")
        sys.exit(1)
    
    test_wav, test_sr = librosa.load(args.test_audio, sr=None)
    ref_wav, ref_sr = librosa.load(args.reference_audio, sr=None)
    
    print(f"Test audio: {len(test_wav) / test_sr:.1f}s at {test_sr}Hz")
    print(f"Reference audio: {len(ref_wav) / ref_sr:.1f}s at {ref_sr}Hz")
    
    # Select models to benchmark
    if args.model:
        models_to_benchmark = [MODEL_CONFIGS[args.model]]
    else:
        models_to_benchmark = list(MODEL_CONFIGS.values())
    
    print(f"\nBenchmarking {len(models_to_benchmark)} model(s) with {len(steps)} step configuration(s)")
    print(f"Block time: {args.block_time}s, Runs per config: {args.runs}")
    
    # Run benchmarks
    results = []
    for model_config in models_to_benchmark:
        print(f"\n{'='*80}")
        print(f"Model: {model_config.name}")
        print(f"Description: {model_config.description}")
        print(f"{'='*80}")
        
        for steps in steps:
            result = run_benchmark(
                model_config=model_config,
                diffusion_steps=steps,
                block_time=args.block_time,
                test_audio=(test_wav, test_sr),
                reference_audio=(ref_wav, ref_sr),
                system_specs=system_specs,
                num_runs=args.runs
            )
            results.append(result)
            print_result(result, verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chip_name = system_specs.get('chip', 'unknown').replace(' ', '_').replace('/', '_')
        output_path = f"benchmark_results_{chip_name}_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    # Return exit code based on feasibility
    feasible_count = sum(1 for r in results if r.feasible)
    if feasible_count > 0:
        print(f"\n✅ {feasible_count} configuration(s) validated for real-time use")
        sys.exit(0)
    else:
        print(f"\n❌ No configurations validated for real-time use on this system")
        sys.exit(1)


if __name__ == "__main__":
    main()
