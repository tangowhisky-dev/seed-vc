# Seed-VC Training and Inference Pipelines

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Training Pipeline](#training-pipeline)
4. [Inference Pipelines](#inference-pipelines)
5. [Model Variants](#model-variants)
6. [Technical Deep Dive](#technical-deep-dive)
7. [Component Details](#component-details)

---

## Project Overview

Seed-VC is a state-of-the-art zero-shot voice conversion system developed for high-quality voice cloning and conversion. The project supports:

- **Zero-shot voice conversion** - Clone a voice from a 1-30 second reference audio
- **Zero-shot real-time voice conversion** - For online meetings, gaming, live streaming (~300ms algorithm delay + ~100ms device delay)
- **Zero-shot singing voice conversion** - Convert singing voices between different singers
- **Voice/accent conversion (V2)** - Best at suppressing source speaker traits

Published on [HuggingFace](https://huggingface.co/Plachta/Seed-VC) with an [arXiv paper](https://arxiv.org/abs/2411.09943).

---

## Architecture Overview

### High-Level Architecture (V1)

```
Source Audio → Content Encoder (Whisper/XLSR) → Length Regulator
                                                            ↓
Reference Audio → Style Encoder (CAMPPlus) → ──────────→ DiT (CFM)
                                                            ↓
                                                        Mel Spectrogram
                                                            ↓
                                                        Vocoder (BigVGAN/HiFi-GAN)
                                                            ↓
                                                    Converted Audio
```

### V2 Architecture (CFM + AR)

```
Source Audio → ASTRAL Content Extractor (Wide + Narrow)
                                              ↓
                                      AR Model (autoregressive)
                                              ↓
Source Audio → Style Encoder (CAMPPlus) → ─────────→ CFM (Flow Matching)
                                              ↓
                                          Vocoder → Output
```

---

## Training Pipeline

### V1 Training (`train.py`)

The V1 training pipeline is designed for fine-tuning the pre-trained Seed-VC models on custom datasets. The training process involves several key stages:

#### 1. Data Preparation

**Supported Formats:**
- `.wav`, `.mp3`, `.flac`, `.m4a`, `.opus`, `.ogg`

**Data Requirements:**
- Audio duration: 1-30 seconds per file
- Minimum: 1 utterance per speaker
- Cleaner data (no BGM or noise preferred)
- File structure is flexible

#### 2. Data Loading (`data/ft_dataset.py`)

The `FT_Dataset` class handles audio loading with:
- Mel spectrogram extraction using config parameters
- Batch processing with configurable batch size
- Multi-worker data loading support

#### 3. Model Components

The training pipeline builds the following components:

| Component | Type | Purpose |
|-----------|------|---------|
| **Content Encoder** | Whisper-small or XLSR-large | Extract semantic features from speech |
| **Style Encoder** | CAMPPlus (192-dim) | Extract speaker embeddings |
| **Length Regulator** | InterpolateRegulator | Align content to target duration |
| **Diffusion Model** | DiT with CFM | Generate mel spectrograms |
| **Vocoder** | BigVGAN or HiFi-GAN | Convert mel to waveform |

#### 4. Training Process

**Key Steps in `train_one_step()`:**

```python
# 1. Load batch (waves, mels, wave_lengths, mel_input_length)
waves, mels, wave_lengths, mel_input_length = batch

# 2. Extract speaker embeddings using ToneColorConverter
se_batch = self.tone_color_converter.extract_se(waves_22k, wave_lengths_22k)

# 3. Convert waves using reference speaker embeddings
converted_waves_22k = self.tone_color_converter.convert(
    waves_22k, wave_lengths_22k, se_batch, ref_se
).squeeze(1)

# 4. Extract semantic features (S_alt and S_ori)
S_ori = self.semantic_fn(waves_16k)      # Original speech tokens
S_alt = self.semantic_fn(converted_waves_16k)  # Converted speech tokens

# 5. Extract F0 if f0_condition is enabled
if self.f0_condition:
    F0_ori = self.rmvpe.infer_from_audio_batch(waves_16k)

# 6. Length regulation
alt_cond, _, alt_codes, alt_commitment_loss, alt_codebook_loss = (
    self.model.length_regulator(S_alt, ylens=target_lengths, f0=F0_ori)
)

# 7. Random prompt length during training
prompt_len_max = target_lengths - 1
prompt_len = (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0

# 8. Apply prompt conditioning
cond = alt_cond.clone()
for bib in range(B):
    cond[bib, :prompt_len[bib]] = ori_cond[bib, :prompt_len[bib]]

# 9. Extract style vectors from prompt
feat = kaldi.fbank(waves_16k[bib:bib+1], num_mel_bins=80, ...)
y = self.sv_fn(feat)  # CAMPPlus speaker embeddings

# 10. Diffusion training with CFM
loss, _ = self.model.cfm(x, target_lengths, prompt_len, cond, y)

# 11. Combined loss
loss_total = (
    loss +
    (alt_commitment_loss + ori_commitment_loss) * 0.05 +
    (ori_codebook_loss + alt_codebook_loss) * 0.15
)
```

#### 5. Training Configuration

**Command:**
```bash
python train.py 
--config <config-path>
--dataset-dir <data-dir>
--run-name <name>
--batch-size 2
--max-steps 1000
--max-epochs 1000
--save-every 500
--num-workers 0
```

**Config Files (in `configs/presets/`):**

| Config | Purpose | Content Encoder | Vocoder | Parameters |
|--------|---------|-----------------|---------|------------|
| `config_dit_mel_seed_uvit_xlsr_tiny.yml` | Real-time VC | XLSR-large | HiFi-GAN | 25M |
| `config_dit_mel_seed_uvit_whisper_small_wavenet.yml` | Offline VC | Whisper-small | BigVGAN | 98M |
| `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` | Singing VC | Whisper-small | BigVGAN | 200M |

#### 6. Checkpointing

- Checkpoints saved every N steps (configurable via `--save-every`)
- Format: `DiT_epoch_XXXXX_step_XXXXX.pth`
- Latest checkpoint auto-resumes training
- Old checkpoints auto-cleaned (keeps latest 2)

#### 7. Optimizer Configuration

From `optimizers.py`:
- Learning rate: 0.00001 (base_lr)
- Warmup steps: 0
- Gradient clipping: 10.0 for both CFM and length_regulator
- Loss smoothing: 0.99

---

### V2 Training (`train_v2.py`)

The V2 training pipeline supports multi-GPU training via `accelerate` and includes:

#### 1. Separate Model Training

- **CFM Model** - Conditional Flow Matching for content generation
- **AR Model** - Autoregressive model for style/emotion/accent conversion

#### 2. Hydra-based Configuration

Uses Hydra for flexible configuration management:
```bash
accelerate launch train_v2.py 
--dataset-dir <data-dir>
--run-name <name>
--batch-size 2
--train-cfm  # Train CFM model
--train-ar   # Train AR model (optional)
```

#### 3. ASTRAL Content Extraction

V2 uses ASTRAL quantization for speaker-disentangled content representation:
- Wide content extractor for coarse features
- Narrow content extractor for fine details

---

## Inference Pipelines

### Offline Inference (`inference.py`)

The offline inference pipeline processes the entire audio at once:

#### Pipeline Flow:

```
1. Load Source & Reference Audio
   ↓
2. Resample to Model Sampling Rate (22050 or 44100 Hz)
   ↓
3. Extract Content Features (Whisper/XLSR)
   - S_alt: Source content representation
   - S_ori: Reference content representation
   ↓
4. Extract Style Features (CAMPPlus)
   - 192-dimensional speaker embedding
   ↓
5. Extract F0 (if f0_condition enabled)
   - Using RMVPE pitch extraction
   ↓
6. Length Regulation
   - Align content to target duration
   ↓
7. Diffusion Inference (CFM)
   - Multi-step denoising process
   - Classifier-free guidance
   ↓
8. Vocoder Processing
   - Mel → Waveform conversion
   ↓
9. Crossfade Chunking (for long audio)
   ↓
10. Output Audio
```

#### Key Parameters:

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--diffusion-steps` | Number of diffusion steps | 25-50 (higher = better quality) |
| `--length-adjust` | Speed/slow factor | 1.0 (default) |
| `--inference-cfg-rate` | CFG classifier-free guidance rate | 0.7 |
| `--f0-condition` | Enable pitch conditioning | False for VC, True for SVC |
| `--auto-f0-adjust` | Auto-adjust pitch to target | False |
| `--semi-tone-shift` | Pitch shift in semitones | 0 |
| `--fp16` | Use float16 inference | True |

#### Processing Logic (from `inference.py` lines 256-408):

```python
# 1. Load and resample audio
source_audio = librosa.load(source, sr=sr)[0]
ref_audio = librosa.load(target_name, sr=sr)[0]

# 2. Extract content features (Whisper/XLSR)
S_alt = semantic_fn(converted_waves_16k)  # Source content
S_ori = semantic_fn(ori_waves_16k)         # Reference content

# 3. Extract mel spectrograms
mel = mel_fn(source_audio)
mel2 = mel_fn(ref_audio)

# 4. Extract style embeddings
feat2 = kaldi.fbank(ori_waves_16k, num_mel_bins=80)
style2 = campplus_model(feat2)

# 5. F0 extraction (if enabled)
if f0_condition:
    F0_ori = f0_fn(ori_waves_16k[0])
    F0_alt = f0_fn(converted_waves_16k[0])

# 6. Length regulation
cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(
    S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
)
prompt_condition, _, codes, ... = model.length_regulator(
    S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
)

# 7. Chunked diffusion with crossfade
while processed_frames < cond.size(1):
    chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
    
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2, style2, None, diffusion_steps,
            inference_cfg_rate=inference_cfg_rate
        )
    
    vc_wave = vocoder_fn(vc_target.float()).squeeze()
    # Apply crossfade between chunks
```

#### Real-time Factor (RTF) Calculation:

```python
# RTF = Processing Time / Audio Duration
# RTF < 1.0 means faster than real-time
# RTF = (time_vc_end - time_vc_start) / vc_wave.size(-1) * sr
```

---

### Real-Time Inference (`real-time-gui.py`)

The real-time pipeline is optimized for low-latency voice conversion:

#### Architecture:

```
Input Audio → VAD (Voice Activity Detection)
                    ↓
         Chunk Processing (configurable block time)
                    ↓
         Content Encoder (Whisper/XLSR)
                    ↓
         Length Regulation
                    ↓
         DiT Diffusion (reduced steps: 4-10)
                    ↓
         Vocoder (BigVGAN/HiFi-GAN)
                    ↓
         SOLA (Signal Overlap-Add) for smooth transitions
                    ↓
         Output Audio
```

#### Key Parameters:

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `block_time` | Time length per audio chunk | 0.18-0.25s |
| `crossfade_time` | Crossfade between chunks | 0.04s |
| `extra_time_ce` | Content encoder context (left) | 2.5s |
| `extra_time` | DiT context (left) | 0.5s |
| `extra_time_right` | Future context (right) | 0.02-2.0s |
| `diffusion_steps` | Diffusion steps | 4-10 (fast) |
| `inference_cfg_rate` | CFG rate | 0.7 |
| `max_prompt_length` | Max reference audio length | 3.0s |

#### Latency Calculation:

```
Algorithm Delay = Block Time × 2 + Extra Context (right)
Device Delay ≈ 100ms
Total Delay = Algorithm Delay + Device Delay

Example: 0.18s × 2 + 0.02s + 0.1s ≈ 430ms
```

#### Performance Targets:

| Model Configuration | Diffusion Steps | Block Time | Latency | Inference Time |
|-------------------|----------------|------------|---------|----------------|
| seed-uvit-xlsr-tiny | 10 | 0.18s | 430ms | 150ms |

**Critical:** Inference time must be less than block time for stable real-time operation.

#### SOLA Algorithm:

The real-time pipeline uses Signal Overlap-Add (SOLA) for smooth audio transitions:

```python
# Convolution-based alignment
cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
cor_den = torch.sqrt(F.conv1d(conv_input**2, ones) + 1e-8)
tensor = cor_nom[0, 0] / cor_den[0, 0]
sola_offset = torch.argmax(tensor, dim=0).item()

# Apply fade windows
infer_wav[sola_offset:][:sola_buffer_frame] *= fade_in_window
infer_wav[sola_offset:][:sola_buffer_frame] += sola_buffer * fade_out_window
```

---

### V2 Inference (`inference_v2.py`)

The V2 model adds autoregressive capabilities for enhanced voice conversion:

#### Enhanced Features:

1. **Dual CFG Rates:**
   - `intelligibility-cfg-rate`: Controls linguistic clarity (0.0-1.0)
   - `similarity-cfg-rate`: Controls voice similarity (0.0-1.0)

2. **AR Model Integration:**
   - Style/emotion/accent conversion
   - Configurable with `--convert-style`

3. **ASTRAL Content Extraction:**
   - Speaker-disentangled content representation
   - Wide + Narrow extractors

4. **TorchCompile Support:**
   - `--compile` flag for ~6x speedup on AR model

#### Command:

```bash
python inference_v2.py 
--source <source-wav>
--target <reference-wav>
--output <output-dir>
--diffusion-steps 25
--intelligibility-cfg-rate 0.7
--similarity-cfg-rate 0.7
--convert-style true
--top-p 0.9
--temperature 1.0
--repetition-penalty 1.0
--compile
```

---

## Model Variants

### Available Models

| Version | Model Name | Purpose | SR | Content Encoder | Vocoder | Params |
|---------|-----------|---------|-----|-----------------|---------|--------|
| v1.0 | seed-uvit-tat-xlsr-tiny | Real-time VC | 22050 | XLSR-large | HiFi-GAN | 25M |
| v1.0 | seed-uvit-whisper-small-wavenet | Offline VC | 22050 | Whisper-small | BigVGAN | 98M |
| v1.0 | seed-uvit-whisper-base | SVC | 44100 | Whisper-small | BigVGAN | 200M |
| v2.0 | hubert-bsqvae-small | Voice/Accent | 22050 | ASTRAL | BigVGAN | 67M+90M |

### Model Selection Guide:

- **Real-time applications**: Use `seed-uvit-tat-xlsr-tiny` (25M params)
- **High-quality offline VC**: Use `seed-uvit-whisper-small-wavenet` (98M params)
- **Singing voice conversion**: Use `seed-uvit-whisper-base` (200M params)
- **Voice anonymization/accent conversion**: Use V2 model

---

## Technical Deep Dive

### Conditional Flow Matching (CFM)

The core diffusion mechanism uses flow matching rather than traditional score-based diffusion:

```python
# Flow matching loss
def compute_loss(x, target_lengths, prompt_len, cond, y):
    # x: target mel spectrogram
    # cond: content tokens (source + prompt)
    # y: style embeddings (CAMPPlus)
    
    t = torch.rand_like(x)  # Random time in [0, 1]
    xt = t * x  # Interpolated flow
    
    # Predict velocity
    v = model(xt, t, cond, y)
    
    # Flow matching loss
    loss = (v - x) ** 2
    return loss.mean()
```

### Multi-Condition Classifier-Free Guidance

During inference, CFG is applied with multiple conditions:

```python
# CFG during inference
vc_target = model.cfm.inference(
    cat_condition,           # Content + prompt
    target_length,
    mel_prompt,              # Mel spectrogram prompt
    style2,                 # Style embeddings
    None,                   # F0 (optional)
    diffusion_steps,
    inference_cfg_rate=0.7  # CFG strength
)
```

### Length Regulation

The length regulator aligns content representations to target durations:

```python
# Interpolate to target length
cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(
    S_alt,                          # Content tokens
    ylens=target_lengths,           # Target lengths
    n_quantizers=3,                 # Number of quantizers
    f0=shifted_f0_alt              # F0 conditioning (optional)
)
```

---

## Component Details

### 1. Content Encoders

| Encoder | Type | Output Dim | Use Case |
|---------|------|------------|----------|
| Whisper-small | Transformer | 768 | High-quality VC, SVC |
| XLSR-large | Conformer | 1024 | Real-time VC |
| CNHuBERT | Transformer | 768 | Alternative |

### 2. Style Encoder (CAMPPlus)

- **Input**: 80-dimensional mel spectrogram features
- **Output**: 192-dimensional speaker embedding
- **Model**: TDNN-based speaker verification network

### 3. Vocoders

| Vocoder | Type | Sampling Rate | Quality |
|---------|------|---------------|---------|
| BigVGAN | GAN | 22kHz/44kHz | High |
| HiFi-GAN | GAN | 22kHz | Medium |
| HIFT | Griffin-Lim | 22kHz | Low (fast) |

### 4. F0 Extraction (RMVPE)

- **Model**: E2E pitch estimation (4→1 architecture)
- **Input**: 30-8000 Hz mel spectrogram
- **Output**: F0 contour in Hz
- **Half-precision**: Supported with `is_half=False` for MPS compatibility

---

## Performance Characteristics

### Memory Requirements

| Model | Parameters | GPU Memory (FP16) |
|-------|------------|-------------------|
| seed-uvit-tat-xlsr-tiny | 25M | ~2GB |
| seed-uvit-whisper-small-wavenet | 98M | ~4GB |
| seed-uvit-whisper-base | 200M | ~8GB |
| V2 (CFM + AR) | 67M + 90M | ~8GB |

### Inference Speed (RTF)

| Model | GPU | RTF (25 steps) |
|-------|-----|----------------|
| seed-uvit-tat-xlsr-tiny | RTX 3060 | ~0.1 |
| seed-uvit-whisper-small-wavenet | RTX 3060 | ~0.3 |
| seed-uvit-whisper-base | RTX 3060 | ~0.5 |

---

## References

- [Seed-VC Paper (arXiv:2411.09943)](https://arxiv.org/abs/2411.09943)
- [HuggingFace Model Hub](https://huggingface.co/Plachta/Seed-VC)
- [Demo Page](https://plachtaa.github.io/seed-vc/)
