# Fix: Model Detection in download_all_models.py

## Issue

**Problem**: Script showed "❌ Not downloaded" for all models even though some were already downloaded.

**Root Cause**: The script was only checking `blobs/` directory, but models were stored in `snapshots/` directory.

**Actual structure**:
```
checkpoints/
└── models--Plachta--Seed-VC/
    ├── snapshots/
    │   └── 257283f9f41585055e8f858fba4fd044e5caed6e/
    │       ├── DiT_uvit_tat_xlsr_ema.pth          ← xlsr_tiny
    │       └── v2/
    │           ├── cfm_small.pth                   ← v2_cfm_small
    │           └── ar_base.pth                     ← v2_ar_base
    └── blobs/                                      ← Empty or different models
```

## Fix Applied

### 1. Updated `print_models_table()` function

**Before**: Only checked `blobs/` directory
```python
checkpoint_path = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "blobs" / info["checkpoint"]
status = "✅ Downloaded" if checkpoint_path.exists() else "❌ Not downloaded"
```

**After**: Checks multiple locations
```python
# Search in blobs directory
checkpoint_blob_dir = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "blobs"
if checkpoint_blob_dir.exists():
    for blob_file in checkpoint_blob_dir.iterdir():
        if blob_file.is_file() and info["checkpoint"].split(".")[-1] in blob_file.name:
            checkpoint_exists = True
            break

# Search in snapshots directory
if not checkpoint_exists:
    checkpoint_snap_dir = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "snapshots"
    if checkpoint_snap_dir.exists():
        for snapshot_subdir in checkpoint_snap_dir.iterdir():
            if snapshot_subdir.is_dir():
                target_file = snapshot_subdir / info["checkpoint"]
                if target_file.exists():
                    checkpoint_exists = True
                    break
                # Also check v2 subdirectory
                v2_dir = snapshot_subdir / "v2"
                if v2_dir.exists():
                    # Check for CFM or AR models
                    if name in ["v2_cfm_small", "v2_ar_base"]:
                        model_type = "cfm" if "cfm" in name else "ar"
                        v2_file = v2_dir / f"{model_type}_small.pth"
                        if v2_file.exists() or (model_type == "ar" and (v2_dir / "ar_base.pth").exists()):
                            checkpoint_exists = True
                            break

# Also check if file exists directly in checkpoints
if not checkpoint_exists:
    direct_path = CHECKPOINTS_DIR / info["checkpoint"]
    if direct_path.exists():
        checkpoint_exists = True
```

### 2. Added `check_model_downloaded()` helper function

New function that centralizes the detection logic:
```python
def check_model_downloaded(model_name: str) -> bool:
    """Check if a model is already downloaded."""
    # Search in blobs, snapshots, v2/, and direct paths
    # Returns True if found anywhere
```

### 3. Updated `download_model()` function

**Before**: Checked only blobs directory
```python
checkpoint_blob_dir = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "blobs"
checkpoint_exists = any(checkpoint_blob_dir.glob("*" + model["checkpoint"].split(".")[-1]))
```

**After**: Uses new helper function
```python
if check_model_downloaded(model_name):
    if verbose:
        print(f"  ⏭️  Model already downloaded, skipping...")
    return True  # Count as successful
```

## Result

### Before Fix
```
Model Name                 Size    Version Purpose                             Status
------------------------------------------------------------------------------------------------------------------------
xlsr_tiny                  135MB       v1.0 ... ❌ Not downloaded
v2_cfm_small               337MB       v1.0 ... ❌ Not downloaded
v2_ar_base                 342MB       v1.0 ... ❌ Not downloaded
```

### After Fix
```
Model Name                 Size    Version Purpose                             Status
------------------------------------------------------------------------------------------------------------------------
xlsr_tiny                  135MB       v1.0 ... ✅ Downloaded
v2_cfm_small               337MB       v1.0 ... ✅ Downloaded
v2_ar_base                 342MB       v1.0 ... ✅ Downloaded
whisper_small              342MB       v1.0 ... ❌ Not downloaded
whisper_base               850MB       v1.0 ... ❌ Not downloaded
```

## Testing

```bash
# List models (shows correct status)
python tools/download_all_models.py --list

# Download remaining models
python tools/download_all_models.py --all

# Will skip already-downloaded models:
# ⏭️  Model already downloaded, skipping...
```

## Files Modified

- `tools/download_all_models.py`:
  - Added `check_model_downloaded()` function
  - Updated `print_models_table()` to search multiple locations
  - Updated `download_model()` to use new helper function

## Status

✅ **Fixed**: Model detection now works correctly
✅ **Tested**: Correctly identifies downloaded vs not downloaded models
✅ **Ready**: Can now download remaining models (whisper_small, whisper_base)
