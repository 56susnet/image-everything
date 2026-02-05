#!/usr/bin/env python3
"""
dancil
Gradient Image Tourney
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml
import shutil
import glob
import random
import math

script_dir = os.path.dirname(os.path.abspath(_file_))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType
from trainer.utils.trainer_downloader import download_image_dataset, download_base_model

OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
CUDA_ERROR = "out of memory"
ERRRR = "cuda out of memory"

def count_images_in_dataset(train_data_dir: str) -> int:
    """Count images in the training dataset directory"""
    num_images = 0
    img_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for root, _, files in os.walk(train_data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_extensions:
                num_images += 1
    return num_images


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        # If it's a diffusers repo, it should have a model_index.json or transformer folder
        if os.path.exists(os.path.join(path, "model_index.json")) or os.path.exists(os.path.join(path, "transformer")):
            return path
            
        # Otherwise, look for the largest safetensors file (likely a single-file checkpoint)
        safetensors_files = glob.glob(os.path.join(path, "*.safetensors"))
        if safetensors_files:
            # Pick the largest one
            return max(safetensors_files, key=os.path.getsize)
    return path



def merge_model_config(default_config: dict, model_config: dict) -> dict:
    """Merge default config with model-specific overrides."""
    merged = {}
    if isinstance(default_config, dict):
        merged.update(default_config)
    if isinstance(model_config, dict):
        merged.update(model_config)
    return merged if merged else None


def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    """Get configuration overrides based on model name."""
    if not isinstance(lrs_config, dict):
        return None
    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})
    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))
    if default_config:
        return default_config
    return None


def load_lrs_config(model_type: str, is_style: bool) -> dict:
    """Load the appropriate LRS configuration based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(_file_))
    config_dir = os.path.join(script_dir, "lrs")
    
    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def load_size_config(model_type: str, is_style: bool = False) -> dict:
    """Load the appropriate Size LRS configuration based on training type"""
    script_dir = os.path.dirname(os.path.abspath(_file_))
    config_dir = os.path.join(script_dir, "lrs")
    
    if is_style:
        config_file = os.path.join(config_dir, "size_style.json")
    else:
        config_file = os.path.join(config_dir, "size_person.json")
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Warning: Could not load Size LRS config from {config_file}: {e}", flush=True)
        return None


def get_config_for_size(size_config: dict, num_images: int) -> dict:
    """Get configuration overrides based on dataset size."""
    if not isinstance(size_config, dict):
        return None
        
    size_ranges = size_config.get("size_ranges", [])
    for range_item in size_ranges:
        min_val = range_item.get("min", 1)
        max_val = range_item.get("max", 999999)
        
        if min_val <= num_images <= max_val:
            return range_item.get("config", {})
            
    return None

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, hours_to_complete, trigger_word: str | None = None, dim_scale: float = 1.0):
    """Create the diffusion config file"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)
    
    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir
                
                if trigger_word:
                    process['trigger_word'] = trigger_word



        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'network' in process and dim_scale != 1.0:
                    net = process['network']
                    if 'linear' in net:
                        net['linear'] = max(1, int(net['linear'] * dim_scale))
                    if 'linear_alpha' in net:
                        net['linear_alpha'] = max(1, int(net['linear_alpha'] * dim_scale))
                    if 'conv' in net:
                        net['conv'] = max(1, int(net['conv'] * dim_scale))
                    if 'conv_alpha' in net:
                        net['conv_alpha'] = max(1, int(net['conv_alpha'] * dim_scale))
                    print(f"Scaled Ai-Toolkit network dims by {dim_scale}", flush=True)

        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
        return config_path, output_dir
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)
        
        lrs_overrides = set()
        
        # --- LRS LOGIC (DISABLED BY DEFAULT) ---
        # To enable, uncomment the block below:
        
        # SDXL Specific Logic: Dynamic Epochs & Size-Based Config
        if model_type == ImageModelType.SDXL.value:
            num_images = count_images_in_dataset(train_data_dir)
            print(f"Found {num_images} images in dataset.", flush=True)

            # 1. Dynamic Epochs
            if 1 <= num_images <= 20:
                total_epochs = 55
            elif num_images > 20 and hours_to_complete <= 1.0:
                total_epochs = 35
            elif 20 < num_images <= 50:
                total_epochs = 65
            else:
                total_epochs = int(config.get("max_train_epochs", 0))

            config["max_train_epochs"] = total_epochs
            config["save_every_n_epochs"] = max(1, math.ceil(total_epochs / 4))

            # 2. Load Size Config (Medium Priority)
            size_config = load_size_config(model_type, is_style)
            if size_config:
                size_settings = get_config_for_size(size_config, num_images)
                if size_settings:
                    print(f"Applying Size-Based Config for {num_images} images...", flush=True)
                    for key, value in size_settings.items():
                        # explicitly allow overriding these keys from size config
                        if key in [
                            "prior_loss_weight", "min_snr_gamma", "train_batch_size", 
                            "optimizer_args", "scale_weight_norms", "network_dim", "network_alpha", "network_args"
                        ]:
                            config[key] = value
                            lrs_overrides.add(key)
                            print(f"  [Size] {key} = {value}", flush=True)
                else:
                    print(f"No size-based settings found for {num_images} images.", flush=True)
        else:
            # For non-SDXL models in this branch (like Flux)
            num_images = count_images_in_dataset(train_data_dir)
            print(f"Found {num_images} images in dataset (Flux training: Skipping SDXL-specific LRS logic).", flush=True)

        # 2. Load Hash Config (High Priority)
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)
            
            if lrs_settings:
                print(f"Applying Hash-Based Config for {model_hash} (Model: {model_name})...", flush=True)
                for optional_key in [
                    "max_grad_norm", "prior_loss_weight", "max_train_epochs",
                    "train_batch_size", "optimizer_args", "unet_lr",
                    "text_encoder_lr", "noise_offset", "min_snr_gamma",
                    "seed", "lr_warmup_steps", "loss_type", "huber_c", "huber_schedule",
                    "optimizer_type", "network_dim", "network_alpha", "network_args", "lr_scheduler", "lr_scheduler_args", "learning_rate",
                    "clip_skip", "caption_dropout_rate", "scale_weight_norms", "max_train_steps","gradient_accumulation_steps"
                ]:
                    if optional_key in lrs_settings:
                        config[optional_key] = lrs_settings[optional_key]
                        lrs_overrides.add(optional_key)
                        print(f"  [Hash] {optional_key} = {lrs_settings[optional_key]}", flush=True)
            else:
                print(f"No hash-based LRS settings found for {model_hash}", flush=True)
        else:
            print(f"Could not load LRS config for model_type={model_type}, is_style={is_style}", flush=True)
        # ---------------------------------------
        
        
        network_config_person = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 467,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 467,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 699,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 467,
            "dataautogpt3/TempestV0.1": 456,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 467,
            "fluently/Fluently-XL-Final": 228,
            "mann-e/Mann-E_Dreams": 456,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 228,
            "recoilme/colorfulxl": 228,
            "zenless-lab/sdxl-aam-xl-anime-mix": 456,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
            "zenless-lab/sdxl-anything-xl": 228,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
            "Corcelio/mobius": 228,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
        }

        network_config_style = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 235,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 235,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 235,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 235,
            "dataautogpt3/TempestV0.1": 228,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 235,
            "fluently/Fluently-XL-Final": 235,
            "mann-e/Mann-E_Dreams": 235,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 235,
            "recoilme/colorfulxl": 235,
            "zenless-lab/sdxl-aam-xl-anime-mix": 235,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
            "zenless-lab/sdxl-anything-xl": 235,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
            "Corcelio/mobius": 235,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
        }

        config_mapping = {
            228: {
                "network_dim": 128,
                "network_alpha": 128,
                "network_args": []
            },
            235: {
                "network_dim": 192,
                "network_alpha": 192,
                "network_args": []
            },
            456: {
                "network_dim": 192,
                "network_alpha": 192,
                "network_args": []
            },
            467: {
                "network_dim": 256,
                "network_alpha": 256,
                "network_args": []
            },
            699: {
                "network_dim": 320,
                "network_alpha": 320,
                "network_args": []
            },
        }
        
        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir
        # Count images in training directory
        # (Moved up for Size Config Logic)
        
        # Dynamic epochs logic moved up as well
        if model_type == "sdxl":
            if is_style:
                network_config = config_mapping[network_config_style.get(model_name, 235)]
            else:
                # Default to 467 (Dim 128) for Person/Likeness
                network_config = config_mapping[network_config_person.get(model_name, 467)]
            
            # Apply default network config
            # if "network_dim" not in lrs_overrides:
            config["network_dim"] = network_config["network_dim"]
            # if "network_alpha" not in lrs_overrides:
            config["network_alpha"] = network_config["network_alpha"]
            
            # Apply scaling
            config["network_dim"] = max(1, int(config["network_dim"] * dim_scale))
            config["network_alpha"] = max(1, int(config["network_alpha"] * dim_scale))
            
            scaled_args = []
            # Determine base args (Dynamic Logic based on dataset size)
            base_args = network_config["network_args"]
            
            # Determine base args (Dynamic Logic based on dataset size)
            base_args = network_config["network_args"]
            
            # Use dynamic base_args, unless overridden by LRS hash config
            if "network_args" not in lrs_overrides:
                if num_images <= 20:
                    print(f"Dataset size {num_images} (<=20): Applying Small Dataset Args (Dropout 0.1, Conv 64/64)", flush=True)
                    base_args = ["conv_dim=64", "conv_alpha=64", "dropout=0.1"]
                else:
                    print(f"Dataset size {num_images} (>20): Applying Medium/Large Dataset Args (Dropout 0.05, Conv 128/128)", flush=True)
                    base_args = ["conv_dim=128", "conv_alpha=128", "dropout=0.05"]
                source_args = base_args
            else:
                source_args = config.get("network_args", [])
            
            for arg in source_args:
                if "conv_dim=" in arg:
                    val = int(arg.split("=")[1])
                    val = max(1, int(val * dim_scale))
                    scaled_args.append(f"conv_dim={val}")
                elif "conv_alpha=" in arg:
                    val = int(arg.split("=")[1])
                    val = max(1, int(val * dim_scale))
                    scaled_args.append(f"conv_alpha={val}")
                else:
                    scaled_args.append(arg)
            config["network_args"] = scaled_args
            
            if dim_scale != 1.0:
                print(f"Scaled network dims by {dim_scale}", flush=True)
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"Created config at {config_path}", flush=True)
        print("\n" + "="*60, flush=True)
        print(f"Value configs is : {config}", flush=True)
        print("="*60, flush=True)
        return config_path, output_dir


def find_latest_checkpoint(output_dir: str, epoch: int) -> str:
    """Find the checkpoint file or directory for a specific epoch."""
    import glob
    
    # 1. Search for safetensors files (Recursive to find ai-toolkit nested files)
    # Try exact matches first
    patterns = [
        f"**/*epoch_{epoch}.safetensors",
        f"**/*epoch{epoch}.safetensors",
        f"*/-{epoch}.safetensors",
        f"*/_{epoch}.safetensors",
        f"*/-{epoch:06d}.safetensors",
        f"*/_{epoch:06d}.safetensors",
        f"*/_{epoch:09d}.safetensors", # ai-toolkit often uses 9 digits
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(output_dir, pattern), recursive=True)
        if matches:
            # If multiple matches, assume they are equivalent or pick one
            return matches[0]
        
    # Fallback: Find any safetensors file and sort by time
    checkpoints = glob.glob(os.path.join(output_dir, "*/.safetensors"), recursive=True)
    if checkpoints:
        # Return the most recently modified file
        return max(checkpoints, key=os.path.getmtime)

    # 2. Search for checkpoint directories (new logic for ai-toolkit)
    # ai-toolkit often outputs "step_X" or similar folders
    try:
        # List all subdirectories in output_dir
        subdirs = [
            os.path.join(output_dir, d) 
            for d in os.listdir(output_dir) 
            if os.path.isdir(os.path.join(output_dir, d)) and d != "evaluation"
        ]
        
        # Filter for directories that look like checkpoints
        # Heuristic: contains "checkpoint", "step", "epoch" or matches the epoch number
        checkpoint_dirs = []
        for d in subdirs:
            dirname = os.path.basename(d)
            if any(k in dirname.lower() for k in ["checkpoint", "step", "epoch"]) or str(epoch) in dirname:
                checkpoint_dirs.append(d)
        
        if checkpoint_dirs:
            # Sort by creation time or name? Name usually contains step number
             # Let's try to extract a number from the name to sort
            def extract_step(name):
                import re
                nums = re.findall(r'\d+', os.path.basename(name))
                if nums:
                    return int(nums[-1])
                return 0
            
            checkpoint_dirs.sort(key=extract_step)
            
            # If we are looking for a specific epoch (which acts as step/identifier here), try to match it
            # note: eval_identifier passed to this function might be epoch or step
            for d in checkpoint_dirs:
                if str(epoch) in os.path.basename(d):
                    return d
            
            return checkpoint_dirs[-1]
            
    except Exception as e:
        print(f"Warning: Error searching for checkpoint directories: {e}")
    
    return None


def run_training(
    model_type: str,
    config_path: str,
    output_dir: str,
    base_model_path: str,
    task_id: str = "training",
    deadline_timestamp: float = None
):
    """
    Run training without evaluation.
    
    Args:
        model_type: Model type (sdxl, flux, etc.)
        config_path: Path to training config
        output_dir: Output directory for checkpoints
        base_model_path: Path to base model
        task_id: Task ID for tracking
        deadline_timestamp: Timestamp when the training MUST stop (hours_to_complete)
    """
    print(f"Starting training for {model_type}", flush=True)
    print("LETS FUCKING GO", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = ["python3", "/app/ai-toolkit/run.py", config_path]
    else:
        if model_type in [ImageModelType.SDXL.value, ImageModelType.FLUX.value]:
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-script/{model_type}_train_network.py",
                "--config_file", config_path
            ]
        else:
             raise ValueError(f"Unsupported model type for training command: {model_type}")
    
    print("Starting training subprocess...\n", flush=True)
    process = subprocess.Popen(
        training_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "HF_HUB_OFFLINE": "1"}
    )
    
    current_epoch = 0
    total_epochs = 0
    oom_detected = False
    
    if deadline_timestamp:
        training_cutoff_time = deadline_timestamp
        print(f"Training will run until {time.ctime(training_cutoff_time)}", flush=True)
    
    # Checkpoint tracking
    detected_checkpoints = [] # List of (identifier, path)
    last_poll_time = time.time()
    last_polled_ckpt_path = None
    last_polled_ckpt_mtime = None
    stopped_by_timeout = False # Flag to track exit reason
    
    # LOOP: MONITOR TRAINING (90% TIME)
    try:
        while True:
            # Check for process exit
            ret = process.poll()
            if ret is not None:
                break
                
            # Read output (non-blockingish)
            line = process.stdout.readline()
            if not line and ret is not None:
                break
                
            if line:
                print(line, end="", flush=True)
                
                # OOM Detection
                if "out of memory" in line.lower() or "cuda error: out of memory" in line.lower():
                    oom_detected = True
                    # ... [OOM handling code same as before] ...
                    print("\n❌ Detected CUDA Out of Memory error!", flush=True)
                    process.terminate()
                    break

                # Checkpoint Detection (Logs)
                ai_save_match_2 = re.search(r'Saved checkpoint to\s+(.*)', line, re.IGNORECASE)
                if ai_save_match_2:
                    path = ai_save_match_2.group(1).strip()
                    print(f"\n[Tracker] Logs: Checkpoint saved at {path}", flush=True)
                    # Extract ID
                    step_match = re.search(r'step_(\d+)', path)
                    e_id = int(step_match.group(1)) if step_match else f"ckpt_{int(time.time())}"
                    
                    if path not in [p for _, p in detected_checkpoints]:
                        detected_checkpoints.append((e_id, path))
            
            # Checkpoint Detection (Polling Filesystem)
            current_time = time.time()
            if current_time - last_poll_time > 5.0:
                last_poll_time = current_time
                latest_ckpt = find_latest_checkpoint(output_dir, current_epoch)
                
                if latest_ckpt and os.path.exists(latest_ckpt):
                    try:
                        ckpt_mtime = os.path.getmtime(latest_ckpt)
                        if (latest_ckpt != last_polled_ckpt_path or ckpt_mtime != last_polled_ckpt_mtime):
                            # Debounce
                            if last_polled_ckpt_mtime is None or abs(ckpt_mtime - last_polled_ckpt_mtime) > 1.0:
                                print(f"\n[Tracker] Disk: Found new checkpoint {os.path.basename(latest_ckpt)}", flush=True)
                                
                                # Extract ID
                                step_match = re.search(r'(?:step|epoch)[-_]?(\d+)', os.path.basename(latest_ckpt), re.IGNORECASE)
                                if step_match:
                                    e_id = int(step_match.group(1))
                                else:
                                    e_id = f"t{int(ckpt_mtime)}"
                                
                                if latest_ckpt not in [p for _, p in detected_checkpoints]:
                                     detected_checkpoints.append((e_id, latest_ckpt))
                                
                                last_polled_ckpt_path = latest_ckpt
                                last_polled_ckpt_mtime = ckpt_mtime
                    except OSError:
                        pass
            
            # Time Limit Check
            if training_cutoff_time and time.time() >= training_cutoff_time:
                print(f"\n{'!'*60}", flush=True)
                print(f"⏳ Training time allocation finished.", flush=True)
                print(f"{'!'*60}\n", flush=True)
                stopped_by_timeout = True 
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutError:
                    print("Force killing process...", flush=True)
                    process.kill()
                break

    except KeyboardInterrupt:
        # ... [Interrupt handling] ...
        print("\nTraining interrupted by user.", flush=True)
        process.terminate()
        
    # Ensure process is dead and GPU is free
    if process.poll() is None:
        process.terminate()
        process.wait()
    
    if oom_detected:
        raise RuntimeError(OOM_ERROR)
    
    if ret != 0 and not stopped_by_timeout:
        raise RuntimeError(f"Training subprocess failed with exit code {ret}")
    
    return stopped_by_timeout


def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed


async def main():
    print("---STARTING IMAGE TRAINING---", flush=True)
    parser = argparse.ArgumentParser(description="Image Model Training")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"])
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True)
    args = parser.parse_args()
    
    # Enforce offline mode for Hugging Face Hub if running in a restricted network
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)
    
    print(f"Ensuring base model {args.model} is available...", flush=True)
    model_path = await download_base_model(args.model, train_cst.CACHE_MODELS_DIR, args.model_type)
    model_path = get_model_path(model_path)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Base model path {model_path} does not exist. The downloader likely failed or skipped the model.")
        sys.exit(1)
        
    print(f"Using base model at: {model_path}", flush=True)
    
    start_time = time.time()
    deadline_timestamp = start_time + (args.hours_to_complete * 3600)
    print(f"Training started at {time.ctime(start_time)}", flush=True)
    print(f"Training deadline set for {time.ctime(deadline_timestamp)} ({args.hours_to_complete} hours)", flush=True)
    
    print("Preparing dataset...", flush=True)
    await download_image_dataset(args.dataset_zip, args.task_id, train_cst.CACHE_DATASETS_DIR)
    
    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )
    
    repeats = cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS
    folder_name = f"{repeats}_{cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT} {cst.DIFFUSION_DEFAULT_CLASS_PROMPT}"
    concept_folder = os.path.join(train_paths.get_image_training_images_dir(args.task_id), folder_name)
    
    
    dim_scale = 1.0
    max_retries = 3
    
    for attempt in range(max_retries):
        if time.time() >= deadline_timestamp:
            print("\n⏳ Training deadline already reached. Skipping further attempts.", flush=True)
            break

        try:
            print(f"\n--- Training Attempt {attempt+1}/{max_retries} (Dim Scale: {dim_scale}) ---", flush=True)
            
            config_path, output_dir = create_config(
                args.task_id,
                model_path,
                args.model,
                args.model_type,
                args.expected_repo_name,
                args.hours_to_complete,
                args.trigger_word,
                dim_scale=dim_scale
            )
            
            stopped_by_timeout = run_training(
                model_type=args.model_type,
                config_path=config_path,
                output_dir=output_dir,
                base_model_path=model_path,
                task_id=args.task_id,
                deadline_timestamp=deadline_timestamp
            )
            
            if stopped_by_timeout:
                print("\n⏳ Training stopped because time allocation finished. Exiting.", flush=True)
                break

            print("\n✅ Training complete!", flush=True)
            break
            
        except RuntimeError as e:
            msg = str(e).lower()
            if any(x in msg for x in (OOM_ERROR, ERRRR, CUDA_ERROR)):
                if attempt < max_retries - 1:
                    dim_scale *= 0.5
                    print(f"📉 Reducing network dimensions by 50% for next attempt (New Scale: {dim_scale})", flush=True)
                    if os.path.exists(output_dir):
                        print(f"Cleaning up previous output directory: {output_dir}", flush=True)
                        try:
                            shutil.rmtree(output_dir)
                            os.makedirs(output_dir, exist_ok=True)
                        except Exception as clean_err:
                            print(f"Warning: Failed to clean output dir: {clean_err}")
                else:
                    print("❌ Max retries reached with OOM. Training failed.", flush=True)
                    sys.exit(1)
            else:
                print(f"\n❌ Training failed with error: {e}", flush=True)
                sys.exit(1)

                
        except Exception as e:
            print(f"\n❌ Unrecoverable error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if _name_ == "_main_":
    asyncio.run(main())