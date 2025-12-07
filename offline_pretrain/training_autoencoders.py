import json
import sys
from collections import deque

import numpy as np
import signal
import atexit
import subprocess
import threading
import random
import time
import pickle
import shutil
import pandas as pd
import torch
import psutil
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import RobustScaler
from config.config import autoencoder_params, training_mode_config
from src.duplimend_framework.streaming_sparse_denoising_autoencoder import SparseDenoisingAutoencoder
from src.duplimend_framework.utils.online_activity_embeddings import DynamicActivityEmbeddings
from src.duplimend_framework.utils.online_onehot_encoder import OnlineOneHotEncoder
from src.duplimend_framework.utils.json_feature_dataset import JSONLFeatureDataset
from torch.amp import GradScaler, autocast
from src.duplimend_framework.utils.online_activity_embeddings import DynamicActivityEmbeddings
from src.duplimend_framework.utils.online_onehot_encoder import OnlineOneHotEncoder
from torch.utils.tensorboard import SummaryWriter
from src.duplimend_framework.utils.config_saver import save_config_to_output

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
_tensorboard_process = None

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.cuda.empty_cache()

def get_random_sample_vectors(feature_file_path, sample_size=10000):
    """Randomly sample feature vectors from a JSONL file for scaler fitting."""
    offsets = []
    with open(feature_file_path, 'r') as f:
        offset = 0
        for line in f:
            offsets.append(offset)
            offset += len(line.encode('utf-8'))
    total_lines = len(offsets)
    if total_lines == 0:
        raise ValueError("No lines found in file.")

    sample_indices = sorted(random.sample(range(total_lines), min(sample_size, total_lines)))

    sample_vectors = []
    with open(feature_file_path, 'r') as f:
        for idx in sample_indices:
            f.seek(offsets[idx])
            line = f.readline()
            feature_data = json.loads(line)
            feature_vector = feature_data.get("feature_vector", [])
            if feature_vector:
                sample_vectors.append(feature_vector)
    return np.array(sample_vectors, dtype=np.float32)

class ChunkedJSONLFeatureDataset(Dataset):
    """Streams feature vectors from a JSONL file in chunks for memory efficiency."""
    def __init__(self, feature_file_path, scaler, chunk_size=100):
        self.feature_file_path = feature_file_path
        self.scaler = scaler
        self.chunk_size = chunk_size
        self.offsets = []
        with open(feature_file_path, 'r') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line.encode('utf-8'))
        self.current_chunk = None
        self.current_chunk_start = -1

    def __len__(self):
        return len(self.offsets)

    def _load_chunk(self, chunk_start):
        chunk = []
        with open(self.feature_file_path, 'r') as f:
            f.seek(self.offsets[chunk_start])
            for _ in range(self.chunk_size):
                line = f.readline()
                if not line:
                    break
                feature_data = json.loads(line)
                feature_vector = feature_data.get("feature_vector", [])
                chunk.append(feature_vector)
        return chunk

    def __getitem__(self, idx):
        chunk_start = (idx // self.chunk_size) * self.chunk_size
        if self.current_chunk is None or self.current_chunk_start != chunk_start:
            self.current_chunk = self._load_chunk(chunk_start)
            self.current_chunk_start = chunk_start
        local_idx = idx - chunk_start
        feature_vector = self.current_chunk[local_idx]
        scaled_vec = self.scaler.transform([feature_vector])[0]
        return torch.tensor(scaled_vec, dtype=torch.float32)

def get_chunked_loader(feature_file_path, scaler, batch_size, num_workers=0, chunk_size=100):
    dataset = ChunkedJSONLFeatureDataset(feature_file_path, scaler, chunk_size=chunk_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    return loader 


class PreloadedScaledDataset(Dataset):
    """Pre-scaled dataset that loads and scales all data once."""
    def __init__(self, feature_file_path, scaler=None):
        self.data, self.scaler = self._load_and_scale_all_data(feature_file_path, scaler)

    def _load_and_scale_all_data(self, feature_file_path, scaler):
        feature_vectors = []
        with open(feature_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    feature_data = json.loads(line)
                    feature_vector = feature_data.get("feature_vector", [])
                    if feature_vector:
                        feature_vectors.append(feature_vector)

        if not feature_vectors:
            raise ValueError(f"No feature vectors found in {feature_file_path}")

        feature_matrix = np.array(feature_vectors, dtype=np.float32)

        if scaler is None:
            scaler = RobustScaler()
            scaled_matrix = scaler.fit_transform(feature_matrix)
        else:
            scaled_matrix = scaler.transform(feature_matrix)

        return scaled_matrix, scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

def get_optimized_loader(feature_file_path, batch_size, scaler=None, num_workers=0, use_full_batch=True):
    """Optimized DataLoader that loads and scales all data once."""
    dataset = PreloadedScaledDataset(feature_file_path, scaler)

    if use_full_batch:
        optimal_batch_size = len(dataset)
    else:
        optimal_batch_size = min(batch_size * 8, len(dataset) // 2, 1024)

    loader = DataLoader(
        dataset,
        batch_size=optimal_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return loader, dataset.scaler

try:
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
def get_memory_safe_loader(feature_file_path, batch_size, scaler=None, max_gb=1.0):
    """Memory-safe data loader that adapts to file size."""
    file_size_gb = os.path.getsize(feature_file_path) / (1024**3)
    force_chunked_threshold = autoencoder_params.get("force_chunked_mode_threshold_gb", 10.0)

    if file_size_gb > force_chunked_threshold:
        sample_size = min(100000, int(file_size_gb * 10000))
        sample_vectors = get_random_sample_vectors(feature_file_path, sample_size=sample_size)

        if scaler is None:
            scaler = RobustScaler().fit(sample_vectors)

        chunk_size = autoencoder_params.get("chunk_size", 1000)
        if file_size_gb > 50:
            chunk_size = min(chunk_size, 10000)

        dataset = ChunkedJSONLFeatureDataset(feature_file_path, scaler, chunk_size=chunk_size)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )

        return loader, scaler
    else:
        return get_optimized_loader(feature_file_path, batch_size, scaler, use_full_batch=False)
    
class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset with adaptive loading and LRU cache."""
    def __init__(self, feature_file_path, scaler, max_memory_mb=1000):
        self.feature_file_path = feature_file_path
        self.scaler = scaler
        self.max_memory_mb = max_memory_mb

        self.offsets = []
        with open(feature_file_path, 'r') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line.encode('utf-8'))

        avg_line_size = offset / len(self.offsets) if self.offsets else 1000
        self.cache_size = min(
            len(self.offsets),
            int((max_memory_mb * 1024 * 1024) / avg_line_size)
        )

        self.cache = {}
        self.cache_order = deque(maxlen=self.cache_size)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        with open(self.feature_file_path, 'r') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            feature_data = json.loads(line)
            feature_vector = feature_data.get("feature_vector", [])

        scaled_vec = self.scaler.transform([feature_vector])[0]
        tensor = torch.tensor(scaled_vec, dtype=torch.float32)

        if len(self.cache) >= self.cache_size:
            oldest_idx = self.cache_order.popleft()
            del self.cache[oldest_idx]

        self.cache[idx] = tensor
        self.cache_order.append(idx)

        return tensor
    
def train_autoencoder_worker(args):
    """Train a single activity-specific autoencoder."""
    activity_label, feature_file_path, device_str, save_dir, ae_params = args

    try:
        torch.cuda.empty_cache()
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            file_size_gb = os.path.getsize(feature_file_path) / (1024**3)
            if file_size_gb > 30:
                torch.cuda.set_per_process_memory_fraction(0.7)
            else:
                torch.cuda.set_per_process_memory_fraction(0.85)

        max_batch_size_gb = ae_params.get("max_batch_size_gb", 1.0)
        loader, scaler = get_memory_safe_loader(
            feature_file_path,
            ae_params["batch_size"],
            max_gb=max_batch_size_gb
        )

        sample_batch = next(iter(loader))
        input_dim = sample_batch.shape[1]
        del sample_batch
        torch.cuda.empty_cache()

        if TENSORBOARD_AVAILABLE and ae_params.get("enable_tensorboard", True):
            log_dir = os.path.join(save_dir, "tensorboard_logs", f"activity_{activity_label}")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = None

        device = torch.device(device_str)
        model = SparseDenoisingAutoencoder(
            input_dim=input_dim,
            latent_dim=ae_params["latent_dim"],
            hidden_dims=ae_params["hidden_dims"],
            sparsity_lambda=ae_params["sparsity_lambda"],
            noise_std=ae_params["noise_std"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=ae_params["learning_rate"],
            betas=ae_params.get("betas", (0.9, 0.999)),
            eps=ae_params.get("eps", 1e-8),
            weight_decay=ae_params.get("weight_decay", 1e-5),
            amsgrad=ae_params.get("amsgrad", False)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=ae_params.get("lr_decay_factor", 0.7),
            patience=ae_params.get("lr_patience", 5),
            min_lr=ae_params.get("min_lr", 1e-7),
        )

        gradient_accumulation_steps = ae_params.get("gradient_accumulation_steps", 1)

        model.train()
        best_loss = float("inf")
        best_state = None
        patience = 0

        use_amp = torch.cuda.is_available()
        if use_amp:
            scaler_amp = GradScaler()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        total_batches = len(loader)
        progress_interval = 1 if total_batches == 1 else max(1, total_batches // 10)

        for epoch in range(ae_params["max_epochs"]):
            epoch_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_sparsity_loss = 0.0
            accumulation_loss = 0.0
            batch_count = 0
            epoch_start_time = time.time()

            torch.cuda.empty_cache()

            for batch_idx, batch in enumerate(loader):
                batch = batch.to(device, non_blocking=True)

                if torch.cuda.is_available() and batch_idx % 100 == 0:
                    memory_used = torch.cuda.memory_allocated() / (1024**3)
                    if memory_used > gpu_memory_gb * 0.9:
                        torch.cuda.empty_cache()

                if use_amp:
                    with autocast(device_type=device.type):
                        x_hat, z = model(batch)
                        rec_loss = torch.nn.functional.mse_loss(x_hat, batch)
                        sparsity = ae_params["sparsity_lambda"] * torch.mean(torch.abs(z))
                        loss = (rec_loss + sparsity) / gradient_accumulation_steps
                    
                    scaler_amp.scale(loss).backward()
                    accumulation_loss += loss.item()
                    
                    # OPTIMIZER STEP (every N batches)
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), ae_params["grad_clip"])
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                        optimizer.zero_grad(set_to_none=True)
                        
                        epoch_loss += accumulation_loss
                        epoch_rec_loss += (accumulation_loss * gradient_accumulation_steps) - (ae_params["sparsity_lambda"] * torch.mean(torch.abs(z)).item())
                        epoch_sparsity_loss += ae_params["sparsity_lambda"] * torch.mean(torch.abs(z)).item()
                        batch_count += 1
                        accumulation_loss = 0.0
                else:
                    x_hat, z = model(batch)
                    rec_loss = torch.nn.functional.mse_loss(x_hat, batch)
                    sparsity = ae_params["sparsity_lambda"] * torch.mean(torch.abs(z))
                    loss = (rec_loss + sparsity) / gradient_accumulation_steps
                    
                    loss.backward()
                    accumulation_loss += loss.item()
                    
                    # OPTIMIZER STEP (every N batches)
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), ae_params["grad_clip"])
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        
                        epoch_loss += accumulation_loss
                        epoch_rec_loss += rec_loss.item()
                        epoch_sparsity_loss += sparsity.item()
                        batch_count += 1
                        accumulation_loss = 0.0
                
                if writer and (batch_idx + 1) % gradient_accumulation_steps == 0:
                    global_step = epoch * (len(loader) // gradient_accumulation_steps) + (batch_idx // gradient_accumulation_steps)
                    writer.add_scalar(f'{activity_label}/Batch/Total_Loss', accumulation_loss * gradient_accumulation_steps, global_step)
                    writer.add_scalar(f'{activity_label}/Batch/Reconstruction_Loss', rec_loss.item(), global_step)
                    writer.add_scalar(f'{activity_label}/Batch/Sparsity_Loss', sparsity.item(), global_step)
                    writer.add_scalar(f'{activity_label}/Batch/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

                    with torch.no_grad():
                        z_mean = torch.mean(z).item()
                        z_std = torch.std(z).item()
                        z_sparsity = (torch.abs(z) < 0.01).float().mean().item()
                        writer.add_scalar(f'{activity_label}/Latent/Mean_Activation', z_mean, global_step)
                        writer.add_scalar(f'{activity_label}/Latent/Std_Activation', z_std, global_step)
                        writer.add_scalar(f'{activity_label}/Latent/Sparsity_Ratio', z_sparsity, global_step)

                del batch, x_hat, z, rec_loss, sparsity, loss

                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

            if batch_count > 0:
                epoch_loss /= batch_count
                epoch_rec_loss /= batch_count
                epoch_sparsity_loss /= batch_count
            epoch_time = time.time() - epoch_start_time

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if writer:
                writer.add_scalar(f'{activity_label}/Epoch/Total_Loss', epoch_loss, epoch)
                writer.add_scalar(f'{activity_label}/Epoch/Reconstruction_Loss', epoch_rec_loss, epoch)
                writer.add_scalar(f'{activity_label}/Epoch/Sparsity_Loss', epoch_sparsity_loss, epoch)
                writer.add_scalar(f'{activity_label}/Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar(f'{activity_label}/Epoch/Training_Time', epoch_time, epoch)
                writer.add_scalar(f'{activity_label}/Epoch/Patience_Counter', patience, epoch)

                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    writer.add_scalar(f'{activity_label}/Memory/GPU_Allocated_GB', memory_allocated, epoch)
                    writer.add_scalar(f'{activity_label}/Memory/GPU_Reserved_GB', memory_reserved, epoch)

                for name, param in model.named_parameters():
                    try:
                        if param.data is not None and param.data.numel() > 0:
                            if torch.isfinite(param.data).all():
                                writer.add_histogram(f'{activity_label}/Weights/{name}', param.data, epoch)
                        if param.grad is not None and param.grad.data.numel() > 0:
                            if torch.isfinite(param.grad.data).all():
                                writer.add_histogram(f'{activity_label}/Gradients/{name}', param.grad.data, epoch)
                    except Exception:
                        pass

            scheduler.step(epoch_loss)

            if epoch_loss < best_loss - ae_params["min_delta"]:
                best_loss = epoch_loss
                patience = 0
                best_state = model.state_dict().copy()
            else:
                patience += 1

            if patience >= ae_params["early_stopping_patience"]:
                break

        torch.cuda.empty_cache()

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        model_cpu = model.to('cpu')

        model_path = os.path.join(save_dir, f"autoencoder_{activity_label}.pth")
        torch.save(model_cpu.state_dict(), model_path)

        scaler_path = os.path.join(save_dir, f"scaler_{activity_label}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        if writer:
            writer.close()

        return (activity_label, model, scaler, "success")

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        if 'writer' in locals() and writer:
            writer.close()
        return (activity_label, None, None, f"CUDA OOM: {str(e)}")
    except Exception as e:
        if 'writer' in locals() and writer:
            writer.close()
        return (activity_label, None, None, str(e))
    


def launch_tensorboard_for_training(log_dir, port=6007):
    """Launch TensorBoard for training visualization."""
    global _tensorboard_process
    import shutil
    import sys
    import time

    try:
        time.sleep(2)

        tensorboard_path = shutil.which("tensorboard")
        if tensorboard_path is None:
            _tensorboard_process = subprocess.Popen([
                sys.executable, "-m", "tensorboard.main",
                "--logdir", log_dir,
                "--port", str(port),
                "--reload_interval", "3",
                "--bind_all"
            ])
        else:
            _tensorboard_process = subprocess.Popen([
                "tensorboard",
                "--logdir", log_dir,
                "--port", str(port),
                "--reload_interval", "3",
                "--bind_all"
            ])

        _tensorboard_process.wait()

    except Exception:
        pass

def train_autoencoders_offline(feature_vectors_dir, models_save_dir):
    """Train autoencoders for all activities from pre-generated feature vectors."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_path = os.path.join(feature_vectors_dir, "generation_metadata.json")
    if not os.path.exists(metadata_path):
        return

    with open(metadata_path, 'r') as f:
        generation_metadata = json.load(f)

    all_activities = generation_metadata["activities"]

    exclude_activities = os.environ.get('EXCLUDE_ACTIVITIES', '').split(',')
    exclude_activities = [act.strip() for act in exclude_activities if act.strip()]

    if exclude_activities:
        all_activities = [act for act in all_activities if act not in exclude_activities]

    if os.path.exists(os.path.join(feature_vectors_dir, "activity_embeddings.pkl")):
        with open(os.path.join(feature_vectors_dir, "activity_embeddings.pkl"), 'rb') as f:
            activity_embeddings = pickle.load(f)
    else:
        activity_embeddings = DynamicActivityEmbeddings(embedding_dim=generation_metadata.get("embedding_dim", 8), seed=42)

    if os.path.exists(os.path.join(feature_vectors_dir, "onehot_encoders.pkl")):
        with open(os.path.join(feature_vectors_dir, "onehot_encoders.pkl"), 'rb') as f:
            onehot_encoders = pickle.load(f)
    else:
        onehot_encoders = {}

    os.makedirs(models_save_dir, exist_ok=True)

    args_list = []
    for activity in sorted(all_activities):
        feature_path = os.path.join(feature_vectors_dir, f"features_{activity}.jsonl")
        if os.path.exists(feature_path):
            args_list.append((activity, feature_path, device_str, models_save_dir, autoencoder_params))

    results = []
    for args in args_list:
        result = train_autoencoder_worker(args)
        results.append(result)

    autoencoders = {}
    scalers = {}
    failed = []

    for activity_label, model, scaler, status in results:
        if status == "success":
            autoencoders[activity_label] = model
            scalers[activity_label] = scaler
        else:
            failed.append(activity_label)

    metadata = {
        "trained_activities": list(autoencoders.keys()),
        "autoencoder_config": autoencoder_params,
        "training_config": {
            "max_epochs": autoencoder_params["max_epochs"],
            "batch_size": autoencoder_params["batch_size"],
            "early_stopping_patience": autoencoder_params["early_stopping_patience"],
            "min_delta": autoencoder_params["min_delta"],
            "grad_clip": autoencoder_params["grad_clip"],
            "lr_scheduler_config": {
                "mode": "min",
                "factor": autoencoder_params["lr_decay_factor"],
                "patience": autoencoder_params["lr_patience"],
                "min_lr": autoencoder_params["min_lr"]
            }
        },
        "training_timestamp": pd.Timestamp.now().isoformat(),
        "training_method": "parallel_prescaled"
    }
    with open(os.path.join(models_save_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    for fname in ["activity_embeddings.pkl", "onehot_encoders.pkl", "global_state.pkl"]:
        src = os.path.join(feature_vectors_dir, fname)
        dst = os.path.join(models_save_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    return autoencoders, scalers

def cleanup_tensorboard():
    """Clean up TensorBoard process on exit."""
    global _tensorboard_process
    if _tensorboard_process and _tensorboard_process.poll() is None:
        try:
            _tensorboard_process.terminate()
            _tensorboard_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _tensorboard_process.kill()
        except Exception:
            pass

def setup_cleanup_handlers():
    """Register cleanup handlers for various exit scenarios."""
    atexit.register(cleanup_tensorboard)

    def signal_handler(signum, frame):
        cleanup_tensorboard()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main function for offline autoencoder training."""
    import torch
    setup_cleanup_handlers()

    feature_vectors_dir_override = os.environ.get('FEATURE_VECTORS_DIR')
    models_save_dir_override = os.environ.get('experiment_output_dir')

    if feature_vectors_dir_override and models_save_dir_override:
        feature_vectors_dir = feature_vectors_dir_override
        models_save_dir = models_save_dir_override
    else:
        training_folder = training_mode_config.get("training_folder")
        feature_vectors_base_dir = training_mode_config.get("feature_vectors_base_dir")
        models_save_dir = training_mode_config.get("models_save_dir", feature_vectors_base_dir)

        if not training_folder:
            return

        folder_name = os.path.basename(training_folder.rstrip('/\\'))
        feature_vectors_dir = os.path.join(feature_vectors_base_dir, f"feature_vectors_{folder_name}")
        models_save_dir = os.path.join(feature_vectors_base_dir, f"trained_models_{folder_name}")

    additional_info = {
        "feature_vectors_dir": feature_vectors_dir,
        "models_save_dir": models_save_dir,
        "pytorch_version": torch.__version__
    }
    save_config_to_output(models_save_dir, "autoencoder_training", additional_info)

    if not os.path.exists(feature_vectors_dir):
        cleanup_tensorboard()
        sys.exit(1)

    metadata_path = os.path.join(feature_vectors_dir, "generation_metadata.json")
    if not os.path.exists(metadata_path):
        cleanup_tensorboard()
        sys.exit(1)

    try:
        if TENSORBOARD_AVAILABLE and autoencoder_params.get("enable_tensorboard", True):
            tensorboard_log_dir = os.path.join(models_save_dir, "tensorboard_logs")
            os.makedirs(tensorboard_log_dir, exist_ok=True)

            tensorboard_thread = threading.Thread(
                target=launch_tensorboard_for_training,
                args=(tensorboard_log_dir, 6007),
                daemon=True
            )
            tensorboard_thread.start()
            time.sleep(3)

        autoencoders, scalers = train_autoencoders_offline(feature_vectors_dir, models_save_dir)

    except KeyboardInterrupt:
        cleanup_tensorboard()
    except Exception as e:
        cleanup_tensorboard()
        raise
    finally:
        cleanup_tensorboard()

if __name__ == "__main__":
    main()