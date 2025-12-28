import os
import pickle
import jax
from dataclasses import asdict, is_dataclass

def save_checkpoint(step, params, ema_params, opt_state, config, ckpt_dir, max_to_keep=5):
    state_dict = {
        "step": step,
        "params": jax.device_get(params),
        "ema_params": jax.device_get(ema_params),
        "opt_state": jax.device_get(opt_state),
        "config": jax.device_get(asdict(config))
    }

    if jax.process_index() == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        
        save_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pkl")
        temp_path = save_path + ".tmp"
        
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(state_dict, f)
            os.rename(temp_path, save_path)
            print(f"✅ [Rank 0] Checkpoint saved: {save_path}")
        except Exception as e:
            print(f"❌ [Rank 0] Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return

        try:
            all_files = [
                f for f in os.listdir(ckpt_dir) 
                if f.startswith("checkpoint_") and f.endswith(".pkl")
            ]
            
            all_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            if len(all_files) > max_to_keep:
                files_to_remove = all_files[:-max_to_keep]
                
                for f_name in files_to_remove:
                    f_path = os.path.join(ckpt_dir, f_name)
                    os.remove(f_path)
                    print(f"[Rank 0] Pruned old checkpoint: {f_name}")
                    
        except Exception as e:
            print(f"⚠️ [Rank 0] Failed to prune old checkpoints: {e}")

def restore_checkpoint(ckpt_dir, devices):
    if not os.path.exists(ckpt_dir):
        if jax.process_index() == 0:
            print(f"⚠️ Checkpoint directory {ckpt_dir} does not exist. Starting from scratch.")
        return None

    files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
    if not files:
        if jax.process_index() == 0:
            print(f"No checkpoint files found in {ckpt_dir}. Starting from scratch.")
        return None

    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_file = files[-1]
    load_path = os.path.join(ckpt_dir, latest_file)

    if jax.process_index() == 0:
        print(f"Loading checkpoint from: {load_path}")

    try:
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
    except Exception as e:
        print(f"❌ [Rank {jax.process_index()}] Failed to load checkpoint: {e}")
        return None

    
    restored_step = state_dict["step"]
    restored_config = state_dict["config"] 

    if jax.process_index() == 0:
        print(f"✅ Successfully restored state at step {restored_step}")

    return restored_step, state_dict["params"], state_dict["ema_params"], state_dict["opt_state"], restored_config
