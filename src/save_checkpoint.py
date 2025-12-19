import os
import pickle
import jax
from dataclasses import asdict, is_dataclass

def save_checkpoint(step, params, ema_params, opt_state, config, ckpt_dir, max_to_keep=5):
    """
    保存检查点，并自动清理旧的检查点，只保留最近 max_to_keep 个。
    """
    # 1. 准备数据 (所有 Rank 执行)
    state_dict = {
        "step": step,
        "params": jax.device_get(params),
        "ema_params": jax.device_get(ema_params),
        "opt_state": jax.device_get(opt_state),
        "config": jax.device_get(asdict(config))
    }

    # 2. 只有 Rank 0 写文件和清理旧文件
    if jax.process_index() == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # --- A. 保存新文件 ---
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
            return # 保存失败则不进行清理

        # --- B. 清理旧文件 (Keep Max N) ---
        try:
            # 1. 找出目录下所有的 checkpoint_xxx.pkl
            all_files = [
                f for f in os.listdir(ckpt_dir) 
                if f.startswith("checkpoint_") and f.endswith(".pkl")
            ]
            
            # 2. 按 step 从小到大排序
            # 文件名格式 checkpoint_100.pkl -> 提取 100
            all_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            # 3. 检查是否超过限制
            if len(all_files) > max_to_keep:
                # 需要删除的文件 (列表头部是旧文件)
                files_to_remove = all_files[:-max_to_keep]
                
                for f_name in files_to_remove:
                    f_path = os.path.join(ckpt_dir, f_name)
                    os.remove(f_path)
                    print(f"[Rank 0] Pruned old checkpoint: {f_name}")
                    
        except Exception as e:
            print(f"⚠️ [Rank 0] Failed to prune old checkpoints: {e}")

def restore_checkpoint(ckpt_dir, devices):
    """
    加载检查点：
    1. 寻找最新的 .pkl 文件
    2. 所有 Rank 同时读取文件
    3. 使用 jax.device_put 将数据推送到各自的 GPU 上
    """
    # 1. 寻找最新的 checkpoint 文件 (简单的按文件名排序)
    if not os.path.exists(ckpt_dir):
        if jax.process_index() == 0:
            print(f"⚠️ Checkpoint directory {ckpt_dir} does not exist. Starting from scratch.")
        return None

    # 列出所有 .pkl 文件
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
    if not files:
        if jax.process_index() == 0:
            print(f"No checkpoint files found in {ckpt_dir}. Starting from scratch.")
        return None

    # 提取 step 数字并排序找到最新的
    # 文件名格式: checkpoint_1000.pkl
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_file = files[-1]
    load_path = os.path.join(ckpt_dir, latest_file)

    if jax.process_index() == 0:
        print(f"Loading checkpoint from: {load_path}")

    # 2. 读取文件 (所有 Rank 都要读)
    # 并行文件系统通常能很好地处理多个进程读同一个文件
    try:
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
    except Exception as e:
        print(f"❌ [Rank {jax.process_index()}] Failed to load checkpoint: {e}")
        return None

    
    restored_step = state_dict["step"]
    restored_config = state_dict["config"] # Config 留在 CPU 上即可

    if jax.process_index() == 0:
        print(f"✅ Successfully restored state at step {restored_step}")

    return restored_step, state_dict["params"], state_dict["ema_params"], state_dict["opt_state"], restored_config
