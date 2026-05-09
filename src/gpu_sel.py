import os
import subprocess


def gpu_sel(local_size):
    """Select visible GPUs before JAX is imported.

    Slurm and many cluster launchers set CUDA_VISIBLE_DEVICES themselves. In that
    case we keep their assignment, because overriding it can make JAX look at
    GPUs outside the job allocation.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        print(f"CUDA_VISIBLE_DEVICES preset: {visible_devices}", flush=True)
        return

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        memory_gpu = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip()
        ]
    except Exception as exc:
        print(f"GPU auto-selection skipped: {exc}", flush=True)
        return

    if not memory_gpu:
        print("GPU auto-selection skipped: nvidia-smi returned no GPUs", flush=True)
        return

    gpu_queue = sorted(range(len(memory_gpu)), key=lambda k: memory_gpu[k])
    selected = ",".join(str(i) for i in gpu_queue[:local_size])
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    print(f"CUDA_VISIBLE_DEVICES auto-selected: {selected}", flush=True)
