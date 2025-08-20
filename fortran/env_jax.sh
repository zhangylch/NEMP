module purge
module load miniconda3/latest   
source /opt/local/miniconda3/etc/profile.d/conda.sh
module load intel-oneapi-compilers/2025.1.1-b3qi   
module load intel-oneapi-mkl/2024.2.2-tczp
module list
conda activate jax
which python
