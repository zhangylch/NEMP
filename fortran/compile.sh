#!/bin/bash

# ================= 配置区域 =================
# 你的代码目录
CODE_DIR="/work/home/bjiangch/zyl/program/NEMP/fortran/"
# ===========================================

# 1. 激活 Conda 环境
source ~/zyl/miniconda/install/bin/activate jax

# 2. 确保进入代码目录
if [ ! -d "$CODE_DIR" ]; then
  echo "错误：找不到目录 $CODE_DIR"
  exit 1
fi
cd "$CODE_DIR" || exit

# 3. 设置语言环境 (修复 Unicode 报错)
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# 4. 显式指定编译器 (修复 Meson 找不到编译器的问题)
# 这一步非常关键！告诉 Meson 使用系统(或conda)里的 gfortran 和 gcc
export FC=gfortran
export CC=gcc

# 5. 设置 OpenBLAS 路径
export OPENBLAS_LIB="$CONDA_PREFIX/lib"
# 将 Conda 的库路径加入 PKG_CONFIG_PATH，帮助 Meson 找到库
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# 6. 安装必要的构建工具 (确保 pkg-config 存在)
echo "正在检查构建工具..."
pip install meson ninja > /dev/null 2>&1
# 尝试安装 pkg-config，如果 conda 里没有可能会跳过，但这步能增加成功率
conda install -y pkg-config > /dev/null 2>&1 || true

# 7. 【核心修复】强力清理旧文件
# Meson 的构建目录通常叫 bbdir，必须删掉它才能重新 setup
echo "正在清理环境..."
rm -rf bbdir build tmp _tmp* *.so *.o *.mod getneigh.pyf

# 8. 生成签名文件
echo "正在生成签名文件..."
python -m numpy.f2py -m getneigh -h getneigh.pyf \
  init_dealloc.f90 get_neigh.f90 inverse_matrix.f90 \
  --overwrite-signature

if [ ! -f "getneigh.pyf" ]; then
    echo "❌ 错误：签名文件 getneigh.pyf 生成失败！"
    exit 1
fi

# 9. 编译
echo "正在编译 (Meson Backend)..."
# 注意：我们去掉了 --f90flags，因为 Meson 处理 flags 的方式不同
# 我们通过环境变量 FFLAGS 来传递优化参数
export FFLAGS="-fopenmp -O3"
export CFLAGS="-O3"
export LDFLAGS="-L$OPENBLAS_LIB -lgomp -lopenblas"

# 运行 f2py
# 注意：这里不再传 -L 和 -l 给 f2py 命令行，而是依靠上面的 LDFLAGS 环境变量
# Meson 后端有时对命令行参数很挑剔
python -m numpy.f2py \
  -m getneigh -c \
  getneigh.pyf init_dealloc.f90 get_neigh.f90 inverse_matrix.f90 \
  --backend=meson

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
else
    echo "❌ 编译失败！请向上滚动查看具体的 Meson 错误日志。"
    exit 1
fi
