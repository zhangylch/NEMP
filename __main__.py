import sys 
if sys.argv[1]=="mae":
    import mae.eval
elif sys.argv[1]=="rmse":
    import rmse.eval
elif sys.argv[1]=="train":
    import train.train
elif sys.argv[1]=="jax_md" or sys.argv[1]=="jax_md_nvt":
    import JAX_MD.nvt
elif sys.argv[1]=="jax_md_nve":
    import JAX_MD.nve
