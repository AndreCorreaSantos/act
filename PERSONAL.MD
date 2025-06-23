this needed to run replay:
```
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LIBGL_ALWAYS_INDIRECT=0
export MUJOCO_GL="egl"  
```

then 

```
python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data_dir/ --num_episodes 50 --onscreen_render
```
