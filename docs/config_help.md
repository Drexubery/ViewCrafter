## Important configuration options for [inference.py](../inference.py):

### 1. General configs
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--image_dir` | './test/images/fruit.png' | Image file path |
| `--out_dir` | './output' | Output directory |
| `--device` | 'cuda:0' | The device to use |
| `--exp_name` | None | Experiment name, use image file name by default |
### 2. Point cloud render configs
#### The definition of world coordinate system and tips for adjusting point cloud render configs are illustrated in [render document](./render_help.md).
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--mode` | 'single_view_txt' | Currently we support 'single_view_txt' and 'single_view_target' |
| `--traj_txt` | None | Required for 'single_view_txt' mode, a txt file that specify camera trajectory |
| `--elevation` | 5. | The elevation angle of the input image in degree. Estimate a rough value based on your visual judgment |
| `--center_scale` | 1. | Scale factor for the spherical radius (r). By default, r is set to the depth value of the center pixel (H//2, W//2) of the reference image |
| `--d_theta` | 10. | Required for 'single_view_target' mode, specify target theta angle as (theta + d_theta) |
| `--d_phi` | 30. | Required for 'single_view_target' mode, specify target phi angle as (phi + d_phi) |
| `--d_r` | -.2 | Required for 'single_view_target' mode, specify target radius as (r + r*dr) |
### 3. Diffusion configs
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--ckpt_path` | './checkpoints/ViewCrafter_25.ckpt' | Checkpoint path |
| `--config` | './configs/inference_pvd_1024.yaml' | Config (yaml) path |
| `--ddim_steps` | 50 | Steps of ddim if positive, otherwise use DDPM, reduce to 10 to speed up inference |
| `--ddim_eta` | 1.0 | Eta for ddim sampling (0.0 yields deterministic sampling) |
| `--bs` | 1 | Batch size for inference, should be one |
| `--height` | 576 | Image height, in pixel space |
| `--width` | 1024 | Image width, in pixel space |
| `--frame_stride` | 10 | Fixed |
| `--unconditional_guidance_scale` | 7.5 | Prompt classifier-free guidance |
| `--seed` | 123 | Seed for seed_everything |
| `--video_length` | 25 | Inference video length, change to 16 if you use 16 frame model |
| `--negative_prompt` | False | Unused |
| `--text_input` | False | Unused |
| `--prompt` | 'Rotating view of a scene' | Fixed |
| `--multiple_cond_cfg` | False | Use multi-condition cfg or not |
| `--cfg_img` | None | Guidance scale for image conditioning |
| `--timestep_spacing` | "uniform_trailing" | The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information. |
| `--guidance_rescale` | 0.7 | Guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) |
| `--perframe_ae` | True | If we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024 |
| `--n_samples` | 1 | Num of samples per prompt |
