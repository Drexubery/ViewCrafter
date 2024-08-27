## Important confiuration options for `inference.py`:

### 1. General configs
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--image_dir` | Required | The directory to process |
| `--out_dir` | Required | The directory to save |
| `--device` | `cuda:0` | The device to use for processing |
| `--exp_name` | None | None |
### 2. Point cloud render configs
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--mask_image` | False | None |
| `--mask_pc` | True | None |
| `--reduce_pc` | False | None |
| `--bg_trd` | 0. | 0. is no mask |
| `--dpt_trd` | 0. | Limit the max depth by * dpt_trd |
| `--center_scale` | 1. | Center * depth in (H//2,W//2) |
| `--d_theta` | 10. | None |
| `--d_phi` | 30. | None |
| `--d_r` | .8 | None |
| `--elevation` | 5. | Garden is 10 |
| `--mode` | `single_view_specify` | None |
| `--traj_txt` | '' | None |
### 4. DUSt3R configs
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--model_path` | `/apdcephfs_cq10/share_1290939/karmyu/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | The path of the model |
| `--batch_size` | 1 | None |
| `--schedule` | `linear` | None |
| `--niter` | 300 | None |
| `--lr` | 0.01 | None |
| `--min_conf_thr` | 3.0 | Minimum=1.0, maximum=20 |