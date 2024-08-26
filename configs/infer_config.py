import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--image_dir', type=str, required=True, help='The directory to process')
    parser.add_argument('--out_dir', type=str, required=True, help='The directory to save')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to use for processing')
    parser.add_argument('--exp_name',  type=str, default='test')

    ## dust3r
    parser.add_argument('--model_path', type=str, default='/apdcephfs_cq10/share_1290939/karmyu/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', help='The path of the model')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--niter', default=300)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--min_conf_thr', default=3.0) # minimum=1.0, maximum=20

    ## renderer
    parser.add_argument('--mask_image', type=bool, default=False)
    parser.add_argument('--mask_pc',  type=bool, default=True)
    parser.add_argument('--reduce_pc', default=False)
    parser.add_argument('--bg_trd',  type=float, default=0., help='0. is no mask')
    parser.add_argument('--dpt_trd',  type=float, default=0., help='limit the max depth by * dpt_trd')
    parser.add_argument('--center_scale',  type=float, default=1., help='center * depth in (H//2,W//2)')
    parser.add_argument('--d_theta', nargs='+', type=int, default=10.)
    parser.add_argument('--d_phi', nargs='+', type=int, default=30.)
    parser.add_argument('--d_r', nargs='+', type=float, default=.8)
    parser.add_argument('--elevation',  type=float, default=5.) # garden is 10
    parser.add_argument('--mode',  type=str, default='single_view_specify')
    parser.add_argument('--traj_txt',  type=str, default='')

    ## diffusion
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--prompt", type=str, default='Rotating view of a scene', help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)

    return parser