import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    ## general
    parser.add_argument('--image_dir', type=str, default='./test/images/fruit.png', help='Image file path')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to use')
    parser.add_argument('--exp_name',  type=str, default=None, help='Experiment name, use image file name by default')

    ## renderer
    parser.add_argument('--mode',  type=str,  default='single_view_txt', help="Currently we support 'single_view_txt' and 'single_view_target'")
    parser.add_argument('--traj_txt',  type=str, help="Required for 'single_view_txt' mode, a txt file that specify camera trajectory")
    parser.add_argument('--elevation',  type=float, default=5., help='The elevation angle of the input image in degree. Estimate a rough value based on your visual judgment' )
    parser.add_argument('--center_scale',  type=float, default=1., help='Range: (0, 2]. Scale factor for the spherical radius (r). By default, r is set to the depth value of the center pixel (H//2, W//2) of the reference image')
    parser.add_argument('--d_theta', nargs='+', type=int, default=10., help="Range: [-40, 40]. Required for 'single_view_target' mode, specify target theta angle as theta + d_theta")
    parser.add_argument('--d_phi', nargs='+', type=int, default=30., help="Range: [-45, 45]. Required for 'single_view_target' mode, specify target phi angle as phi + d_phi")
    parser.add_argument('--d_r', nargs='+', type=float, default=-.2, help="Range: [-.5, .5]. Required for 'single_view_target' mode, specify target radius as r + r*dr")
    parser.add_argument('--mask_image', type=bool, default=False, help='Required for mulitpule reference images and iterative mode')
    parser.add_argument('--mask_pc',  type=bool, default=True, help='Required for mulitpule reference images and iterative mode')
    parser.add_argument('--reduce_pc', default=False, help='Required for mulitpule reference images and iterative mode')
    parser.add_argument('--bg_trd',  type=float, default=0., help='Required for mulitpule reference images and iterative mode, set to 0. is no mask')
    parser.add_argument('--dpt_trd',  type=float, default=1., help='Required for mulitpule reference images and iterative mode, limit the max depth by * dpt_trd')


    ## diffusion
    parser.add_argument("--ckpt_path", type=str, default='./checkpoints/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default='./configs/inference_pvd_1024.yaml', help="config (yaml) path")
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM, reduce to 10 to speed up inference")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=576, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=1024, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=10, help="Fixed")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=25, help="inference video length, change to 16 if you use 16 frame model")
    parser.add_argument("--negative_prompt", default=False, help="unused")
    parser.add_argument("--text_input", default=True, help="unused")
    parser.add_argument("--prompt", type=str, default='Rotating view of a scene', help="Fixed")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform_trailing", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.7, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", default=True, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")

    ## dust3r
    parser.add_argument('--model_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', help='The path of the model')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--niter', default=300)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--min_conf_thr', default=3.0) # minimum=1.0, maximum=20

    return parser