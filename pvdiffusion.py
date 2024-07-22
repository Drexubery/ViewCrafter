import sys
sys.path.append('./dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis

class PVDiffusion:
    def __init__(self, opts):
        
        self.opts = opts
        self.device = opts.device
        self.setup_dust3r()
        self.setup_diffusion()
        # self.images, self.img_ori = self.loda_initial_images(image_dir=opts.image_dir)
        # self.init_scene(input_images=self.images)

    def run_dust3r(self, input_images):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        # self.scene = scene.clean_pointcloud()
        self.scene = scene

    def render_pcd(self,pts3d,imgs,views,renderer,device):

        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)
        # pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, mask)])).to(device)
        # col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, mask)])).to(device)
        pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        mask = torch.ones(col.shape).to(device)

        point_cloud_mask = Pointclouds(points=[pts],features=[mask]).extend(views)
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)
        masks = renderer(point_cloud_mask)
        return images, masks
    
    def run_render(self, pcd, imgs, H, W, camera_traj,num_views):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_pcd(pcd, imgs,num_views,renderer,self.device)
        # render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        # render_results[0] = self.img_ori

        # save_image(render_results.permute(0,3,1,2), os.path.join(images_path, 'render.jpg'), normalize=True, value_range=(0, 1))
        # save_video(render_results, os.path.join(save_dir, f'{img_name}_{phi}_{theta}.mp4'))
        return render_results, viewmask

    
    def run_diffusion(self, renderings):

        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            # [1,1,c,t,h,w]
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index)

            # save_results_seperate(batch_samples[0], self.opts.save_dir, fps=8)
            # torch.Size([1, 3, 25, 576, 1024]) [-1,1]

        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.) 

    def run_3dgs(self):
        #3dgs
        pass

    def nvs_single_view(self):
        # 最后一个view为 0 pose
        self.images, self.img_ori = self.loda_initial_images(image_dir=self.opts.image_dir)
        self.run_dust3r(input_images=self.images)
        c2ws = self.scene.get_im_poses().detach()[1:] 
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d()] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center #缩放调整

        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=depth_avg, elevation=self.opts.elevation, device=self.device)

        # # masks掉点漂浮点云噪点
        # scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
        # depth = scene.get_depthmaps()
        # depth_avg = torch.mean(depth[-1]).cpu().numpy()
        # gradient_y, gradient_x = np.gradient(depth[-1].cpu().numpy())
        # gradient_magnitude = np.sqrt(gradient_y**2 + gradient_x**2)
        # #调节/15控制mask区域
        # masks = np.expand_dims((gradient_magnitude < depth_avg/15), axis=-1)

        imgs = np.array(self.scene.imgs)
        camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta, self.opts.d_phi, self.opts.d_r,self.opts.video_length, self.device)
        render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]], H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        render_results[0] = self.img_ori
        save_video(render_results, os.path.join(self.opts.save_dir, f'{self.opts.exp_name}_{self.opts.d_phi}_{self.opts.d_theta}_render.mp4'))
        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, sparse_path=self.opts.save_dir , mask_pc=False, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results)
        diffusion_results = (diffusion_results + 1.0) / 2.0
        save_video(diffusion_results, os.path.join(self.opts.save_dir, f'{self.opts.exp_name}_{self.opts.d_phi}_{self.opts.d_theta}_diffusion.mp4'))
    
    def nvs_itrative(self):
        #3dgs
        pass

    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)
    
    def get_input_dict(self,images):
        # 将张量format成和load_images一致的格式
        pass

    def loda_initial_images(self, image_dir):
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),准成张量形式
        images = load_images([image_dir], size=512,force_1024 = True)
        img_ori = Image.open(image_dir).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((576, 1024)),  
            transforms.ToTensor(), 
            transforms.Normalize((0., 0., 0.), (1., 1., 1.))  # 归一化到[-1,1]，如果要归一化到[0,1]，请使用transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        ])

        img_ori = transform(img_ori).permute(1,2,0).to(self.device)
        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1

        return images, img_ori