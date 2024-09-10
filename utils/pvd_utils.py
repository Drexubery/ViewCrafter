import trimesh
import torch
import numpy as np
import os
import math
import torchvision
from tqdm import tqdm
import cv2  # Assuming OpenCV is used for image saving
from PIL import Image
import pytorch3d
import random
from PIL import ImageGrab
torchvision
from torchvision.utils import save_image
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import imageio
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import copy
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys
sys.path.append('./extern/dust3r')
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision.transforms import CenterCrop, Compose, Resize

def save_video(data,images_path,folder=None):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder]*len(data)
        images = [np.array(Image.open(os.path.join(folder_name,path))) for folder_name,path in zip(folder,data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=8, video_codec='h264', options={'crf': '10'})

def get_input_dict(img_tensor,idx,dtype = torch.float32):

    return {'img':F.interpolate(img_tensor.to(dtype), size=(288, 512), mode='bilinear', align_corners=False), 'true_shape': np.array([[288, 512]], dtype=np.int32), 'idx': idx, 'instance': str(idx), 'img_ori':img_tensor.to(dtype)}
    # return {'img':F.interpolate(img_tensor.to(dtype), size=(288, 512), mode='bilinear', align_corners=False), 'true_shape': np.array([[288, 512]], dtype=np.int32), 'idx': idx, 'instance': str(idx), 'img_ori':ToPILImage()((img_tensor.squeeze(0)+ 1) / 2)}


def rotate_theta(c2ws_input, theta, phi, r, device): 
    # theta: 图像的倾角,新的y’轴(位于yoz平面)与y轴的夹角
    #让相机在以[0,0,depth_avg]为球心的球面上运动,可以先让其在[0,0,0]为球心的球面运动，方便计算旋转矩阵，之后在平移
    c2ws = copy.deepcopy(c2ws_input)
    c2ws[:,2, 3] = c2ws[:,2, 3] + r  #将相机坐标系沿着世界坐标系-z方向平移r
    # 计算旋转向量
    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    v = torch.tensor([0, torch.cos(theta), torch.sin(theta)])
    # 计算反对称矩阵
    v_x = torch.zeros(3, 3).to(device)
    v_x[0, 1] = -v[2]
    v_x[0, 2] = v[1]
    v_x[1, 0] = v[2]
    v_x[1, 2] = -v[0]
    v_x[2, 0] = -v[1]
    v_x[2, 1] = v[0]

    # 计算反对称矩阵的平方
    v_x_square = torch.matmul(v_x, v_x)

    # 计算旋转矩阵
    R = torch.eye(3).to(device) + torch.sin(phi) * v_x + (1 - torch.cos(phi)) * v_x_square

    # 转换为齐次表示
    R_h = torch.eye(4)
    R_h[:3, :3] = R
    Rot_mat = R_h.to(device)

    c2ws = torch.matmul(Rot_mat, c2ws)
    c2ws[:,2, 3]= c2ws[:,2, 3] - r #最后减去r,相当于绕着z=|r|为中心旋转

    return c2ws

def sphere2pose(c2ws_input, theta, phi, r, device):
    c2ws = copy.deepcopy(c2ws_input)

    #先沿着世界坐标系z轴方向平移再旋转
    c2ws[:,2,3] += r

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = torch.tensor([[1, 0, 0, 0],
                    [0, cos_value_x, -sin_value_x, 0],
                    [0, sin_value_x, cos_value_x, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1).to(device)
    
    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = torch.tensor([[cos_value_y, 0, sin_value_y, 0],
                    [0, 1, 0, 0],
                    [-sin_value_y, 0, cos_value_y, 0],
                    [0, 0, 0, 1]]).unsqueeze(0).repeat(c2ws.shape[0],1,1).to(device)
    
    c2ws = torch.matmul(rot_mat_x,c2ws)
    c2ws = torch.matmul(rot_mat_y,c2ws)

    return c2ws 

def generate_candidate_poses(c2ws_anchor,H,W,fs,c,theta, phi,num_candidates,device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """
    if num_candidates == 2:
        thetas = np.array([0,-theta])
        phis = np.array([phi,phi])
    elif num_candidates == 3:
        thetas = np.array([0,-theta,theta/2.]) #avoid too many downward
        phis = np.array([phi,phi,phi])
    else:
        raise ValueError("NBV mode only supports 2 or 3 candidates per iteration.")
    
    c2ws_list = []

    for th, ph in zip(thetas,phis):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), r=None, device= device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)
    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## 将dust3r坐标系转成pytorch3d坐标系
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,thetas,phis

def generate_traj_specified(c2ws_anchor,H,W,fs,c,theta, phi,d_r,frame,device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    thetas = np.linspace(0,theta,frame)
    phis = np.linspace(0,phi,frame)
    rs = np.linspace(0,d_r*c2ws_anchor[0,2,3].cpu(),frame)
    c2ws_list = []
    for th, ph, r in zip(thetas,phis,rs):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)
    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## 将dust3r坐标系转成pytorch3d坐标系
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,num_views

def generate_traj_txt(c2ws_anchor,H,W,fs,c,phi, theta, r,frame,device,viz_traj=False, save_dir = None):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    if len(phi)>3:
        phis = txt_interpolation(phi,frame,mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi,frame,mode='linear')

    if len(theta)>3:
        thetas = txt_interpolation(theta,frame,mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta,frame,mode='linear')
    
    if len(r) >3:
        rs = txt_interpolation(r,frame,mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]        
    else:
        rs = txt_interpolation(r,frame,mode='linear')
    rs = rs*c2ws_anchor[0,2,3].cpu().numpy()

    c2ws_list = []
    for th, ph, r in zip(thetas,phis,rs):
        c2w_new = sphere2pose(c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list,dim=0)

    if viz_traj:
        poses = c2ws.cpu().numpy()
        # visualizer(poses, os.path.join(save_dir,'viz_traj.png'))
        frames = [visualizer_frame(poses, i) for i in range(len(poses))]
        save_video(np.array(frames)/255.,os.path.join(save_dir,'viz_traj.mp4'))

    num_views = c2ws.shape[0]

    R, T = c2ws[:,:3, :3], c2ws[:,:3, 3:]
    ## 将dust3r坐标系转成pytorch3d坐标系
    R = torch.stack([-R[:,:, 0], -R[:,:, 1], R[:,:, 2]], 2) # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0,0,0,1]]]).to(device).repeat(new_c2w.shape[0],1,1)),1))
    R_new, T_new = w2c[:,:3, :3].permute(0,2,1), w2c[:,:3, 3] # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    return cameras,num_views

def setup_renderer(cameras, image_size):
    # Define the settings for rasterization and shading.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius = 0.01,
        points_per_pixel = 10,
        bin_size = 0
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'renderer': renderer}

    return render_setup

def interpolate_sequence(sequence, k,device):

    N, M = sequence.size()
    weights = torch.linspace(0, 1, k+1).view(1, -1, 1).to(device)
    left_values = sequence[:-1].unsqueeze(1).repeat(1, k+1, 1)
    right_values = sequence[1:].unsqueeze(1).repeat(1, k+1, 1)
    new_sequence = torch.einsum("ijk,ijl->ijl", (1 - weights), left_values) + torch.einsum("ijk,ijl->ijl", weights, right_values)
    new_sequence = new_sequence.reshape(-1, M)
    new_sequence = torch.cat([new_sequence, sequence[-1].view(1, -1)], dim=0)
    return new_sequence

def focus_point_fn(c2ws: torch.Tensor) -> torch.Tensor:
    """Calculate nearest point to all focal axes in camera-to-world matrices."""
    # Extract camera directions and origins from c2ws
    directions, origins = c2ws[:, :3, 2:3], c2ws[:, :3, 3:4]
    m = torch.eye(3).to(c2ws.device) - directions * torch.transpose(directions, 1, 2)
    mt_m = torch.transpose(m, 1, 2) @ m
    focus_pt = torch.inverse(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_camera_path(c2ws: torch.Tensor, n_inserts: int = 15, device='cuda') -> torch.Tensor:
    n_poses = c2ws.shape[0] 
    interpolated_poses = []

    for i in range(n_poses-1):
        start_pose = c2ws[i]
        end_pose = c2ws[(i + 1) % n_poses]
        focus_point = focus_point_fn(torch.stack([start_pose,end_pose]))
        interpolated_path = interpolate_poses(start_pose, end_pose, focus_point, n_inserts, device)
        
        # Exclude the last pose (end_pose) for all pairs
        interpolated_path = interpolated_path[:-1]

        interpolated_poses.append(interpolated_path)
    # Concatenate all the interpolated paths
    interpolated_poses.append(c2ws[-1:])
    full_path = torch.cat(interpolated_poses, dim=0)
    return full_path

def interpolate_poses(start_pose: torch.Tensor, end_pose: torch.Tensor, focus_point: torch.Tensor, n_inserts: int = 15, device='cuda') -> torch.Tensor:
    dtype = start_pose.dtype
    start_distance = torch.sqrt((start_pose[0, 3] - focus_point[0])**2 + (start_pose[1, 3] - focus_point[1])**2 + (start_pose[2, 3] - focus_point[2])**2)
    end_distance = torch.sqrt((end_pose[0, 3] - focus_point[0])**2 + (end_pose[1, 3] - focus_point[1])**2 + (end_pose[2, 3] - focus_point[2])**2)
    start_rot = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    end_rot = R.from_matrix(end_pose[:3, :3].cpu().numpy())
    slerp_obj = Slerp([0, 1], R.from_quat([start_rot.as_quat(), end_rot.as_quat()]))

    inserted_c2ws = []

    for t in torch.linspace(0., 1., n_inserts + 2, dtype=dtype):  # Exclude the first and last point
        interpolated_rot = slerp_obj(t).as_matrix()
        interpolated_translation = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]
        interpolated_distance = (1 - t) * start_distance + t * end_distance
        direction = (interpolated_translation - focus_point) / torch.norm(interpolated_translation - focus_point)
        interpolated_translation = focus_point + direction * interpolated_distance

        inserted_pose = torch.eye(4, dtype=dtype).to(device)
        inserted_pose[:3, :3] = torch.from_numpy(interpolated_rot).to(device)
        inserted_pose[:3, 3] = interpolated_translation
        inserted_c2ws.append(inserted_pose)

    path = torch.stack(inserted_c2ws)
    return path

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def save_pointcloud_with_normals(imgs, pts3d, msk, save_path, mask_pc, reduce_pc):
    pc = get_pc(imgs, pts3d, msk,mask_pc,reduce_pc)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))


def get_pc(imgs, pts3d, mask, mask_pc=False, reduce_pc=False):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    if mask_pc:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    else:
        pts = np.concatenate([p for p in pts3d])
        col = np.concatenate([p for p in imgs])

    if reduce_pc:
        pts = pts.reshape(-1, 3)[::3]
        col = col.reshape(-1, 3)[::3]
    else:
        pts = pts.reshape(-1, 3)
        col = col.reshape(-1, 3)
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    # debug
    # pct.export('output.ply')
    # print('exporting output.ply')
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts

def world_to_kth(poses, k):
    # 将世界坐标系转到和第k个pose的相机坐标系一致
    kth_pose = poses[k]
    inv_kth_pose = torch.inverse(kth_pose)
    new_poses = torch.bmm(inv_kth_pose.unsqueeze(0).expand_as(poses), poses)
    return new_poses

def world_point_to_kth(poses, points, k, device):
    # 将世界坐标系转到和第k个pose的相机坐标系一致,同时处理点云
    kth_pose = poses[k]
    inv_kth_pose = torch.inverse(kth_pose)
    # 给所有pose左成kth_w2c,将其都变到kth_pose的camera coordinate下
    new_poses = torch.bmm(inv_kth_pose.unsqueeze(0).expand_as(poses), poses)
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homogeneous_points = torch.cat([points, torch.ones(N, W*H, 1).to(device)], dim=-1)  
    new_points = inv_kth_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1)@ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[...,:3].view(N, W, H, _)

    return new_poses, new_points


def world_point_to_obj(poses, points, k, r, elevation, device):
    ## 作用:将世界坐标系转到object的中心

    ## 先将世界坐标系转到指定相机
    poses, points = world_point_to_kth(poses, points, k, device)
    
    ## 定义目标坐标系位姿, 原点位于object中心(远世界坐标系[0,0,r]),Y轴向上, Z轴垂直屏幕向外, X轴向右
    elevation_rad = torch.deg2rad(torch.tensor(180-elevation)).to(device)
    sin_value_x = torch.sin(elevation_rad)
    cos_value_x = torch.cos(elevation_rad)
    R = torch.tensor([[1, 0, 0,],
                    [0, cos_value_x, sin_value_x],
                    [0, -sin_value_x, cos_value_x]]).to(device)
    
    t = torch.tensor([0, 0, r]).to(device)
    pose_obj = torch.eye(4).to(device)
    pose_obj[:3, :3] = R
    pose_obj[:3, 3] = t

    ## 给所有点和pose乘以目标坐标系的逆(w2c),将它们变换到目标坐标系下
    inv_obj_pose = torch.inverse(pose_obj)
    new_poses = torch.bmm(inv_obj_pose.unsqueeze(0).expand_as(poses), poses)
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homogeneous_points = torch.cat([points, torch.ones(N, W*H, 1).to(device)], dim=-1)  
    new_points = inv_obj_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(1)@ homogeneous_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[...,:3].view(N, W, H, _)
    
    return new_poses, new_points

def txt_interpolation(input_list,n,mode = 'smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew

def visualizer(camera_poses, save_path="out.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["blue" for _ in camera_poses]
    for pose, color in zip(camera_poses, colors):

        camera_positions = pose[:3, 3]
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera trajectory")
    # ax.view_init(90+30, -90)
    plt.savefig(save_path)
    plt.close()

def visualizer_frame(camera_poses, highlight_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 获取camera_positions[2]的最大值和最小值
    z_values = [pose[:3, 3][2] for pose in camera_poses]
    z_min, z_max = min(z_values), max(z_values)

    # 创建一个颜色映射对象
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["#00008B", "#ADD8E6"])
    # cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, pose in enumerate(camera_poses):
        camera_positions = pose[:3, 3]
        color = "blue" if i == highlight_index else "blue"
        size = 100 if i == highlight_index else 25
        color = sm.to_rgba(camera_positions[2])  # 根据camera_positions[2]的值映射颜色
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
            s=size,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Camera trajectory")
    ax.view_init(90+30, -90)

    plt.ylim(-0.1,0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    # new_width = int(width * 0.6)
    # start_x = (width - new_width) // 2 + new_width // 5
    # end_x = start_x + new_width
    # img = img[:, start_x:end_x, :]
    
    
    plt.close()

    return img


def center_crop_image(input_image):

    height = 576
    width = 1024
    _,_,h,w = input_image.shape
    h_ratio = h / height
    w_ratio = w / width

    if h_ratio > w_ratio:
        h = int(h / w_ratio)
        if h < height:
            h = height
        input_image = Resize((h, width))(input_image)
        
    else:
        w = int(w / h_ratio)
        if w < width:
            w = width
        input_image = Resize((height, w))(input_image)

    transformer = Compose([
        # Resize(width),
        CenterCrop((height, width)),
    ])

    input_image = transformer(input_image)
    return input_image


