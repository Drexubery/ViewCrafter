import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
#import torchvision.transforms._transforms_video as transforms_video
def string_not_contains_any(substrings, target_string):
    return not any(substring in target_string for substring in substrings)

word = ['digital', 'Digital', 'DIGITAL', 'concept', 'Concept', 'CONCEPT', 'abstract', 'Abstract', 'ABSTRACT', 'particle', 'Particle', 'PARTICLE', 'loop', 'Loop','LOOP']

class WebVid(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 filter_CG=False,
                 human_dynamic=False,
                 sample_basedon_keyframe=False,
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.filter_CG = filter_CG
        self.human_dynamic = human_dynamic
        self.sample_basedon_keyframe = sample_basedon_keyframe
        self._load_metadata()
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize((self.resolution)),
                    ])
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
                
    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path)
        print('Loaded: ', len(metadata))

        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.data_dir, sample['oripath'][1:] if sample['oripath'][0] == '/' else sample['oripath'])
        cond_full_video_fp = os.path.join(self.data_dir, sample['videopath'][1:] if sample['videopath'][0] == '/' else sample['videopath'])
        return full_video_fp, cond_full_video_fp
    
    def __getitem__(self, index):
        ##
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, cond_video_path = self._get_video_path(sample)
            #video_path = "/apdcephfs/share_1290939/0_public_datasets/WebVid/videos/002001_002050/1023214570.mp4"
            caption = sample['caption']
            frameid = int(sample['frameid'])
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    cond_video_reader = VideoReader(cond_video_path, ctx=cpu(0))
                else:
                    NotImplementedError("Must use load_raw_resolution=True")

                if len(video_reader) < self.video_length or len(cond_video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) or Cond video length ({len(cond_video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            frame_stride = 1
            
            start_idx = 0


            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                frames_cond = cond_video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        assert(frames_cond.shape[0] == self.video_length),f'{len(frames_cond)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        frames_cond = torch.tensor(frames_cond.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        

        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            frames_cond = self.spatial_transform(frames_cond)

        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        frames_cond = (frames_cond / 255 - 0.5) * 2

        frames_cond[:,frameid,:,:] = frames[:,frameid,:,:]
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': 10, 'frame_stride': frame_stride, 'video_cond': frames_cond, 'frameid': frameid}
        return data
    
    def __len__(self):
        return len(self.metadata)