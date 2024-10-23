# @GonzaloMartinGarcia
# This file houses our dataset mixer and training dataset classes.

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2

#################
# Dataset Mixer
#################

class MixedDataLoader:
    def __init__(self, loader1, loader2, split1=9, split2=1):
        self.loader1 = loader1
        self.loader2 = loader2
        self.split1 = split1
        self.split2 = split2
        self.frac1, self.frac2 = self.get_split_fractions()
        self.randchoice1=None

    def __iter__(self):
        self.loader_iter1 = iter(self.loader1)
        self.loader_iter2 = iter(self.loader2)
        self.randchoice1 = self.create_split()
        self.indx = 0
        return self
    
    def get_split_fractions(self):
        size1 = len(self.loader1)
        size2 = len(self.loader2)
        effective_fraction1 = min((size2/size1) * (self.split1/self.split2), 1) 
        effective_fraction2 = min((size1/size2) * (self.split2/self.split1), 1) 
        print("Effective fraction for loader1: ", effective_fraction1)
        print("Effective fraction for loader2: ", effective_fraction2)
        return effective_fraction1, effective_fraction2

    def create_split(self):
        randchoice1 = [True]*int(len(self.loader1)*self.frac1) + [False]*int(len(self.loader2)*self.frac2)
        np.random.shuffle(randchoice1)
        return randchoice1

    def __next__(self):
        if self.indx == len(self.randchoice1):
            raise StopIteration
        if self.randchoice1[self.indx]:
            self.indx += 1
            return next(self.loader_iter1)
        else:
            self.indx += 1
            return next(self.loader_iter2)
        
    def __len__(self):
        return int(len(self.loader1)*self.frac1) + int(len(self.loader2)*self.frac2)
    

#################
# Transforms 
#################

# Hyperism
class SynchronizedTransform_Hyper:
    def __init__(self, H, W):
        self.resize          = transforms.Resize((H,W))
        self.resize_depth    = transforms.Resize((H,W), interpolation=Image.NEAREST)
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor       = transforms.ToTensor()

    def __call__(self, rgb_image, depth_image, normal_image=None):
        # h-flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            depth_image = self.horizontal_flip(depth_image)
            if normal_image is not None:
                normal_image = self.horizontal_flip(normal_image)
                # correct normals for horizontal flip
                np_normal_image = np.array(normal_image)
                np_normal_image[:, :, 0] = 255 - np_normal_image[:, :, 0]
                normal_image = Image.fromarray(np_normal_image)
        # resize
        rgb_image   = self.resize(rgb_image)
        depth_image = self.resize_depth(depth_image)
        if normal_image is not None:
            normal_image = self.resize(normal_image)
        # to tensor
        rgb_tensor = self.to_tensor(rgb_image)
        depth_tensor = self.to_tensor(depth_image)
        if normal_image is not None:
            normal_tensor = self.to_tensor(normal_image)
        # retrun
        if normal_image is not None:
            return rgb_tensor, depth_tensor, normal_tensor
        return rgb_tensor, depth_tensor
    
# Virtual KITTI 2
class SynchronizedTransform_VKITTI:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

    # KITTI benchmark crop from Marigold:
    # https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/dataset/kitti_dataset.py#L75
    @staticmethod
    def kitti_benchmark_crop(input_img):
        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216
        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out
    
    def __call__(self, rgb_image, depth_image, normal_image=None):
        # h-flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            depth_image = self.horizontal_flip(depth_image)
            if normal_image is not None:
                normal_image = self.horizontal_flip(normal_image)
                # correct normals for horizontal flip
                np_normal_image = np.array(normal_image)
                np_normal_image[:, :, 0] = 255 - np_normal_image[:, :, 0]
                normal_image = Image.fromarray(np_normal_image)
        # to tensor
        rgb_tensor = self.to_tensor(rgb_image)      
        depth_tensor = self.to_tensor(depth_image)  
        if normal_image is not None:
            normal_tensor = self.to_tensor(normal_image)
        # kitti benchmark crop
        rgb_tensor = self.kitti_benchmark_crop(rgb_tensor)
        depth_tensor = self.kitti_benchmark_crop(depth_tensor)
        if normal_image is not None:
            normal_tensor = self.kitti_benchmark_crop(normal_tensor)
        # return
        if normal_image is not None:
            return rgb_tensor, depth_tensor, normal_tensor
        return rgb_tensor, depth_tensor
    

#####################
# Training Datasets
#####################

# Hypersim   
class Hypersim(Dataset):
    def __init__(self, root_dir, transform=True, near_plane=1e-5, far_plane=65.0):
        self.root_dir   = root_dir
        self.split_path = os.path.join("data/hypersim/processed/train/filename_meta_train.csv")
        self.near_plane = near_plane
        self.far_plane  = far_plane
        self.align_cam_normal = True
        self.pairs = self._find_pairs()
        self.transform =  SynchronizedTransform_Hyper(H=480, W=640) if transform else None

    def _find_pairs(self):
        df = pd.read_csv(self.split_path)
        pairs = []
        for _, row in df.iterrows():
            if row['included_in_public_release'] and (row['split_partition_name'] == "train"):
                rgb_path = os.path.join(self.root_dir, "train", row['rgb_path'])
                depth_path = os.path.join(self.root_dir, "train", row['depth_path'])
                head, _ = os.path.split(os.path.join(self.root_dir, "train"))
                normal_dir = os.path.join(os.path.join(head, 'normals'), row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_preview',f'frame.{str(row["frame_id"]).zfill(4) }.normal_cam.png')
                if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(normal_dir):
                    pair_info = {'rgb_path': rgb_path, 'depth_path': depth_path, 'normal_path': normal_dir}    
                    pairs.append(pair_info)
        return pairs

    def __len__(self):
        return len(self.pairs)
    
    # Some Hypersim normals are not properly oriented towards the camera.
    # The align_normals and creat_uv_mesh functions are from GeoWizard
    # https://github.com/fuxiao0719/GeoWizard/blob/5ff496579c6be35d9d86fe4d0760a6b5e6ba25c5/geowizard/training/dataloader/file_io.py#L115
    def align_normals(self, normal, depth, K, H, W):
            '''
            Orientation of surface normals in hypersim is not always consistent
            see https://github.com/apple/ml-hypersim/issues/26
            '''
            # inv K
            K = np.array([[K[0],    0, K[2]], 
                          [   0, K[1], K[3]], 
                          [   0,    0,    1]])
            inv_K = np.linalg.inv(K)
            # reprojection depth to camera points
            xy = self.creat_uv_mesh(H, W)
            points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
            points = depth * points
            points = points.transpose((1,2,0))
            # align normal
            orient_mask = np.sum(normal * points, axis=2) > 0
            normal[orient_mask] *= -1
            return normal   
    
    def creat_uv_mesh(self, H, W):
        y, x = np.meshgrid(np.arange(0, H, dtype=np.float64), np.arange(0, W, dtype=np.float64), indexing='ij')
        meshgrid = np.stack((x,y))
        ones = np.ones((1,H*W), dtype=np.float64)
        xy = meshgrid.reshape(2, -1)
        return np.concatenate([xy, ones], axis=0)
    
    def __getitem__(self, idx):
        pairs = self.pairs[idx]

        # get RGB
        rgb_path   = pairs['rgb_path']
        rgb_image  = Image.open(rgb_path).convert('RGB')
        # get depth
        depth_path  = pairs['depth_path']
        depth_image = Image.open(depth_path)
        depth_image = np.array(depth_image)        
        depth_image = depth_image / 1000 # mm to meters
        depth_image = Image.fromarray(depth_image)
        # get normals
        normal_path = pairs['normal_path']
        normal_image = Image.open(normal_path).convert('RGB')
        if self.align_cam_normal:
            # align normals towards camera
            normal_array = (np.array(normal_image) / 255.0) * 2.0 - 1.0
            H, W = normal_array.shape[:2]
            normal_array[:,:,1:] *= -1
            normal_array = self.align_normals(normal_array, np.array(depth_image), [886.81,886.81,W/2, H/2], H, W) * -1
            normal_image = Image.fromarray(((normal_array + 1.0) / 2.0 * 255).astype(np.uint8))
 
        # transfrom
        if self.transform is not None:
            rgb_tensor, depth_tensor, normal_tensor = self.transform(rgb_image, depth_image, normal_image)
        else:
            rgb_tensor    = transforms.ToTensor()(rgb_image)
            depth_tensor  = transforms.ToTensor()(depth_image)
            normal_tensor = transforms.ToTensor()(normal_image)

        # get valid depth mask
        valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)

        # Process RGB 
        rgb_tensor  = rgb_tensor*2.0 - 1.0 # [-1,1]

        # Process depth
        if valid_depth_mask.any():
            flat_depth = depth_tensor[valid_depth_mask].flatten().float()
            min_depth = torch.quantile(flat_depth, 0.02)  
            max_depth = torch.quantile(flat_depth, 0.98)
            if min_depth == max_depth:
                depth_tensor     = torch.zeros_like(depth_tensor)
                metric_tensor    = torch.zeros_like(depth_tensor)
                valid_depth_mask = torch.zeros_like(depth_tensor).bool() # empty mask
            else:
                depth_tensor = torch.clamp(depth_tensor, min_depth, max_depth) # remove outliers
                depth_tensor[~valid_depth_mask] = max_depth                    # set invalid depth to relative far plane
                metric_tensor = depth_tensor.clone()                           # keep metric depth for e2e loss ft   
                depth_tensor = torch.clamp((((depth_tensor - min_depth) / (max_depth - min_depth))*2.0)-1.0, -1, 1) # [-1,1]
        else:
            depth_tensor = torch.zeros_like(depth_tensor)
            metric_tensor = torch.zeros_like(depth_tensor)
        depth_tensor   = torch.stack([depth_tensor, depth_tensor, depth_tensor]).squeeze() # stack depth map for VAE encoder

        # Process normals
        normal_tensor = normal_tensor * 2.0 - 1.0                                 # [-1,1]
        normal_tensor =  torch.nn.functional.normalize(normal_tensor, p=2, dim=0) # normalize
        # set invalid pixels to the zero vector (color grey)
        normal_tensor[0,~valid_depth_mask.squeeze()] = 0
        normal_tensor[1,~valid_depth_mask.squeeze()] = 0
        normal_tensor[2,~valid_depth_mask.squeeze()] = 0
            
        return {"rgb": rgb_tensor, "depth": depth_tensor, 'metric': metric_tensor, 'normals': normal_tensor, "val_mask": valid_depth_mask, "domain": "indoor"}

    
# Virtual KITTI 2.0
class VirtualKITTI2(Dataset):
    def __init__(self, root_dir, transform=None, near_plane=1e-5, far_plane=80.0):
        self.root_dir = root_dir
        self.near_plane = near_plane
        self.far_plane  = far_plane
        self.pairs = self._find_pairs()
        self.transform = SynchronizedTransform_VKITTI() if transform else None

    def _find_pairs(self):
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        weather_conditions = ["morning", "fog", "rain", "sunset", "overcast"]
        cameras = ["Camera_0", "Camera_1"]
        vkitti2_rgb_path = os.path.join(self.root_dir, "vkitti_2.0.3_rgb")
        vkitti2_depth_path =  os.path.join(self.root_dir, "vkitti_2.0.3_depth")
        vkitti2_normal_path = os.path.join(self.root_dir, "vkitti_DAG_normals")
        pairs = []
        for scene in scenes:
            for weather in weather_conditions:
                for camera in cameras:
                    rgb_dir = os.path.join(vkitti2_rgb_path, scene, weather, "frames", "rgb" ,camera)
                    depth_dir = os.path.join(vkitti2_depth_path, scene, weather, "frames","depth" , camera)
                    normal_dir = os.path.join(vkitti2_normal_path, scene, weather, "frames", "normal", camera)
                    if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
                        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
                        rgb_files  = [file[3:] for file in rgb_files]
                        for file in rgb_files:
                            rgb_file = "rgb" + file
                            depth_file = "depth" + file.replace('.jpg', '.png')
                            normal_file = "normal" + file.replace('.jpg', '.png')
                            rgb_path = os.path.join(rgb_dir, rgb_file)
                            depth_path = os.path.join(depth_dir, depth_file)
                            normal_path = os.path.join(normal_dir, normal_file)
                            pairs.append((rgb_path, depth_path, normal_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path, normal_path = self.pairs[idx]

        # get RGB
        rgb_image   = Image.open(rgb_path).convert('RGB')
        # get depth
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.astype(np.float32)/100.0    # cm to meters
        depth_image = Image.fromarray(depth_image)            # PIL
        # get normals
        normal_image = Image.open(normal_path).convert('RGB') 
        
        # transform
        if self.transform is not None:
            rgb_tensor, depth_tensor, normal_tensor = self.transform(rgb_image, depth_image, normal_image)
        else:
            rgb_tensor    = transforms.ToTensor()(rgb_image)
            depth_tensor  = transforms.ToTensor()(depth_image)
            normal_tensor = transforms.ToTensor()(normal_image)
            
        # get valid depth mask
        valid_depth_mask =  (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
            
        # Process RGB
        rgb_tensor = rgb_tensor*2.0 - 1.0 # [-1,1]

        # Process depth
        if valid_depth_mask.any():
            flat_depth = depth_tensor[valid_depth_mask].flatten().float()
            min_depth = torch.quantile(flat_depth, 0.02)  
            max_depth = torch.quantile(flat_depth, 0.98)
            if min_depth == max_depth:
                depth_tensor     = torch.zeros_like(depth_tensor)
                metric_tensor    = torch.zeros_like(depth_tensor)
                valid_depth_mask = torch.zeros_like(depth_tensor).bool() # empty mask
            else:
                depth_tensor = torch.clamp(depth_tensor, min_depth, max_depth) # remove outliers
                depth_tensor[~valid_depth_mask] = max_depth                    # set invalid depth to relative far plane
                metric_tensor = depth_tensor.clone()                           # keep metric depth for e2e loss ft   
                depth_tensor = torch.clamp((((depth_tensor - min_depth) / (max_depth - min_depth))*2.0)-1.0, -1, 1) # [-1,1]
        else:
            depth_tensor = torch.zeros_like(depth_tensor)
            metric_tensor = torch.zeros_like(depth_tensor)
        depth_tensor   = torch.stack([depth_tensor, depth_tensor, depth_tensor]).squeeze() # stack depth map for VAE encoder

        # Process normals
        normal_tensor = normal_tensor * 2.0 - 1.0                                 # [-1,1]
        normal_tensor =  torch.nn.functional.normalize(normal_tensor, p=2, dim=0) # normalize
        # set invalid pixels to the zero vector (color grey)
        normal_tensor[0,~valid_depth_mask.squeeze()] = 0
        normal_tensor[1,~valid_depth_mask.squeeze()] = 0
        normal_tensor[2,~valid_depth_mask.squeeze()] = 0

        return {"rgb": rgb_tensor, "depth": depth_tensor, 'metric': metric_tensor, 'normals': normal_tensor, "val_mask": valid_depth_mask, "domain": "outdoor"}