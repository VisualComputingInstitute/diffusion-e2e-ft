# @GonzaloMartinGarcia
# This file is based on the official demo.py from the original repository
# https://github.com/fengyi233/depth-to-normal-translator.
# The code was modified to generate Virtual KITTI 2 normals from the ground truth depth maps.

from utils import *

# add
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

###############################
# Modified VKITTI 2 DATALOADER
###############################

# add
# Modified Virtual KITTI 2.0 Dataset class to output the ground truth depth, intrinsics and normal path
class VirtualKITTI2(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.pairs = self._find_pairs()

    def _find_pairs(self):
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        weather_conditions = ["15-deg-left","15-deg-right", "30-deg-left", "30-deg-right", "clone", "morning", "fog", "rain", "sunset", "overcast"]
        cameras = ["Camera_0", "Camera_1"]
        vkitti2_rgb_path    = os.path.join(self.root_dir, "vkitti_2.0.3_rgb")
        vkitti2_depth_path  = os.path.join(self.root_dir, "vkitti_2.0.3_depth")
        vkitti2_normal_path = os.path.join(self.root_dir, "vkitti_DAG_normals") # name of the new normals folder
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
        _, depth_path, normal_path = self.pairs[idx]

        # get depth
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.astype(np.float32)/100.0    # cm to meters
        depth_image = Image.fromarray(depth_image)            # PIL        
        depth_tensor  = transforms.ToTensor()(depth_image)

        # intrinsics (from the official vkitti_2.0.3_textgt.tar files)
        fx_d = 725.0087
        fy_d = 725.0087
        cx_d = 620.5  
        cy_d = 187
        K = torch.tensor([  [fx_d,    0, cx_d],
                            [   0, fy_d, cy_d],
                            [   0,    0,    1]])

        return {"depth": depth_tensor, 'normal_path': normal_path, "intrinsics": K}


####################
# Depth to Normals
####################

# version choices: ['d2nt_basic', 'd2nt_v2', 'd2nt_v3']
VERSION = 'd2nt_v3'

if __name__ == '__main__':
    
    # add
    # init dataset
    print(f"Generating Normals using Version {VERSION}")
    root_dir = "data/virtual_kitti_2"
    dataset = VirtualKITTI2(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"Number of samples: {len(dataset)}")

    for i, data in enumerate(dataloader):

        # add
        depth       = data['depth'][:,0,:,:].squeeze().numpy()*100 # [H, W] 
        intrinsics  = data['intrinsics'].squeeze().numpy()          # [3, 3]
 
        # get camera parameters and depth
        cam_fx, cam_fy, u0, v0 = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
        h, w = depth.shape
        u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0                # u-u0
        v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0  # v-v0

        # DAG Depth to Normals:
        if VERSION == 'd2nt_basic':
            Gu, Gv = get_filter(depth)
        else:
            Gu, Gv = get_DAG_filter(depth)

        # Depth to Normal Translation
        est_nx = Gu * cam_fx
        est_ny = Gv * cam_fy
        est_nz = -(depth + v_map * Gv + u_map * Gu)
        est_normal = cv2.merge((est_nx, est_ny, est_nz))

        # vector normalization
        est_normal = vector_normalization(est_normal)

        # MRF-based Normal Refinement
        if VERSION == 'd2nt_v3':
            est_normal = MRF_optim(depth, est_normal)

        # redirect normals against camera
        est_normal = est_normal * -1 # [H,W,3]

        # add
        # save normals
        est_normal_16bit = ((est_normal + 1) * 32767.5).astype(np.uint16)
        est_normal_16bit = cv2.cvtColor(est_normal_16bit, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(data['normal_path'][0]), exist_ok=True)
        cv2.imwrite(data['normal_path'][0], est_normal_16bit)