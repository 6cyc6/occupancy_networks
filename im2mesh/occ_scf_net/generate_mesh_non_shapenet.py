import argparse

import torch
import os
import time
import util
import trimesh
import numpy as np

from im2mesh import config
from im2mesh.common import make_3d_grid
from torch.utils.data import DataLoader
from torch import distributions as dist

from im2mesh.occ_scf_net.dataio_joint import JointNonShapenetOccScfTrainDataset
from im2mesh.utils import libmcubes

from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
from im2mesh.occ_scf_net.dataio import JointOccScfTrainDataset
from im2mesh.occ_scf_net.models import vnn_occupancy_scf_net

# parser = argparse.ArgumentParser(
#     description='Extract meshes from occupancy process.'
# )
# parser.add_argument('config', type=str, help='Path to config file.')
# args = parser.parse_args()
#
# cfg = config.load_config(args.config, '/home/ikun/git_project/occupancy_networks/configs/default.yaml')
threshold = 0.6
box_size = 0.6
resolution0 = 32
upsampling_steps = 2

# load model
device = torch.device('cuda:0')
model = vnn_occupancy_scf_net.VNNOccScfNetMulti(latent_dim=256, o_dim=5, sigmoid=False, return_features=True).cuda()

# load weight
# model_path = "model_final.pth"
model_path = "hammer_model_final.pth"
model_dict = torch.load(model_path)

model.load_state_dict(torch.load(model_path))
model.eval()

val_dataset = JointNonShapenetOccScfTrainDataset(128, o_dim=5, obj_class="hammer", single_view=True, phase="val")

val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True,
                            drop_last=True, num_workers=4, pin_memory=True)

threshold = np.log(threshold) - np.log(1. - threshold)
# generator = config.get_generator(model, cfg, device=device)
with torch.no_grad():
    for step, (model_input, gt) in enumerate(val_dataloader):
        model_input = util.dict_to_gpu(model_input)
        gt = util.dict_to_gpu(gt)

        start_time = time.time()

        test_input = {}
        test_input["point_cloud"] = model_input["point_cloud"][0][None, :, :]
        test_input["coords"] = model_input["coords"][0][None, :, :]
        model_output = model(test_input)

        pcd = test_input["point_cloud"].squeeze(0).cpu().numpy()
        pcd_mean = np.mean(pcd, axis=0)
        pcd = pcd - pcd_mean
        shape_pcd = trimesh.PointCloud(pcd * 1.2)
        bb = shape_pcd.bounding_box
        box_size = np.max(bb.bounds - bb.centroid) * 2.1

        if upsampling_steps == 0:
            nx = resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )

            p = pointsf
            p_split = torch.split(p, 100000)
            occ_hats = []

            for pi in p_split:
                pi = pi.unsqueeze(0).to(device)
                with torch.no_grad():
                    test_input["coords"] = pi
                    logits = model(test_input)["occ"]
                    occ_hat = dist.Bernoulli(logits=logits).logits

                occ_hats.append(occ_hat.squeeze(0).detach().cpu())

            values = torch.cat(occ_hats, dim=0).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(resolution0, upsampling_steps, threshold)
            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                p = pointsf
                p_split = torch.split(p, 100000)
                occ_hats = []

                for pi in p_split:
                    pi = pi.unsqueeze(0).to(device)
                    with torch.no_grad():
                        test_input["coords"] = pi
                        logits = model(test_input)["occ"]
                        occ_hat = dist.Bernoulli(logits=logits).logits

                    occ_hats.append(occ_hat.squeeze(0).detach().cpu())

                values = torch.cat(occ_hats, dim=0).cpu().numpy()

                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        occ_hat = value_grid
        n_x, n_y, n_z = occ_hat.shape

        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        normals = None

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        pcd = test_input["point_cloud"].squeeze(0).cpu().numpy() + np.array([0.1, 0, 0])
        pcl = trimesh.points.PointCloud(pcd)

        scene = trimesh.scene.Scene([mesh, pcl])
        scene.show()
        print(1)
