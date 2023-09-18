import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
import os.path as osp
from scipy.spatial.transform import Rotation
import pickle

from utils import path_util, geometry
from scipy.special import softmax


def transform_pcd(pcd, transform):
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


class JointNonShapenetOccScfTrainDataset(Dataset):
    def __init__(self, sidelength, single_view=False, depth_aug=False, multiview_aug=False, phase='train', obj_class='all', o_dim=5):
        self.o_dim = o_dim
        # Path setup (change to folder where your training data is kept)
        # these are the names of the full dataset folders
        assert obj_class in ["hammer", "cup"], "Unrecognized object class"

        hammer_path = osp.join(path_util.get_ndf_data(), 'training_data_non_shapenet/hammer')
        cup_path = osp.join(path_util.get_ndf_data(), 'training_data_non_shapenet/cup')

        self.occ_hammer_dict = pickle.load(open(osp.join(path_util.get_ndf_data(), 'training_data_non_shapenet/hammer_occ.p'), 'rb'))
        self.occ_cup_dict = pickle.load(open(osp.join(path_util.get_ndf_data(), 'training_data_non_shapenet/cup_occ.p'), 'rb'))

        if obj_class == 'all':
            paths = [hammer_path, cup_path]
        else:
            paths = []
            if 'hammer' in obj_class:
                paths.append(hammer_path)
            if 'cup' in obj_class:
                paths.append(cup_path)

        print('Loading from paths: ', paths)

        files_total = []
        for path in paths:
            files = list(sorted(glob.glob(path+"/*.npz")))
            n = len(files)
            idx = int(0.9 * n)

            if phase == 'train':
                files = files[:idx]
            else:
                files = files[idx:]

            files_total.extend(files)

        self.files = files_total

        self.sidelength = sidelength
        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug
        self.single_view = single_view

        block = 128
        bs = 1 / block
        hbs = bs * 0.5
        self.bs = bs
        self.hbs = hbs

        self.projection_mode = "perspective"

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            posecam = data['object_pose_cam_frame']

            idxs = list(range(posecam.shape[0]))
            random.shuffle(idxs)
            select = random.randint(1, 4)

            if self.multiview_aug:
                idxs = idxs[:select]

            if self.single_view:
                idxs = [idxs[select - 1]]

            poses = []
            quats = []
            for i in idxs:
                pos = posecam[i, :3]
                quat = posecam[i, 3:]

                poses.append(pos)
                quats.append(quat)

            depths = []
            segs = []
            cam_poses = []

            for i in idxs:
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                rix = np.random.permutation(depth.shape[0])[:1000]
                seg = seg[rix]
                depth = depth[rix]

                if self.depth_aug:
                    depth = depth + np.random.randn(*depth.shape) * 0.003

                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            # change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
            y, x = torch.meshgrid(torch.arange(480), torch.arange(640))
            # x, y = torch.meshgrid(torch.arange(640), torch.arange(480), indexing='xy')

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                [0., vert_f, sensor_half_height, 0.],
                [0., 0., 1., 0.]]
            )

            # Rescale to new sidelength
            intrinsics = torch.from_numpy(intrinsics)

            # build depth images from data
            dp_nps = []
            for i in range(len(segs)):
                seg_mask = segs[i]
                dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
                dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
                dp_nps.append(dp_np)

            # load in voxel occupancy data
            obj_category = data['obj_file'].item().split('/')[0]
            obj_id = data['obj_file'].item().split('/')[-1].replace('.obj', '')
            if obj_category == "hammer":
                coord = self.occ_hammer_dict[f'{obj_category}_{obj_id}'][0]
                voxel_bool = self.occ_hammer_dict[f'{obj_category}_{obj_id}'][1]
            else:
                coord = self.occ_cup_dict[f'{obj_category}_{obj_id}'][0]
                voxel_bool = self.occ_cup_dict[f'{obj_category}_{obj_id}'][1]

            scf_data = np.load(osp.join(path_util.get_ndf_data(),
                               f'training_data_non_shapenet/scfs/{obj_category}/{obj_id}.npy'))
            scf = scf_data[:, 3:]

            rix = np.random.permutation(coord.shape[0])

            coord = coord[rix[:1500]]
            label_scf = scf[rix[:1500], :self.o_dim]
            label_occ = voxel_bool[rix[:1500]]

            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset
            coord = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            # transform everything into the same frame
            transforms = []
            for quat, pos in zip(quats, poses):
                quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                rotation_matrix = Rotation.from_quat(quat_list)
                rotation_matrix = rotation_matrix.as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, -1] = pos
                transform = torch.from_numpy(transform)
                transforms.append(transform)

            transform = transforms[0]
            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            points_world = []

            for i, dp_np in enumerate(dp_nps):
                point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
                dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
                points_world.append(dp_np[..., :3])

            point_cloud = torch.cat(points_world, dim=0)

            rix = torch.randperm(point_cloud.size(0))
            point_cloud = point_cloud[rix[:1000]]
            # point_cloud = point_cloud[rix[:2000]]

            # if point_cloud.size(0) != 2000:
            if point_cloud.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            label_occ = (label_occ - 0.5) * 2.0

            # translate everything to the origin based on the point cloud mean
            center = point_cloud.mean(dim=0)
            coord = coord - center[None, :]
            point_cloud = point_cloud - center[None, :]

            # at the end we have 3D point cloud observation from depth images, voxel occupancy values and corresponding voxel coordinates
            res = {'point_cloud': point_cloud.float(),
                   'coords': coord.float(),
                   'intrinsics': intrinsics.float(),
                   'cam_poses': np.zeros(1)}  # cam poses not used
            return res, {'scf': torch.from_numpy(label_scf).float(), 'occ': torch.from_numpy(label_occ).float()}

        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)


class JointOccScfSdfTrainDataset(Dataset):
    def __init__(self, sidelength, single_view=False, depth_aug=False, multiview_aug=False, phase='train', obj_class='all', o_dim=5):
        self.o_dim = o_dim
        # Path setup (change to folder where your training data is kept)
        # these are the names of the full dataset folders
        assert obj_class in ["hammer"], "Unrecognized object class"

        training_data_root = osp.join(path_util.get_ndf_data(), 'training_data_joint')
        hammer_path = osp.join(training_data_root, 'hammer_pcd')

        if obj_class == 'hammer':
            self.label_path = osp.join(training_data_root, 'labels', obj_class)
        else:
            raise ValueError('Unrecognized object class')

        if obj_class == 'all':
            paths = [hammer_path]
        else:
            paths = []
            if 'hammer' in obj_class:
                paths.append(hammer_path)

        print('Loading from paths: ', paths)

        files_total = []
        for path in paths:
            files = list(sorted(glob.glob(path+"/*.npz")))
            n = len(files)
            idx = int(0.9 * n)

            if phase == 'train':
                files = files[:idx]
            else:
                files = files[idx:]

            files_total.extend(files)

        self.files = files_total

        self.sidelength = sidelength
        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug
        self.single_view = single_view

        block = 128
        bs = 1 / block
        hbs = bs * 0.5
        self.bs = bs
        self.hbs = hbs

        self.projection_mode = "perspective"

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            posecam = data['object_pose_cam_frame']

            idxs = list(range(posecam.shape[0]))
            random.shuffle(idxs)
            select = random.randint(1, 4)

            if self.multiview_aug:
                idxs = idxs[:select]

            if self.single_view:
                idxs = [idxs[select - 1]]

            poses = []
            quats = []
            for i in idxs:
                pos = posecam[i, :3]
                quat = posecam[i, 3:]

                poses.append(pos)
                quats.append(quat)

            depths = []
            segs = []
            cam_poses = []

            for i in idxs:
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                rix = np.random.permutation(depth.shape[0])[:1000]
                seg = seg[rix]
                depth = depth[rix]

                if self.depth_aug:
                    depth = depth + np.random.randn(*depth.shape) * 0.003

                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            # change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
            # y, x = torch.meshgrid(torch.arange(480), torch.arange(640))
            x, y = torch.meshgrid(torch.arange(640), torch.arange(480), indexing='xy')

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                [0., vert_f, sensor_half_height, 0.],
                [0., 0., 1., 0.]]
            )

            # Rescale to new sidelength
            intrinsics = torch.from_numpy(intrinsics)

            # build depth images from data
            dp_nps = []
            for i in range(len(segs)):
                seg_mask = segs[i]
                dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
                dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
                dp_nps.append(dp_np)

            # load in voxel occupancy data
            obj_file = data['obj_file'].item().split('/')[-1].replace('.obj', '')

            if obj_file in os.listdir(osp.join(self.label_path, 'bad_meshes')):
                return self.get_item(index=random.randint(0, self.__len__() - 1))
            if '_dec' in obj_file:
                obj_file = obj_file.replace('_dec', '')
            label_path = osp.join(self.label_path, f'cloud/{obj_file}.npz')
            label_data = np.load(label_path)

            labels = label_data["coords_sdf"]
            norm_factor = label_data["norm_factor"].item()

            coord = labels[:, :3]
            occ = labels[:, 3]
            sdf = labels[:, 4]
            occ[sdf <= self.bs] = 1
            scf = labels[:, 5:]

            rix = np.random.permutation(coord.shape[0])

            coord = coord[rix[:1500]]
            label_occ = occ[rix[:1500]]
            label_sdf = sdf[rix[:1500]]
            label_scf = scf[rix[:1500]]
            # coord = coord[rix[:]]
            # label_occ = occ[rix[:]]
            # label_sdf = sdf[rix[:]]
            # label_scf = scf[rix[:]]

            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset
            coord = coord * norm_factor
            coord = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            # transform everything into the same frame
            transforms = []
            for quat, pos in zip(quats, poses):
                quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                rotation_matrix = Rotation.from_quat(quat_list)
                rotation_matrix = rotation_matrix.as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, -1] = pos
                transform = torch.from_numpy(transform)
                transforms.append(transform)

            transform = transforms[0]
            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            points_world = []

            for i, dp_np in enumerate(dp_nps):
                point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
                dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
                points_world.append(dp_np[..., :3])

            point_cloud = torch.cat(points_world, dim=0)

            rix = torch.randperm(point_cloud.size(0))
            point_cloud = point_cloud[rix[:1000]]
            # point_cloud = point_cloud[rix[:2000]]

            # if point_cloud.size(0) != 2000:
            if point_cloud.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            label_occ = (label_occ - 0.5) * 2.0

            # translate everything to the origin based on the point cloud mean
            center = point_cloud.mean(dim=0)
            coord = coord - center[None, :]
            point_cloud = point_cloud - center[None, :]

            # at the end we have 3D point cloud observation from depth images, voxel occupancy values and corresponding voxel coordinates
            res = {'point_cloud': point_cloud.float(),
                   'coords': coord.float(),
                   'intrinsics': intrinsics.float(),
                   'cam_poses': np.zeros(1)}  # cam poses not used
            return res, {'scf': torch.from_numpy(label_scf).float(), 'occ': torch.from_numpy(label_occ).float(), 'sdf': torch.from_numpy(label_sdf).float()}

        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)




