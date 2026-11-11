import datetime
import json
import os
import pickle
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from numpy.f2py.symbolic import normalize
from tqdm import tqdm

from JamMa.src.config.default import get_cfg_defaults
from JamMa.src.lightning.lightning_jamma import PL_JamMa
from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from scene.kpdetector import KpDetector, simple_nms
from scene.salient_sample_detector import SalientSampleDetector
from utils.graphics_utils import fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose


# TODO use interpolate
def lift_2d_to_3d(points2d, intrinsic, Twc, depth_map):
    """
    points2d: tensor [N, 2]
    intrinsic: tensor [3, 3]
    Twc: tensor [4, 4]
    depth_map: tensor [H, W]
    """
    device = points2d.device
    depth_idx = points2d.long()
    points2d = points2d + 0.5
    points2d_homo = torch.cat(
        [points2d, torch.ones((points2d.shape[0], 1), device=device)], dim=1
    )
    points3d_camera = (
        torch.inverse(intrinsic)
        @ points2d_homo.T
        * depth_map[depth_idx[:, 1], depth_idx[:, 0]]
    )  # [3, N]
    points3d_camera_homo = torch.cat(
        [
            points3d_camera,
            torch.ones((1, points3d_camera.shape[-1]), device=device),
        ],
        dim=0,
    )  # [4, N]
    points3d_world = Twc @ points3d_camera_homo  # [4, N]
    points3d = points3d_world.T[:, :3]
    return points3d


def sample_gaussians(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(3)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians._loc_feature[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians


def mnn_match(corr_matrix, thr=-1):
    """
    corr_matrix: torch.Tensor, shape (B, N, M)
    """
    mask = corr_matrix > thr
    mask = (
        mask
        * (corr_matrix == corr_matrix.max(dim=-1, keepdim=True)[0])
        * (corr_matrix == corr_matrix.max(dim=-2, keepdim=True)[0])
    )
    b_ids, i_ids, j_ids = torch.where(mask)
    return b_ids.squeeze(), i_ids.squeeze(), j_ids.squeeze()


def topk_match(corr_matrix, topk, thr=-1):
    """
    corr_matrix: torch.Tensor, shape (B, N, M)
    """
    N_im = corr_matrix.shape[-2]
    val, idx = torch.topk(corr_matrix, topk, dim=-1)
    val_flattened = val.flatten(1)
    idx_flattened = idx.flatten(1)
    mask = val_flattened > thr
    arange_tensor = torch.arange(N_im, device=corr_matrix.device)
    idx_im = arange_tensor[None].repeat(corr_matrix.shape[0], topk)[mask]
    idx_gs = idx_flattened[mask]
    val = val_flattened[mask]

    return idx_im, idx_gs, val


def dual_softmax(corr_matrix, temp=1):
    corr_matrix = corr_matrix / temp
    corr_matrix = F.softmax(corr_matrix, dim=-2) * \
        F.softmax(corr_matrix, dim=-1)
    return corr_matrix


def get_intrinsic(fovx, fovy, width, height):
    focalX = fov2focal(fovx, width)
    focalY = fov2focal(fovy, height)
    K = np.array(
        [
            [focalX, 0.0, width / 2],
            [0.0, focalY, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


class FASIReloc_E:
    def __init__(self, gaussians, config, match_config):
        self.gaussians = gaussians
        self.config = config
        self.matcher = PL_JamMa(match_config, pretrained_ckpt='official')
        self.matcher.backbone.to("cuda").eval()
        self.matcher.matcher.to("cuda").eval()
        self.inlier_count = 0
        
        sampled_idx = pickle.load(
            open(
                os.path.join(config["model_path"],
                             "detector/salient_sampled_idx.pkl"),
                "rb",
            )
        )
        self.landmarks = sample_gaussians(
            gaussians, sampled_idx)

        self.feature_extractor = FeatureExtractor(
            config["feature_type"]).cuda().eval()
        self.longest_edge = config["longest_edge"]

        self.detector = SalientSampleDetector(
            self.feature_extractor.feature_dim)
        self.detector.load_state_dict(
            torch.load(
                os.path.join(config["model_path"],
                             config["sparse"]["detector_path"])
            )
        ) 
        self.detector.eval().cuda()  

    @torch.no_grad()
    def localize(self, query_image, fovx, fovy, img_name):
        """
        image: torch.Tensor, shape (3, H, W)
        """
        t0 = time.time()
        query_fine_feature_map, query_coarse_feature_map = self.get_feature_map(
            query_image
        )

        t1 = time.time()
        get_feature_time = t1 - t0

        # Init loc
        t2 = time.time()

        Hc, Wc = query_coarse_feature_map.shape[-2:]
        H = Hc * 8
        W = Wc * 8

        if query_image.shape[-2] != H or query_image.shape[-1] != W:
            query_image = F.interpolate(
                query_image.unsqueeze(0),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
                antialias=True
            ).squeeze(0)


        sparse_result = self.loc_sparse(query_fine_feature_map, fovx, fovy)

        t3 = time.time()
        init_loc_time = t3 - t2

        # refine loc
        t4 = time.time()

        pose_w2c = sparse_result["pose_w2c"]
        dense_results = []
        for iter in range(self.config["dense"]["iters"]):
            dense_result = self.loc_dense(
                query_image, query_coarse_feature_map, pose_w2c, fovx, fovy, img_name
            )
            pose_w2c = dense_result["pose_w2c"]

            dense_results.append(dense_result)

        t5 = time.time()
        refine_loc_time = t5 - t4

        total_time = get_feature_time + init_loc_time + refine_loc_time

        return {"sparse": sparse_result,
                "dense": dense_results,
                "timing": {
                    "get_feature": get_feature_time,
                    "init_loc": init_loc_time,
                    "refine_loc": refine_loc_time,
                    "total": total_time
                }}

    @torch.no_grad()
    def loc_sparse(self, query_feature_map, fovx, fovy):
        
        H, W = query_feature_map.shape[-2:]

        heat_map = self.detector(query_feature_map)  

        kp_scores_after_nms = simple_nms(
            heat_map, self.config["sparse"].get("nms", 4)).flatten()  # [H*W,]
        _, kp_ids = torch.topk(
            kp_scores_after_nms, self.config["sparse"].get("detect_num", 2048))  # [kp,]
        pos_mask = kp_scores_after_nms > 0  
        kp_ids = kp_ids[pos_mask[kp_ids]]  

        kp_mask = torch.zeros_like(
            kp_scores_after_nms, dtype=torch.bool)  # [H*W,]
        kp_mask[kp_ids] = True

        sampled_features = query_feature_map.reshape(
            query_feature_map.shape[0], -1)[:, kp_mask]  # [C,kp]

        landmark_features = F.normalize(
            self.landmarks.get_loc_feature.squeeze(), dim=-1)  # [land,C]

        corr_matrix = torch.matmul(
            sampled_features.T, landmark_features.T)  # [kp,land]

        if config["sparse"]["dual_softmax"] is True:
            corr_matrix = dual_softmax(
                corr_matrix=corr_matrix, temp=config["sparse"]["dual_softmap_temp"]
            )

        if config["sparse"]["mnn_match"] is True:  
            b_ids, im_idx, gs_ids = mnn_match(
                corr_matrix[None], thr=self.config["sparse"]["threshold"]
            )
        else:  
            im_idx, gs_ids, val = topk_match(
                corr_matrix[None],
                self.config["sparse"]["topk"],
                thr=self.config["sparse"]["threshold"],
            )

        p2d = torch.stack([torch.arange(H * W) %
                          W, torch.arange(H * W) // W], dim=1)

        p2d = p2d[kp_mask.cpu()][im_idx.cpu()].numpy()  
        p3d = self.landmarks.get_xyz[gs_ids].cpu().numpy()

        K = get_intrinsic(fovx, fovy, W, H)  

        pose_w2c, inliers = solve_pose(  
            p2d + 0.5,
            p3d,
            K,
            self.config["sparse"]["solver"],
            self.config["sparse"]["reprojection_error"],
            self.config["sparse"]["confidence"],
            self.config["sparse"]["max_iterations"],
            self.config["sparse"]["min_iterations"],
        )

        return {  
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
        }

    @torch.no_grad()
    def loc_dense(self, query, coarse_query_feature_map, pose_w2c, fovx, fovy, img_name):
        
        Hc, Wc = coarse_query_feature_map.shape[-2:]
        C = self.feature_extractor.feature_dim
        K = get_intrinsic(fovx, fovy, Wc * 8, Hc * 8)

        render_pkg = render_from_pose_gsplat(
            self.gaussians,
            torch.tensor(pose_w2c, device="cuda"),
            fovx,
            fovy,
            Wc * 8,
            Hc * 8,
            render_mode="RGB+ED",
            norm_feat_bf_render=self.config["dense"]["norm_before_render"],
            rasterize_mode="antialiased",
        )

        depth = render_pkg["depth"].squeeze()

        fine_rendered_feature_map = render_pkg["feature_map"]
        if (fine_rendered_feature_map == 0).all():
            print("[skip] Rendered feature map is all zero")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}

        if hasattr(gaussians, 'feat_encoder'):
            fine_rendered_feature_map = gaussians.feat_encoder(
                fine_rendered_feature_map.unsqueeze(0)).squeeze(0)

        coarse_rendered_feature_map = F.interpolate(
            fine_rendered_feature_map[None],
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        )[0]
        coarse_rendered_feature_map = F.normalize(
            coarse_rendered_feature_map, dim=0)

        coarse_corr_matrix = torch.matmul(
            coarse_query_feature_map.permute(1, 2, 0).reshape(1, -1, C),
            coarse_rendered_feature_map.reshape(1, C, -1),
        )  
        coarse_corr_matrix = dual_softmax(
            coarse_corr_matrix, temp=self.config["dense"]["coarse_dual_softmax_temp"]
        )

        c_b_ids, c_i_ids, c_j_ids = mnn_match(
            coarse_corr_matrix, thr=self.config["dense"]["coarse_threshold"]
        )

        if c_i_ids.dim() == 0:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}
        elif c_i_ids.shape[0] < 10:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}

        K_tensor = torch.tensor(K, device="cuda")
        B = 1
        h_8 = Hc
        w_8 = Wc
        data = {
            'bs': B,
            'c': C,
            'hw_8': h_8 * w_8,
            'imagec_0': query.unsqueeze(0),
            'imagec_1': render_pkg["render"].unsqueeze(0),
            'K0': K_tensor.unsqueeze(0),  # (3, 3)
            'K1': K_tensor.unsqueeze(0),
        }




        with torch.autocast(enabled=False, device_type='cuda'):
            self.matcher.backbone(data)
            data.update({
                'hw0_i': data['imagec_0'].shape[2:],
                'hw1_i': data['imagec_1'].shape[2:],
                'hw0_c': [data['h_8'], data['w_8']],
                'hw1_c': [data['h_8'], data['w_8']],
            })
            self.matcher.matcher.coarse_match(data)

            mkpts0_c = torch.stack(
                [c_i_ids % data['hw0_c'][1], torch.div(
                    c_i_ids, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * 8
            mkpts1_c = torch.stack(
                [c_j_ids % data['hw1_c'][1], torch.div(
                    c_j_ids, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * 8

            data.update({
                'b_ids': c_b_ids,
                'i_ids': c_i_ids,
                'j_ids': c_j_ids,
                'mkpts0_c': mkpts0_c,
                'mkpts1_c': mkpts1_c
            })
            feat_f0_unfold, feat_f1_unfold = self.matcher.matcher.fine_preprocess(
                data, None)

            self.matcher.matcher.fine_matching(feat_f0_unfold.transpose(
                1, 2), feat_f1_unfold.transpose(1, 2), data)

        pose_c2w = np.linalg.inv(pose_w2c)
        rendered_p2d = data['mkpts1_f']
        p3d = lift_2d_to_3d(rendered_p2d, K_tensor, torch.tensor(
            pose_c2w, device="cuda"), depth)

        query_p2d = data['mkpts0_f'].cpu().numpy()
        p3d = p3d.cpu().numpy()

        pose_w2c_new, inliers = solve_pose(
            query_p2d + 0.5,
            p3d,
            K,
            self.config["dense"]["solver"],
            self.config["dense"]["reprojection_error"],
            self.config["dense"]["confidence"],
            self.config["dense"]["max_iterations"],
            self.config["dense"]["min_iterations"],
        )

        if inliers.shape[0] < 10:
            print("[skip] Failed in fine match")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}

        return {
            "pose_w2c": pose_w2c_new,
            "inliers": inliers.shape[0],
        }

    def get_feature_map(self, image):
        fine_resolution = get_resolution_from_longest_edge(
            image.shape[-2], image.shape[-1], self.longest_edge
        )
        feature_map = self.feature_extractor(
            image[None])["feature_map"]  # 1, C, H, W

        coarse_feature_map = F.interpolate(
            feature_map, size=self.config["coarse_resolution"], mode="bilinear", align_corners=False
        )[0]  
        coarse_feature_map = F.normalize(coarse_feature_map, p=2, dim=0)  
        fine_feature_map = F.interpolate(
            feature_map, size=fine_resolution, mode="bilinear", align_corners=False
        )[0]  
        fine_feature_map = F.normalize(fine_feature_map, p=2, dim=0)  

        return fine_feature_map, coarse_feature_map  


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")  
    model = ModelParams(parser, sentinel=True)  
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)  
    parser.add_argument("--cfg", default=None, type=str)  
    parser.add_argument("--prefix", default=None, type=str)  
    parser.add_argument('--main_cfg_path', type=str, help='main config path')
    parser.add_argument("--feat_dim", type=int, default=32)
    args = get_combined_args(parser)  
    args.eval = True  

    if hasattr(args, "prefix"):  
        output_path = f"results/{args.prefix}-{args.model_path.replace('/', '_')}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_path = f"results/{args.model_path.replace('/', '_')}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("Output path:", output_path)
    os.makedirs(output_path, exist_ok=True)


    dataset = model.extract(args)  
    if dataset.gaussian_type == "3dgs":  
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "fgs":
        from scene.gaussian_model import FeatureGaussianModel
        if dataset.feature_type == 'sp':
            feature_dim = 256
        gaussians = FeatureGaussianModel(
            dataset.sh_degree, args.feat_dim, high_dim=feature_dim)
    else:
        raise ValueError("Gaussian type not supported")
    
    scene = Scene(
        dataset,
        gaussians,
        loc_feat_dim=args.feat_dim,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
    )

    
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)  

    
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type  
    config["longest_edge"] = dataset.longest_edge  
    config["model_path"] = dataset.model_path  

    match_config = get_cfg_defaults()
    match_config.merge_from_file(args.main_cfg_path)

    
    yaml.dump(config, open(os.path.join(
        output_path, os.path.basename(args.cfg)), "w"))

    # loc main
    
    fasireloc_e = FASIReloc_E(gaussians, config, match_config)

    test_cameras = scene.getTestCameras()  
    nums = len(test_cameras)

    results = []  
    sparse_aes = []  
    sparse_tes = []  
    sparse_inliers = []  
    dense_aes = []  
    dense_tes = []  
    dense_inliers = []  
    sparse_time = []  
    dense_time = []  
    get_feature_times = []
    init_loc_times = []
    refine_loc_times = []
    total_times = []

    start_time = time.time()
    num = 0
    for idx, camera_info in enumerate(tqdm(test_cameras, desc="FASIReloc_E")):
        gt_w2c = camera_info.world_view_transform.transpose(
            0, 1).cpu().numpy()  
        query_image = camera_info.original_image.to("cuda")  
        fovx = camera_info.FoVx 
        fovy = camera_info.FoVy

        # localization
        loc_res = fasireloc_e.localize(
            query_image, fovx, fovy, camera_info.image_name)  

        # evaluation
        sparse_ae, sparse_te = cal_pose_error(
            loc_res["sparse"]["pose_w2c"], gt_w2c)  
        sparse_aes.append(sparse_ae)
        sparse_tes.append(sparse_te)
        sparse_inliers.append(loc_res["sparse"]["inliers"])
        loc_res["sparse_AE"] = sparse_ae
        loc_res["sparse_TE"] = sparse_te

        dense_ae, dense_te = cal_pose_error(
            loc_res["dense"][-1]["pose_w2c"], gt_w2c)  # degree, cm
        dense_aes.append(dense_ae)
        dense_tes.append(dense_te)
        dense_inliers.append(loc_res["dense"][-1]["inliers"])
        print(
            f"AE: {dense_ae:.3f}deg, TE: {dense_te:.3f}cm, inliers: {loc_res['dense'][-1]['inliers']}")

        loc_res["gt_pose_w2c"] = gt_w2c.tolist()
        loc_res["dense_AE"] = dense_ae
        loc_res["dense_TE"] = dense_te

        get_feature_times.append(loc_res["timing"]["get_feature"])
        init_loc_times.append(loc_res["timing"]["init_loc"])
        refine_loc_times.append(loc_res["timing"]["refine_loc"])
        total_times.append(loc_res["timing"]["total"])

        results.append(loc_res)
        num += 1

    # get summary
    sparse_aes = np.array(sparse_aes)
    sparse_tes = np.array(sparse_tes)
    dense_aes = np.array(dense_aes)
    dense_tes = np.array(dense_tes)
    end_time = time.time()
    total_time = end_time - start_time

    if num <= 0 or total_time <= 0:
        avg_time = 0.0
        fps = 0.0
    else:
        avg_time = total_time / num
        fps = num / total_time  



    results_summary = {
        "model_path": dataset.model_path,
        "sparse": {
            "median_ae": np.median(sparse_aes),  
            "median_te": np.median(sparse_tes),  
            "recall_025m_2d": ((sparse_aes <= 2) & (sparse_tes <= 25)).sum() / len(sparse_aes),
            "recall_05m_5d": ((sparse_aes <= 5) & (sparse_tes <= 50)).sum() / len(sparse_aes),
            "recall_1m_10d": ((sparse_aes <= 10) & (sparse_tes <= 100)).sum() / len(sparse_aes),
            "recall_5m_10d": ((sparse_aes <= 10) & (sparse_tes <= 500)).sum()
            / len(sparse_aes),
            "recall_2m_5d": ((sparse_aes <= 5) & (sparse_tes <= 200)).sum()
            / len(sparse_aes),
            "recall_5cm_5d": ((sparse_aes <= 5) & (sparse_tes <= 5)).sum()
            / len(sparse_aes),
            "recall_2cm_2d": ((sparse_aes <= 2) & (sparse_tes <= 2)).sum()
            / len(sparse_aes),
            "avg_inliers": np.array(sparse_inliers).mean(), 
        },
        "dense": {
            "median_ae": np.median(dense_aes),
            "median_te": np.median(dense_tes),
            "recall_025m_2d": ((dense_aes <= 2) & (dense_tes <= 25)).sum() / len(sparse_aes),
            "recall_05m_5d": ((dense_aes <= 5) & (dense_tes <= 50)).sum() / len(sparse_aes),
            "recall_1m_10d": ((dense_aes <= 10) & (dense_tes <= 100)).sum() / len(sparse_aes),
            "recall_5m_10d": ((dense_aes <= 10) & (dense_tes <= 500)).sum()
            / len(dense_aes),
            "recall_2m_5d": ((dense_aes <= 5) & (dense_tes <= 200)).sum()
            / len(dense_aes),
            "recall_5cm_5d": ((dense_aes <= 5) & (dense_tes <= 5)).sum()
            / len(dense_aes),
            "recall_2cm_2d": ((dense_aes <= 2) & (dense_tes <= 2)).sum()
            / len(dense_aes),
            "avg_inliers": np.array(dense_inliers).mean(),
        },
        "timing": {
            "mean_get_feature (s)": np.mean(get_feature_times),
            "mean_init_loc (s)": np.mean(init_loc_times),
            "mean_refine_loc (s)": np.mean(refine_loc_times),
            "mean_total (s)": np.mean(total_times),
        },
        "time": {
            "all_time": total_time,  
            "avg_time": avg_time, 
            "fps": fps  
        }
    }

    print("Result Summary:")  
    print(json.dumps(results_summary, indent=4))
    json.dump(
        results_summary, open(os.path.join(output_path, "summary.json"), "w"), indent=4
    )
    for item in results:
        item["sparse"]["pose_w2c"] = item["sparse"]["pose_w2c"].tolist()
        for dense_item in item["dense"]:
            dense_item["pose_w2c"] = dense_item["pose_w2c"].tolist()
    json.dump(results, open(os.path.join(
        output_path, "results.json"), "w"), indent=4)

    print("Result are saved in", output_path)
