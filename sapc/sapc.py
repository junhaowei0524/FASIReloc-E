import os, torch, argparse, math
import pickle
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
import numpy as np
from copy import deepcopy
from tqdm import tqdm, trange
# from scene.gaussian_model import FeatureGaussianModel,GaussianModel
from utils import read_ply_data, write_ply_data, load_vqgaussian


def parse_args():
    parser = argparse.ArgumentParser(description="SAPC compression and decompression")
    parser.add_argument("--mode", choices=["compress", "decompress", "both"],
                        default="both", help="Run compression, decompression, or both")
    parser.add_argument("--archive_path", type=str, default=None,
                        help="Optional ZIP archive for decompression")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Directory for reconstructed PLY (default: --save_path)")
    parser.add_argument("--important_score_npz_path", type=str, default='room')
    parser.add_argument("--input_path", type=str, default='room/iteration_40000/point_cloud.ply')
    
    parser.add_argument("--save_path", type=str, default='./output/room')  
    parser.add_argument("--no_load_data", type=bool, default=False)
    parser.add_argument("--no_save_ply", type=bool, default=False)
    parser.add_argument("--sh_degree", type=int, default=3)

    parser.add_argument("--iteration_num", type=int, default=1000)
    parser.add_argument("--vq_ratio", type=float, default=0.6)
    parser.add_argument("--codebook_size", type=int, default=2**13)  # 2**13 = 8192
    parser.add_argument("--no_IS", type=bool, default=False)
    parser.add_argument("--vq_way", type=str, default='half') 
    opt = parser.parse_args() 
    return opt
    

class Quantization():
    def __init__(self, opt):
        
        if opt.sh_degree == 3:
            self.sh_dim = 3+45
        elif opt.sh_degree == 2:
            self.sh_dim = 3+24

        self.feats = read_ply_data(opt.input_path)
        self.feats = torch.tensor(self.feats)
        self.feats_bak = self.feats.clone()
        self.feats = self.feats[:, 6:6+self.sh_dim]

        print("==== debug feats_bak ====")
        print("feats_bak shape =", self.feats_bak.shape)
        row0 = self.feats_bak[0]
        print("row0 total dim: ", row0.shape[0])

        print("xyz [0:3]        :", row0[0:3])
        print("normals [3:6]     :", row0[3:6])
        print("SH [6:6+self.sh_dim]:", row0[6:6 + self.sh_dim])

        sh_end = 6 + self.sh_dim
        print(f"SH ends at index = {sh_end}")
        print("other_attr [sh_end:sh_end+8]:", row0[sh_end:sh_end + 8])
        print("loc_feature [sh_end+8: ]:", row0[sh_end + 8:])

        # ----- define model -----
        from vq import VectorQuantize
        self.model_vq = VectorQuantize(
                    dim = self.feats.shape[1],              
                    codebook_size = opt.codebook_size,
                    decay = 0.8,                            
                    commitment_weight = 1.0,                
                    use_cosine_sim = False,
                    threshold_ema_dead_code=0,
                ).to(device)

        # ----- other -----
        self.save_path = opt.save_path
        self.ply_path = opt.save_path
        self.imp_path = opt.important_score_npz_path
        self.high = None
        self.VQ_CHUNK = 80000
        self.k_expire = 10        

        self.no_IS = opt.no_IS
        self.no_load_data = opt.no_load_data
        self.no_save_ply = opt.no_save_ply
   
        self.codebook_size = opt.codebook_size
        self.iteration_num = opt.iteration_num
        self.vq_way = opt.vq_way

        print("\n================== Print Info ================== ")
        print("Input_feats_shape: ", self.feats_bak.shape)
        print("VQ_feats_shape: ", self.feats.shape)
        print("SH_degree: ", opt.sh_degree)
        print("Add_important_score: ", opt.no_IS==False)
        print("Codebook_size: ", opt.codebook_size)
        print("================================================ ")

    @torch.no_grad()
    def calc_vector_quantized_feature(self):
        CHUNK = 8192
        feat_list = []
        indice_list = []
        self.model_vq.eval()
        self.model_vq._codebook.embed.half().float()   #
        for i in tqdm(range(0, self.feats.shape[0], CHUNK)):
            feat, indices, commit = self.model_vq(self.feats[i:i+CHUNK,:].unsqueeze(0).to(device))
            indice_list.append(indices[0])
            feat_list.append(feat[0])
        self.model_vq.train()
        all_feat = torch.cat(feat_list).half().float()  # [num_elements, feats_dim]
        all_indice = torch.cat(indice_list)             # [num_elements, 1]
        return all_feat, all_indice


    @torch.no_grad()
    def fully_vq_reformat(self):  

        print("\n=============== Start vector quantize ===============")
        all_feat, all_indice = self.calc_vector_quantized_feature()

        if self.save_path is not None:
            save_path = self.save_path
            os.makedirs(f'{save_path}/extreme_saving', exist_ok=True)

            # ----- save basic info -----
            metadata = dict()
            metadata['input_pc_num'] = self.feats_bak.shape[0]  
            metadata['input_pc_dim'] = self.feats_bak.shape[1]  
            metadata['codebook_size'] = self.codebook_size
            metadata['codebook_dim'] = self.sh_dim
            np.savez_compressed(f'{save_path}/extreme_saving/metadata.npz', metadata=metadata)

            def dec2bin(x, bits):
                mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
                return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()    

            self.codebook_vq_index = all_indice[self.vq_mask]
            bin_indices = dec2bin(self.codebook_vq_index.cpu(), int(math.log2(self.codebook_size))).bool().numpy()
            np.savez_compressed(f'{save_path}/extreme_saving/vq_indexs.npz',np.packbits(bin_indices.reshape(-1)))
            
            # ----- save codebook -----                                           
            codebook = self.model_vq._codebook.embed.cpu().half().numpy().squeeze(0)                                                 
            np.savez_compressed(f'{save_path}/extreme_saving/codebook.npz', codebook)

            # ----- save keep mask (non_vq_feats_index)-----
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_mask.npz',np.packbits(self.non_vq_mask.reshape(-1).cpu().numpy()))

            non_vq_feats = self.feats_bak[self.non_vq_mask, 6:6+self.sh_dim]
            wage_non_vq_feats = self.wage_vq(non_vq_feats)
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_feats.npz', wage_non_vq_feats)

            sh_end = 6 + self.sh_dim  # # sh_degree=3，sh_dim=48 → sh_end=54
            other_attribute = self.feats_bak[:, sh_end: sh_end + 8]  # 54:62 opacity/scale/rot
            wage_other_attribute = self.wage_vq(other_attribute)
            np.savez_compressed(f'{save_path}/extreme_saving/other_attribute.npz', wage_other_attribute)

            xyz = self.feats_bak[:, 0:3]
            np.savez_compressed(f'{save_path}/extreme_saving/xyz.npz', xyz)

            loc_feature = self.feats_bak[:, sh_end + 8:]
            np.savez_compressed(f'{save_path}/extreme_saving/loc_feature.npz', loc_feature.cpu().numpy())
            

        # zip everything together to get final size
        subprocess.run(["zip", "-r", f"{save_path}/extreme_saving.zip",
                        f"{save_path}/extreme_saving"], check=True)
        size = os.path.getsize(f'{save_path}/extreme_saving.zip')
        size_MB = size / 1024.0 / 1024.0
        print("Size = {:.2f} MB".format(size_MB))
            
        return all_feat, all_indice
    
    def load_f(self, path, name, allow_pickle=False,array_name='arr_0'):
        return np.load(os.path.join(path, name),allow_pickle=allow_pickle)[array_name]

    def wage_vq(self, feats):
        if self.vq_way == 'half':        
            return feats.half()
        else:
            return feats
    
    def quantize(self):
        salient_idx_path = os.path.join(self.imp_path, "salient_sampled_idx.pkl")
        print(f"Loading salient index from: {salient_idx_path}")
        with open(salient_idx_path, "rb") as f:
            sampled_idx = pickle.load(f)

        total_gauss_num = self.feats.shape[0]
        self.non_vq_mask = torch.zeros(total_gauss_num, dtype=torch.bool, device="cpu")
        self.non_vq_mask[sampled_idx] = True

        self.vq_mask = torch.logical_not(self.non_vq_mask)

        print(f"Total Gaussians: {total_gauss_num}")
        print(f"Salient landmark Gaussians (preserved raw): {torch.sum(self.non_vq_mask).item()}")
        print(f"Non-salient Gaussians (to VQ compress): {torch.sum(self.vq_mask).item()}")

        self.model_vq.train()
        with torch.no_grad():
            # self.vq_mask = torch.logical_xor(self.non_vq_mask)
            feats_needs_vq = self.feats[self.vq_mask].clone()                                       
            # imp = tensor_importance[self.vq_mask].float()
            imp = torch.ones(feats_needs_vq.shape[0]).float().to(device)

            k = self.k_expire                                                              
            if k > self.model_vq.codebook_size:
                k = 0            
            for i in trange(self.iteration_num):
                indexes = torch.randint(low=0, high=feats_needs_vq.shape[0], size=[self.VQ_CHUNK])         
                vq_weight = imp[indexes].to(device)
                vq_feature = feats_needs_vq[indexes,:].to(device)
                quantize, embed, loss = self.model_vq(vq_feature.unsqueeze(0), weight=vq_weight.reshape(1,-1,1))

                replace_val, replace_index = torch.topk(self.model_vq._codebook.cluster_size, k=k, largest=False)      
                _, most_important_index = torch.topk(vq_weight, k=k, largest=True)
                self.model_vq._codebook.embed[:,replace_index,:] = vq_feature[most_important_index,:]

        #=================== Apply vector quantization ====================
        all_feat, all_indices = self.fully_vq_reformat()

    def dequantize(self):
        return restore_directory(Path(self.save_path) / "extreme_saving",
                                 Path(self.ply_path), device, self.no_save_ply)


REQUIRED_FILES = {
    "metadata.npz", "vq_indexs.npz", "codebook.npz", "non_vq_mask.npz",
    "non_vq_feats.npz", "other_attribute.npz", "xyz.npz", "loc_feature.npz",
}


def restore_directory(data_path, output_path, device, no_save_ply=False):
    data_path = Path(data_path)
    missing = sorted(name for name in REQUIRED_FILES if not (data_path / name).is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete compressed map in {data_path}: {missing}")

    with np.load(data_path / "metadata.npz", allow_pickle=True) as saved:
        metadata = saved["metadata"].item()
    sh_dim = int(metadata["codebook_dim"])
    print("\n==================== Dequantize ==================== ")
    dequantized_feats = load_vqgaussian(str(data_path), device=device)
    print("dequantized_feats shape: ", dequantized_feats.shape)
    if not no_save_ply:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        write_ply_data(dequantized_feats.cpu().numpy(), str(output_path), sh_dim)
        print(f"Reconstructed PLY directory: {output_path}")
    return dequantized_feats


def restore_archive(archive_path, output_path, device, no_save_ply=False):
    with tempfile.TemporaryDirectory(prefix="sapc_decode_") as tmp:
        data_path = Path(tmp) / "extreme_saving"
        data_path.mkdir()
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = {}
            for member in archive.infolist():
                parts = member.filename.split("/")
                if member.is_dir() or len(parts) < 2 or parts[-2] != "extreme_saving":
                    continue
                name = parts[-1]
                if name in REQUIRED_FILES:
                    if name in members:
                        raise ValueError(f"Duplicate payload file in archive: {name}")
                    members[name] = member
            missing = sorted(REQUIRED_FILES - members.keys())
            if missing:
                raise FileNotFoundError(f"Incomplete archive {archive_path}: {missing}")
            import shutil
            for name, member in members.items():
                with archive.open(member) as src, (data_path / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return restore_directory(data_path, output_path, device, no_save_ply)


def main():
    global device
    opt = parse_args()
    device = torch.device("cuda")
    output_path = Path(opt.output_path or opt.save_path)
    if opt.archive_path and opt.mode != "decompress":
        raise ValueError("--archive_path is only supported in decompress mode")

    if opt.mode in ("compress", "both"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        vq = Quantization(opt)
        vq.quantize()
        torch.cuda.synchronize()
        print(f"Compression time: {time.perf_counter() - start:.3f} s")
        del vq
        torch.cuda.empty_cache()

    if opt.mode in ("decompress", "both"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        data_path = Path(opt.save_path) / "extreme_saving"
        archive_path = Path(opt.archive_path or (Path(opt.save_path) / "extreme_saving.zip"))
        if opt.mode == "decompress" and (opt.archive_path or archive_path.is_file()):
            restore_archive(archive_path, output_path, device, opt.no_save_ply)
        else:
            restore_directory(data_path, output_path, device, opt.no_save_ply)
        torch.cuda.synchronize()
        print(f"Decompression time: {time.perf_counter() - start:.3f} s")

    print("All done!")


if __name__ == "__main__":
    main()
