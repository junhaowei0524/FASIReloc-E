import os, math, torch
import numpy as np
from plyfile import PlyData, PlyElement
from vector_quantize_pytorch import ResidualVQ


def load_vqgaussian(save_dir, device):
    import os
    import numpy as np
    import torch

    meta = np.load(os.path.join(save_dir, "metadata.npz"), allow_pickle=True)["metadata"].item()
    N = meta["input_pc_num"]
    # xyz
    xyz = np.load(os.path.join(save_dir, "xyz.npz"))["arr_0"]
    other_attr = np.load(os.path.join(save_dir, "other_attribute.npz"))["arr_0"]
    # other attr
    mask_bits = np.load(os.path.join(save_dir, "non_vq_mask.npz"))["arr_0"]
    non_vq_mask_full = torch.tensor(np.unpackbits(mask_bits), dtype=torch.bool)
    non_vq_mask = non_vq_mask_full[:N]  # 截断packbits补齐多余bit
    # non_vq mask
    non_vq_sh = np.load(os.path.join(save_dir, "non_vq_feats.npz"))["arr_0"]
    codebook = np.load(os.path.join(save_dir, "codebook.npz"))["arr_0"]

    vq_index_bits = np.load(os.path.join(save_dir, "vq_indexs.npz"))["arr_0"]
    vq_indices_full = torch.tensor(np.unpackbits(vq_index_bits), dtype=torch.int64)
    vq_indices = vq_indices_full[:torch.sum(~non_vq_mask)]

    sh_dim = 48
    sh_all = torch.zeros((N, sh_dim), dtype=torch.float32, device=device)
    sh_all[non_vq_mask] = torch.tensor(non_vq_sh, dtype=torch.float32).to(device)
    sh_all[~non_vq_mask] = torch.tensor(codebook[vq_indices], dtype=torch.float32).to(device)

    loc_feature_all = np.load(os.path.join(save_dir, "loc_feature.npz"))["arr_0"]

    normals = torch.zeros((N, 3), dtype=torch.float32, device=device)
    full_gauss = torch.cat([
        torch.tensor(xyz, dtype=torch.float32).to(device),        # 0-2 xyz
        normals,                                                  #3-5 normals
        sh_all,                                                   #6~53 SH 48dim
        torch.tensor(other_attr, dtype=torch.float32).to(device), #54~61 other_attr 8dim
        torch.tensor(loc_feature_all, dtype=torch.float32).to(device) #62~ loc_feature
    ], dim=-1)
    print(f"full_gauss shape: {full_gauss.shape}")
    return full_gauss

# utils.py load_vqgaussian
def load_vqgaussian_feat(save_dir, device):
    import os
    import numpy as np
    import torch
    meta = np.load(os.path.join(save_dir, "metadata.npz"), allow_pickle=True)["metadata"].item()
    N = meta["input_pc_num"]
    vq_count = meta["vq_count"]
    vq_bit = meta["vq_bit"]
    total_valid_bits = vq_count * vq_bit

    # xyz
    xyz_np = np.load(os.path.join(save_dir, "xyz.npz"))["arr_0"]
    xyz = torch.from_numpy(xyz_np).to(device)

    # other attr
    other_attr_np = np.load(os.path.join(save_dir, "other_attribute.npz"))["arr_0"]
    other_attr = torch.from_numpy(other_attr_np).to(device)

    # non_vq mask
    mask_bits = np.load(os.path.join(save_dir, "non_vq_mask.npz"))["arr_0"]
    non_vq_mask = torch.tensor(np.unpackbits(mask_bits), dtype=torch.bool)[:N].to(device)
    vq_mask = ~non_vq_mask

    sh_codebook_np = np.load(os.path.join(save_dir, "codebook.npz"))["arr_0"]
    sh_codebook = torch.from_numpy(sh_codebook_np).float().to(device)
    sh_packed_bits = np.load(os.path.join(save_dir, "vq_indexs.npz"))["arr_0"]

    bin_indices = np.unpackbits(sh_packed_bits)
    bin_indices = bin_indices[:total_valid_bits]
    bin_indices = bin_indices.reshape(-1, vq_bit)
    sh_idx_np = bin_indices.dot(2 ** np.arange(vq_bit)[::-1])
    sh_indices = torch.from_numpy(sh_idx_np).long().to(device)

    print(f"DEBUG: sh_indices.shape = {sh_indices.shape}, vq_mask sum = {vq_mask.sum().item()}")

    sh_all = torch.zeros((N, 48), dtype=torch.float32, device=device)
    non_vq_sh_np = np.load(os.path.join(save_dir, "non_vq_feats.npz"))["arr_0"]
    sh_all[non_vq_mask] = torch.from_numpy(non_vq_sh_np).float().to(device)
    sh_all[vq_mask] = sh_codebook[sh_indices]

    loc_codebook_np = np.load(os.path.join(save_dir, "loc_codebook.npz"))["arr_0"]
    loc_codebook = torch.from_numpy(loc_codebook_np).float().to(device)
    loc_idx_np = np.load(os.path.join(save_dir, "loc_vq_indexs.npz"))["arr_0"]
    loc_indices = torch.from_numpy(loc_idx_np).long().to(device)
    loc_all = torch.zeros((N,32), dtype=torch.float32, device=device)
    non_vq_loc_np = np.load(os.path.join(save_dir, "non_vq_loc_feats.npz"))["arr_0"]
    loc_all[non_vq_mask] = torch.from_numpy(non_vq_loc_np).float().to(device)
    loc_all[vq_mask] = loc_codebook[loc_indices]

    normals = torch.zeros_like(xyz)
    full_data = torch.cat([
        xyz,
        normals,
        sh_all,
        other_attr,
        loc_all
    ], dim = -1)
    return full_data

def load_vqgaussian_feat_e(path, device="cuda"):
    meta_npz = np.load(os.path.join(path, "metadata.npz"), allow_pickle=True)
    metadata = meta_npz["metadata"].item()
    total_N = metadata["input_pc_num"]
    sh_codebook_size = metadata["codebook_size"]
    sh_dim = metadata["codebook_dim"]

    loc_num_quantizers = metadata["loc_num_quantizers"]
    loc_cb_size = metadata["loc_codebook_size"]
    loc_dim = metadata["loc_codebook_dim"]

    vq_mask_npz = np.load(os.path.join(path, "vq_mask.npz"))
    vq_mask = torch.from_numpy(vq_mask_npz["arr_0"]).to(device)
    # 截断，严格对齐total_N，防止打包bit带来padding
    vq_mask = vq_mask[:total_N]
    non_vq_mask = ~vq_mask

    sh_codebook_np = np.load(os.path.join(path, "codebook.npz"))["arr_0"]
    sh_codebook = torch.from_numpy(sh_codebook_np).float().to(device)

    sh_idx_packed = np.load(os.path.join(path, "vq_indexs.npz"))["arr_0"]
    sh_bits = int(np.log2(sh_codebook_size))
    unpack_bits = np.unpackbits(sh_idx_packed)
    total_sh_indices = metadata["vq_count"]
    unpack_bits = unpack_bits[: total_sh_indices * sh_bits]
    unpack_bits = unpack_bits.reshape(-1, sh_bits)
    powers = (2 ** torch.arange(sh_bits - 1, -1, -1)).to(device)
    sh_vq_indices = (torch.from_numpy(unpack_bits).to(device) * powers).sum(dim=-1).long()

    sh_full = torch.zeros((total_N, sh_dim), dtype=torch.float32, device=device)
    print("sh_full.shape:", sh_full.shape)
    print("vq_mask.shape:", vq_mask.shape)
    print("vq_mask.sum() = ", vq_mask.sum())
    print("sh_vq_indices.shape = ", sh_vq_indices.shape)
    sh_full[vq_mask] = sh_codebook[sh_vq_indices]
    non_vq_sh_np = np.load(os.path.join(path, "non_vq_feats.npz"))["arr_0"]
    sh_full[non_vq_mask] = torch.from_numpy(non_vq_sh_np).float().to(device)

    codebooks_list = []
    loc_idx_list = []
    for lvl in range(loc_num_quantizers):
        cb_np = np.load(os.path.join(path, f"loc_codebook_level{lvl}.npz"))["arr_0"]
        cb_np = np.squeeze(cb_np)
        assert cb_np.ndim == 2, f"Level {lvl} codebook error: {cb_np.shape}"
        idx_np = np.load(os.path.join(path, f"loc_vq_indexs_level{lvl}.npz"))["arr_0"]
        idx_np = np.squeeze(idx_np)
        assert idx_np.ndim == 1, f"Level {lvl} index error: {idx_np.shape}"
        codebooks_list.append(torch.from_numpy(cb_np).float().to(device))
        loc_idx_list.append(torch.from_numpy(idx_np).long().to(device))

    n_vq = loc_idx_list[0].shape[0]
    loc_vq_recon = torch.zeros((n_vq, loc_dim), dtype=torch.float32, device=device)

    for lvl in range(loc_num_quantizers):
        indices = loc_idx_list[lvl]
        cb = codebooks_list[lvl]
        layer_vec = cb[indices]
        print(f"level {lvl}: indices shape {indices.shape}, layer_vec shape {layer_vec.shape}")
        loc_vq_recon += layer_vec

    loc_full = torch.zeros((total_N, loc_dim), dtype=torch.float32, device=device)
    non_vq_loc_np = np.load(os.path.join(path, "non_vq_loc_feats.npz"))["arr_0"]
    loc_full[non_vq_mask] = torch.from_numpy(non_vq_loc_np).float().to(device)
    loc_full[vq_mask] = loc_vq_recon

    xyz_np = np.load(os.path.join(path, "xyz.npz"))["arr_0"]
    xyz = torch.from_numpy(xyz_np).float().to(device)

    other_attr_np = np.load(os.path.join(path, "other_attribute.npz"))["arr_0"]
    other_attr = torch.from_numpy(other_attr_np).float().to(device)

    normals = torch.zeros_like(xyz, device=device)
    full_pts = torch.cat([
        xyz,
        normals,
        sh_full,
        other_attr,
        loc_full
    ], dim=-1)

    return full_pts

def read_ply_data(input_file):
    ply_data = PlyData.read(input_file)
    i = 0
    vertex = ply_data['vertex']
    for prop in vertex._property_lookup:       
        tmp = vertex.data[prop].reshape(-1,1)
        if i == 0:
            data = tmp
            i += 1
        else:
            data = np.concatenate((data, tmp), axis=1)
    return data


def write_ply_data(feats, save_ply_path, sh_dim):
    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # f_dc
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        # f_rest
        for i in range(sh_dim-3):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        base_attr_num = 6 + sh_dim + 8
        loc_dim = feats.shape[-1] - base_attr_num
        for i in range(loc_dim):
            l.append(f'loc_{i}')
        return l

    import os
    os.makedirs(save_ply_path, exist_ok=True)
    path= os.path.join(save_ply_path, 'point_cloud.ply')
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(feats.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, feats))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Save ply, full shape={feats.shape} -> {path}")





def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

