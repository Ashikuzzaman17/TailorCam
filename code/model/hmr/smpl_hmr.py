"""
SMPL body model (the neutral "cocoplus" variant HMR was trained with).

Arrays come from smpl_hmr.npz, written by convert_weights.py:
  v_template (6890,3)  shapedirs (10,6890,3)  posedirs (207,6890,3)
  J_regressor (6890,24) weights (6890,24)  cocoplus_regressor (6890,19)
  faces (13776,3)
"""

import numpy as np
import torch

# Standard SMPL kinematic tree
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19, 20, 21]


def _rodrigues(aa):
    """aa: N x 3 axis-angle → N x 3 x 3 rotation matrices."""
    angle = torch.norm(aa + 1e-8, dim=1, keepdim=True)
    axis = aa / angle
    c, s = torch.cos(angle)[:, :, None], torch.sin(angle)[:, :, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], 1).view(-1, 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype)[None]
    return eye + s * K + (1 - c) * torch.bmm(K, K)


class SMPL(torch.nn.Module):
    def __init__(self, npz_path):
        super().__init__()
        d = np.load(npz_path)
        for k in ('v_template', 'shapedirs', 'posedirs', 'J_regressor',
                  'weights', 'cocoplus_regressor'):
            self.register_buffer(k, torch.from_numpy(d[k].astype(np.float32)))
        self.faces = d['faces'].astype(np.int64)

    def shaped(self, betas):
        """Rest-pose (T-pose) vertices for N x 10 betas → N x 6890 x 3 (metres)."""
        return self.v_template[None] + torch.einsum('nb,bvc->nvc', betas, self.shapedirs)

    def forward(self, betas, pose):
        """Full linear-blend skinning. Returns (verts N x 6890 x 3, joints19 N x 19 x 3)."""
        n = betas.shape[0]
        v_shaped = self.shaped(betas)
        J = torch.einsum('nvc,vj->njc', v_shaped, self.J_regressor)
        R = _rodrigues(pose.reshape(-1, 3)).view(n, 24, 3, 3)
        eye = torch.eye(3, device=pose.device)
        pose_feat = (R[:, 1:] - eye).reshape(n, 207)
        v_posed = v_shaped + torch.einsum('np,pvc->nvc', pose_feat, self.posedirs)

        # Global rigid transforms along the kinematic chain
        G = [None] * 24
        def T(Rm, t):
            top = torch.cat([Rm, t[:, :, None]], 2)
            bot = torch.tensor([0, 0, 0, 1.], device=pose.device).expand(n, 1, 4)
            return torch.cat([top, bot], 1)
        G[0] = T(R[:, 0], J[:, 0])
        for i in range(1, 24):
            G[i] = torch.bmm(G[PARENTS[i]], T(R[:, i], J[:, i] - J[:, PARENTS[i]]))
        G = torch.stack(G, 1)                                   # n,24,4,4
        # Remove rest-pose joint location
        Jh = torch.cat([J, torch.zeros(n, 24, 1, device=pose.device)], 2)
        G = G.clone()
        G[:, :, :3, 3] -= torch.einsum('njab,njb->nja', G[:, :, :3, :], Jh)

        A = torch.einsum('vj,njab->nvab', self.weights, G)
        vh = torch.cat([v_posed, torch.ones(n, v_posed.shape[1], 1, device=pose.device)], 2)
        verts = torch.einsum('nvab,nvb->nva', A, vh)[:, :, :3]
        joints = torch.einsum('nvc,vj->njc', verts, self.cocoplus_regressor)
        return verts, joints
