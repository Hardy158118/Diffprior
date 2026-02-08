# import torch.nn as nn
# import torch


# class SimulationDecoder(nn.Module):
#     """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

#     def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
#         super(SimulationDecoder, self).__init__()

#         self.loc_max = loc_max
#         self.loc_min = loc_min
#         self.vel_max = vel_max
#         self.vel_min = vel_min

#         self.interaction_type = suffix

#         if "_springs" in self.interaction_type:
#             print("Using spring simulation decoder.")
#             self.interaction_strength = 0.1
#             # original simulation used sample_freq, _delta_T = 100, 0.001
#             # we use 1, 0.1 instead for computational efficiency
#             self.sample_freq = 1
#             self._delta_T = 0.1
#             self.box_size = 5.0
#         else:
#             print("Simulation type could not be inferred from suffix.")

#         self.out = None

#         # NOTE: For exact reproduction, choose sample_freq=100, delta_T=0.001

#         self._max_F = 0.1 / self._delta_T

#     def unnormalize(self, loc, vel):
#         loc = 0.5 * (loc + 1) * (self.loc_max - self.loc_min) + self.loc_min
#         vel = 0.5 * (vel + 1) * (self.vel_max - self.vel_min) + self.vel_min
#         return loc, vel

#     def renormalize(self, loc, vel):
#         loc = 2 * (loc - self.loc_min) / (self.loc_max - self.loc_min) - 1
#         vel = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
#         return loc, vel

#     def clamp(self, loc, vel):
#         over = loc > self.box_size
#         loc[over] = 2 * self.box_size - loc[over]
#         vel[over] = -torch.abs(vel[over])

#         under = loc < -self.box_size
#         loc[under] = -2 * self.box_size - loc[under]
#         vel[under] = torch.abs(vel[under])

#         return loc, vel

#     def get_offdiag_indices(self, num_nodes):
#         """Linear off-diagonal indices."""
#         ones = torch.ones(num_nodes, num_nodes)
#         eye = torch.eye(num_nodes, num_nodes)
#         offdiag_indices = (ones - eye).nonzero().t()
#         offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
#         return offdiag_indices

#     def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
#         # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
#         # Relation mx shape: [num_sims, num_things*num_things]

#         # Only keep single dimension of softmax output
#         relations = relations[:, :, 1]

#         loc = inputs[:, :, :-1, :2].contiguous()
#         vel = inputs[:, :, :-1, 2:].contiguous()

#         # Broadcasting/shape tricks for parallel processing of time steps
#         loc = loc.permute(0, 2, 1, 3).contiguous()
#         vel = vel.permute(0, 2, 1, 3).contiguous()
#         loc = loc.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)
#         vel = vel.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)

#         loc, vel = self.unnormalize(loc, vel)

#         offdiag_indices = self.get_offdiag_indices(inputs.size(1))
#         edges = torch.zeros(relations.size(0), inputs.size(1) * inputs.size(1))

#         if inputs.is_cuda:
#             edges = edges.cuda()
#             offdiag_indices = offdiag_indices.cuda()

#         edges[:, offdiag_indices] = relations.float()

#         edges = edges.view(relations.size(0), inputs.size(1), inputs.size(1))

#         self.out = []

#         for _ in range(0, self.sample_freq):
#             x = loc[:, :, 0].unsqueeze(-1)
#             y = loc[:, :, 1].unsqueeze(-1)

#             xx = x.expand(x.size(0), x.size(1), x.size(1))
#             yy = y.expand(y.size(0), y.size(1), y.size(1))
#             dist_x = xx - xx.transpose(1, 2)
#             dist_y = yy - yy.transpose(1, 2)

#             forces_size = -self.interaction_strength * edges
#             pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), -1)

#             # Tricks for parallel processing of time steps
#             pair_dist = pair_dist.view(
#                 inputs.size(0), (inputs.size(2) - 1), inputs.size(1), inputs.size(1), 2,
#             )
#             forces = (forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(3)

#             forces = forces.view(
#                 inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2
#             )

#             # Leapfrog integration step
#             vel = vel + self._delta_T * forces
#             loc = loc + self._delta_T * vel

#             # Handle box boundaries
#             loc, vel = self.clamp(loc, vel)

#         loc, vel = self.renormalize(loc, vel)

#         loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
#         vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

#         loc = loc.permute(0, 2, 1, 3)
#         vel = vel.permute(0, 2, 1, 3)

#         out = torch.cat((loc, vel), dim=-1)
#         # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

#         return out

import torch.nn as nn
import torch


class SimulationDecoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
        super(SimulationDecoder, self).__init__()

        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min

        self.interaction_type = suffix

        if "_springs" in self.interaction_type:
            print("Using spring simulation decoder.")
            self.interaction_strength = 0.1
            # original simulation used sample_freq, _delta_T = 100, 0.001
            # we use 1, 0.1 instead for computational efficiency
            self.sample_freq = 1
            self._delta_T = 0.1
            self.box_size = 5.0
        else:
            print("Simulation type could not be inferred from suffix.")

        self.out = None

        # NOTE: For exact reproduction, choose sample_freq=100, delta_T=0.001

        self._max_F = 0.1 / self._delta_T

    def unnormalize(self, loc, vel):
        loc = 0.5 * (loc + 1) * (self.loc_max - self.loc_min) + self.loc_min
        vel = 0.5 * (vel + 1) * (self.vel_max - self.vel_min) + self.vel_min
        return loc, vel

    def renormalize(self, loc, vel):
        loc = 2 * (loc - self.loc_min) / (self.loc_max - self.loc_min) - 1
        vel = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
        return loc, vel

    def clamp(self, loc, vel):
        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        vel[over] = -torch.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        vel[under] = torch.abs(vel[under])

        return loc, vel

    def get_offdiag_indices(self, num_nodes):
        """Linear off-diagonal indices."""
        ones = torch.ones(num_nodes, num_nodes)
        eye = torch.eye(num_nodes, num_nodes)
        offdiag_indices = (ones - eye).nonzero().t()
        offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
        return offdiag_indices

    # def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
    #     # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
    #     # Relation mx shape: [num_sims, num_things*num_things]

    #     # Only keep single dimension of softmax output
    #     relations = relations[:, :, 1]

    #     loc = inputs[:, :, :-1, :2].contiguous()
    #     vel = inputs[:, :, :-1, 2:].contiguous()

    # def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
    #     # relations: [B, N*N, edge_types] 或类似
    #     relations = relations[:, :, 1].contiguous()   # [B, N*N]

    #     B = relations.size(0)
    #     N = inputs.size(1)

    #     # 如果这里不是 N*N，就说明你的 relations 实际是 offdiag 格式，需要走另一套逻辑（见下方）
    #     assert relations.size(1) == N * N, f"relations has {relations.size(1)} edges, expected {N*N}"

    #     edges = relations.reshape(B, N, N).float()
    #     diag = torch.eye(N, device=edges.device, dtype=edges.dtype).unsqueeze(0)
    #     edges = edges * (1.0 - diag)

    #     # 后面 loc/vel 那段保持不变
    #     loc = inputs[:, :, :-1, :2].contiguous()
    #     vel = inputs[:, :, :-1, 2:].contiguous()
    

    #     # Broadcasting/shape tricks for parallel processing of time steps
    #     loc = loc.permute(0, 2, 1, 3).contiguous()
    #     vel = vel.permute(0, 2, 1, 3).contiguous()
    #     loc = loc.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)
    #     vel = vel.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)

    #     loc, vel = self.unnormalize(loc, vel)

    #     offdiag_indices = self.get_offdiag_indices(inputs.size(1))
    #     edges = torch.zeros(relations.size(0), inputs.size(1) * inputs.size(1))

    #     if inputs.is_cuda:
    #         edges = edges.cuda()
    #         offdiag_indices = offdiag_indices.cuda()

    #     edges[:, offdiag_indices] = relations.float()

    #     edges = edges.view(relations.size(0), inputs.size(1), inputs.size(1))

    #     self.out = []

    #     for _ in range(0, self.sample_freq):
    #         x = loc[:, :, 0].unsqueeze(-1)
    #         y = loc[:, :, 1].unsqueeze(-1)

    #         xx = x.expand(x.size(0), x.size(1), x.size(1))
    #         yy = y.expand(y.size(0), y.size(1), y.size(1))
    #         dist_x = xx - xx.transpose(1, 2)
    #         dist_y = yy - yy.transpose(1, 2)

    #         forces_size = -self.interaction_strength * edges
    #         pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), -1)

    #         # Tricks for parallel processing of time steps
    #         pair_dist = pair_dist.view(
    #             inputs.size(0), (inputs.size(2) - 1), inputs.size(1), inputs.size(1), 2,
    #         )
    #         forces = (forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(3)

    #         forces = forces.view(
    #             inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2
    #         )

    #         # Leapfrog integration step
    #         vel = vel + self._delta_T * forces
    #         loc = loc + self._delta_T * vel

    #         # Handle box boundaries
    #         loc, vel = self.clamp(loc, vel)

    #     loc, vel = self.renormalize(loc, vel)

    #     loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
    #     vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

    #     loc = loc.permute(0, 2, 1, 3)
    #     vel = vel.permute(0, 2, 1, 3)

    #     out = torch.cat((loc, vel), dim=-1)
    #     # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

    #     return out
    def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
        """
        inputs:    [B, N, T, D]  (D=4: x,y,vx,vy)
        relations: [B, E, edge_types] or [B, E]
                E could be N*N (with diagonal) or N*(N-1) (offdiag)
        """
        B = inputs.size(0)
        N = inputs.size(1)
        T = inputs.size(2)

        # ---- 1) get relation weights (keep one softmax dim) ----
        # 支持 relations 是 3D 或 2D 两种输入
        if relations.dim() == 3:
            # Only keep single dimension of softmax output
            rel = relations[:, :, 1].contiguous()   # [B, E]
        elif relations.dim() == 2:
            rel = relations.contiguous()            # [B, E]
        else:
            raise ValueError(f"Unexpected relations.dim()={relations.dim()}, shape={relations.shape}")

        E = rel.size(1)

        # ---- 2) build adjacency matrix edges: [B, N, N] ----
        # 支持两种格式：E=N*N or E=N*(N-1)
        if E == N * N:
            edges = rel.view(B, N, N).float()
            # 清零对角线（自连接不参与力）
            mask = ~torch.eye(N, device=edges.device, dtype=torch.bool)  # [N, N]
            edges = edges * mask.unsqueeze(0).to(edges.dtype)
        elif E == N * (N - 1):
            edges = torch.zeros(B, N, N, device=rel.device, dtype=torch.float32)
            mask = ~torch.eye(N, device=rel.device, dtype=torch.bool)  # offdiag mask
            edges[:, mask] = rel.float()
        else:
            raise ValueError(
                f"relations has E={E} edges, but expected N*N={N*N} or N*(N-1)={N*(N-1)} (N={N})."
            )

        # ---- 3) split & reshape states ----
        loc = inputs[:, :, :-1, :2].contiguous()   # [B, N, T-1, 2]
        vel = inputs[:, :, :-1, 2:].contiguous()   # [B, N, T-1, 2]

        # [B, T-1, N, 2]
        loc = loc.permute(0, 2, 1, 3).contiguous()
        vel = vel.permute(0, 2, 1, 3).contiguous()

        # [B*(T-1), N, 2]
        loc = loc.view(B * (T - 1), N, 2)
        vel = vel.view(B * (T - 1), N, 2)

        # unnormalize
        loc, vel = self.unnormalize(loc, vel)

        self.out = []

        # ---- 4) simulation steps ----
        for _ in range(self.sample_freq):
            x = loc[:, :, 0].unsqueeze(-1)  # [B*(T-1), N, 1]
            y = loc[:, :, 1].unsqueeze(-1)  # [B*(T-1), N, 1]

            xx = x.expand(x.size(0), N, N)  # [B*(T-1), N, N]
            yy = y.expand(y.size(0), N, N)

            dist_x = xx - xx.transpose(1, 2)  # [B*(T-1), N, N]
            dist_y = yy - yy.transpose(1, 2)

            # edges: [B, N, N]
            forces_size = -self.interaction_strength * edges  # [B, N, N]

            pair_dist = torch.cat((dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)), dim=-1)
            # pair_dist: [B*(T-1), N, N, 2] -> [B, T-1, N, N, 2]
            pair_dist = pair_dist.view(B, (T - 1), N, N, 2)

            # forces: [B, T-1, N, 2]
            forces = (forces_size.unsqueeze(1).unsqueeze(-1) * pair_dist).sum(3)

            # [B*(T-1), N, 2]
            forces = forces.view(B * (T - 1), N, 2)

            # Leapfrog integration
            vel = vel + self._delta_T * forces
            loc = loc + self._delta_T * vel

            # boundaries
            loc, vel = self.clamp(loc, vel)

        # ---- 5) renormalize & reshape back ----
        loc, vel = self.renormalize(loc, vel)

        loc = loc.view(B, (T - 1), N, 2).permute(0, 2, 1, 3).contiguous()
        vel = vel.view(B, (T - 1), N, 2).permute(0, 2, 1, 3).contiguous()

        out = torch.cat((loc, vel), dim=-1)  # [B, N, T-1, 4]
        return out
