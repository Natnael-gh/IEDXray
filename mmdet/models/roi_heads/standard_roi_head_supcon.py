import torch
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.roi_heads import StandardRoIHead
from .contrastive_head import ContrastiveHead


@MODELS.register_module()
class StandardRoIHeadSupCon(StandardRoIHead):

    def __init__(self, contrastive=True, lambda_contrast=0.2, **kwargs):
        super().__init__(**kwargs)
        self.use_contrastive = contrastive
        self.lambda_contrast = lambda_contrast
        if contrastive:
            self.contrastive_head = ContrastiveHead()

    # ---------------------------
    # get roi_feats into loss
    # ---------------------------
    # def _bbox_forward(self, x, rois):
    #     out = super()._bbox_forward(x, rois)
    #     out['roi_feats'] = out['bbox_feats']  # rename for clarity
    #     return out
    def _bbox_forward(self, x, rois):
        # --- This matches exactly StandardRoIHead logic in MMDet 3.3.0 ---
        # extract RoI features
        bbox_feats = self.bbox_roi_extractor(x, rois)

        # shared head (if exists)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        # classification/regression head (returns tuple!)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        # your ROI features should be flattened pooled features
        roi_feats = bbox_feats.flatten(1)

        return dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            roi_feats=roi_feats
        )

    # -----------------------------------------------------------
    # MMDet 3.x performs loss per image → override _loss_single()
    # -----------------------------------------------------------
    def _loss_single(self,
                     cls_score,
                     bbox_pred,
                     labels,
                     label_weights,
                     bbox_targets,
                     bbox_weights,
                     roi_feats):
  

        # 1. Standard detection losses
        losses = super()._loss_single(
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            roi_feats
        )

        # 2. Contrastive loss
        if self.use_contrastive:
            fg = labels >= 0
            feat = roi_feats[fg]
            labs = labels[fg]

            if feat.ndim == 4:
                feat = feat.flatten(1)

            if feat.size(0) > 1:
                emb = self.contrastive_head(feat)

                sim = emb @ emb.T / 0.07
                mask = (labs.unsqueeze(1) == labs.unsqueeze(0)).float()
                mask = mask - torch.eye(mask.size(0), device=mask.device)

                exp_sim = torch.exp(sim) * (
                    1 - torch.eye(sim.size(0), device=sim.device)
                )
                log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

                pos_mean = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
                loss_supcon = -pos_mean.mean()

                losses['loss_supcon'] = self.lambda_contrast * loss_supcon

        return losses

# import torch
# import torch.nn.functional as F

# from mmdet.registry import MODELS
# from mmdet.models.roi_heads import StandardRoIHead
# from .contrastive_head import ContrastiveHead


# @MODELS.register_module()
# class StandardRoIHeadSupCon(StandardRoIHead):

#     def __init__(self, contrastive=True, lambda_contrast=0.1, **kwargs):
#         super().__init__(**kwargs)

#         self.use_contrastive = contrastive
#         self.lambda_contrast = lambda_contrast
#         if contrastive:
#             self.contrastive_head = ContrastiveHead()

#     def _bbox_forward(self, x, rois):
#         """Return bbox features so loss() can access them."""
#         out = super()._bbox_forward(x, rois)
#         out['roi_feats'] = out['bbox_feats']   # rename for clarity
#         return out

#     def loss(self,
#              cls_score,
#              bbox_pred,
#              labels,
#              label_weights,
#              bbox_targets,
#              bbox_weights,
#              roi_feats):
        
#         # --------- standard detection loss ----------
#         losses = super().loss(
#             cls_score,
#             bbox_pred,
#             labels,
#             label_weights,
#             bbox_targets,
#             bbox_weights
#         )

#         # ---------- contrastive loss ----------
#         if self.use_contrastive:

#             fg = labels >= 0
#             feat = roi_feats[fg]
#             labs = labels[fg]

#             if feat.ndim == 4:
#                 feat = feat.flatten(1)

#             if feat.size(0) > 1:
#                 emb = self.contrastive_head(feat)
#                 sim = emb @ emb.T / 0.07

#                 mask = (labs.unsqueeze(1) == labs.unsqueeze(0)).float()
#                 mask = mask - torch.eye(mask.size(0), device=mask.device)

#                 exp_sim = torch.exp(sim) * (1 - torch.eye(sim.size(0), device=sim.device))
#                 log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

#                 pos_mean = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
#                 loss_supcon = -pos_mean.mean()

#                 losses['loss_supcon'] = self.lambda_contrast * loss_supcon

#         return losses

# # from mmdet.models.roi_heads import StandardRoIHead

# # from.contrastive_head import ContrastiveHead
# # import torch
# # import torch.nn.functional as F

# # from mmdet.registry import MODELS

# # @MODELS.register_module()
# # class StandardRoIHeadSupCon(StandardRoIHead):
# #     def __init__(self, contrastive =True, lambda_contrast=0.1, **kwargs): #lambda_contrast is the weight for the contrastive loss
# #         super().__init__(**kwargs)
# #         self.use_contrastive = contrastive
# #         self.lambda_contrast = lambda_contrast
# #         if self.use_contrastive:
# #             self.contrastive_head = ContrastiveHead()
# #     def loss(self, cls_score, bbox_pred, roi_feats, labels, label_weights, bbox_targets, bbox_weights):

# #         #Standard dete loss
# #         losses =super().loss(cls_score, bbox_pred, roi_feats, labels, label_weights, bbox_targets, bbox_weights)

# #         #-----
# #         #Contrastive loss
# #         #-----

# #         if self.use_contrastive:

# #             # fg RoIs
# #             fg_mask = labels >= 0  # positive samples have labels >= 0
# #             feat = roi_feats[fg_mask]  # (num_fg, C, H, W)
# #             labs = labels[fg_mask]  # (num_fg,)

# #             if feat.size(0) > 1:
# #                 emb = self.contrastive_head(feat)


# #                 # Compute contrastive loss
# #                 sim_matrix = torch.div(torch.matmul(emb, emb.T), 0.07 # temperature parameter
# #                 )  # (num_fg, num_fg)

# #                 mask = labs.unsqueeze(1).eq(labs.unsqueeze(0)).float()  # Explanation: here we create a mask where mask[i][j] = 1 if labs[i] == labs[j], else 0
# #                 mask = mask - torch.eye(mask.size(0), device=mask.device)  # demo how it works: if labs = [1,2,1], then mask = [[0,0,1],[0,0,0],[1,0,0]] the code will set diagonal to 0 in this case for i=j, we don't want to consider self-similarity

# #                 exp_sim = torch.exp(sim_matrix)*(1-torch.eye(sim_matrix.size(0), device=sim_matrix.device))  # zero out self-similarity

# #                 log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True)+1e-9)  # log prob for each sample
# #                 pos_mean = (mask * log_prob).sum(dim=1) / (mask.sum(1) + 1e-9)
                
# #                 loss_supcon = -pos_mean.mean()
# #             else:
# #                 loss_supcon = torch.tensor(0.0, device=labels.device)
            
# #             losses['loss_supcon'] = self.lambda_contrast * loss_supcon

# #         return losses
    
# #     def _bbox_forward(self, x, rois):
# #         bbox_results = super()._bbox_forward(x, rois)
# #         cls_score, bbox_pred, roi_feats = bbox_results
# #         return dict(
# #             cls_score=cls_score, bbox_pred=bbox_pred, roi_feats=roi_feats
# #         )