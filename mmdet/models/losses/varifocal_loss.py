# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Optional

# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# from mmdet.registry import MODELS
# from .utils import weight_reduce_loss


# def varifocal_loss(pred: Tensor,
#                    target: Tensor,
#                    weight: Optional[Tensor] = None,
#                    alpha: float = 0.75,
#                    gamma: float = 2.0,
#                    iou_weighted: bool = True,
#                    reduction: str = 'mean',
#                    avg_factor: Optional[int] = None) -> Tensor:
#     """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

#     Args:
#         pred (Tensor): The prediction with shape (N, C), C is the
#             number of classes.
#         target (Tensor): The learning target of the iou-aware
#             classification score with shape (N, C), C is the number of classes.
#         weight (Tensor, optional): The weight of loss for each
#             prediction. Defaults to None.
#         alpha (float, optional): A balance factor for the negative part of
#             Varifocal Loss, which is different from the alpha of Focal Loss.
#             Defaults to 0.75.
#         gamma (float, optional): The gamma for calculating the modulating
#             factor. Defaults to 2.0.
#         iou_weighted (bool, optional): Whether to weight the loss of the
#             positive example with the iou target. Defaults to True.
#         reduction (str, optional): The method used to reduce the loss into
#             a scalar. Defaults to 'mean'. Options are "none", "mean" and
#             "sum".
#         avg_factor (int, optional): Average factor that is used to average
#             the loss. Defaults to None.

#     Returns:
#         Tensor: Loss tensor.
#     """
#     # pred and target should be of the same size
#     assert pred.size() == target.size()
#     pred_sigmoid = pred.sigmoid()
#     target = target.type_as(pred)
#     if iou_weighted:
#         focal_weight = target * (target > 0.0).float() + \
#             alpha * (pred_sigmoid - target).abs().pow(gamma) * \
#             (target <= 0.0).float()
#     else:
#         focal_weight = (target > 0.0).float() + \
#             alpha * (pred_sigmoid - target).abs().pow(gamma) * \
#             (target <= 0.0).float()
#     loss = F.binary_cross_entropy_with_logits(
#         pred, target, reduction='none') * focal_weight
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss


# @MODELS.register_module()
# class VarifocalLoss(nn.Module):

#     def __init__(self,
#                  use_sigmoid: bool = True,
#                  alpha: float = 0.75,
#                  gamma: float = 2.0,
#                  iou_weighted: bool = True,
#                  reduction: str = 'mean',
#                  loss_weight: float = 1.0) -> None:
#         """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

#         Args:
#             use_sigmoid (bool, optional): Whether the prediction is
#                 used for sigmoid or softmax. Defaults to True.
#             alpha (float, optional): A balance factor for the negative part of
#                 Varifocal Loss, which is different from the alpha of Focal
#                 Loss. Defaults to 0.75.
#             gamma (float, optional): The gamma for calculating the modulating
#                 factor. Defaults to 2.0.
#             iou_weighted (bool, optional): Whether to weight the loss of the
#                 positive examples with the iou target. Defaults to True.
#             reduction (str, optional): The method used to reduce the loss into
#                 a scalar. Defaults to 'mean'. Options are "none", "mean" and
#                 "sum".
#             loss_weight (float, optional): Weight of loss. Defaults to 1.0.
#         """
#         super().__init__()
#         assert use_sigmoid is True, \
#             'Only sigmoid varifocal loss supported now.'
#         assert alpha >= 0.0
#         self.use_sigmoid = use_sigmoid
#         self.alpha = alpha
#         self.gamma = gamma
#         self.iou_weighted = iou_weighted
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred: Tensor,
#                 target: Tensor,
#                 weight: Optional[Tensor] = None,
#                 avg_factor: Optional[int] = None,
#                 reduction_override: Optional[str] = None) -> Tensor:
#         """Forward function.

#         Args:
#             pred (Tensor): The prediction with shape (N, C), C is the
#                 number of classes.
#             target (Tensor): The learning target of the iou-aware
#                 classification score with shape (N, C), C is
#                 the number of classes.
#             weight (Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Options are "none", "mean" and "sum".

#         Returns:
#             Tensor: The calculated loss
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if self.use_sigmoid:
#             loss_cls = self.loss_weight * varifocal_loss(
#                 pred,
#                 target,
#                 weight,
#                 alpha=self.alpha,
#                 gamma=self.gamma,
#                 iou_weighted=self.iou_weighted,
#                 reduction=reduction,
#                 avg_factor=avg_factor)
#         else:
#             raise NotImplementedError
#         return loss_cls







# # mmdet/models/losses/supcon_loss.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet.utils import get_root_logger
# # from mmcv.runner import BaseModule

# class SupConLoss(BaseModule):
#     """Supervised Contrastive Loss (Khosla et al. 2020) simplified.

#     Args:
#         temperature (float): scaling temperature.
#         base_temperature (float): used for scaling in loss.
#     """
#     def __init__(self, temperature=0.07, base_temperature=0.07, loss_weight=1.0):
#         super().__init__()
#         self.temperature = temperature
#         self.base_temperature = base_temperature
#         self.loss_weight = loss_weight

#     def forward(self, features, labels, mask=None):
#         """
#         Args:
#             features: tensor of shape [N, D] (embeddings) or [N, K, D] if multiple views.
#             labels: tensor of shape [N] with int class labels (>=0). Use -1 to ignore.
#             mask: optional [N, N] mask (bool) where mask[i,j]=1 means positive pair
#         Returns:
#             loss (tensor)
#         """
#         device = features.device
#         if features.ndim == 3:
#             # collapse views: [N, K, D] -> [N*K, D]
#             N, K, D = features.shape
#             features = features.view(N * K, D)
#             labels = labels.unsqueeze(1).repeat(1, K).view(-1)
#         features = F.normalize(features, p=2, dim=1)  # L2 normalize

#         batch_size = features.shape[0]
#         if labels is None and mask is None:
#             raise ValueError("Either labels or mask must be provided")

#         # Compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.t()),
#             self.temperature
#         )

#         # For numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # Mask (remove self-comparisons)
#         diag = torch.eye(batch_size, dtype=torch.bool, device=device)
#         logits_mask = ~diag

#         if mask is None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError("Labels length must match features")
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)

#         # remove self
#         mask = mask * logits_mask.float()

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask.float()
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

#         # mean of log-likelihood over positive
#         mask_sum = mask.sum(1)
#         # avoid division by zero: if an anchor has no positives, ignore it
#         valid = mask_sum > 0
#         if valid.sum() == 0:
#             return features.new_tensor(0.0, requires_grad=True)

#         mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-12)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss[valid].mean() * self.loss_weight
#         return loss


# # mmdet/models/losses/supcon_loss.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet.utils import get_root_logger
# # from mmcv.runner import BaseModule

# class SupConLoss(BaseModule):
#     """Supervised Contrastive Loss (Khosla et al. 2020) simplified.

#     Args:
#         temperature (float): scaling temperature.
#         base_temperature (float): used for scaling in loss.
#     """
#     def __init__(self, temperature=0.07, base_temperature=0.07, loss_weight=1.0):
#         super().__init__()
#         self.temperature = temperature            # Controls how concentrated similarity distribution is
#         self.base_temperature = base_temperature  # Used for final loss scaling
#         self.loss_weight = loss_weight            # Multiplier for final loss value

#     def forward(self, features, labels, mask=None):
#         """
#         Args:
#             features: tensor of shape [N, D] (embeddings) or [N, K, D] if multiple views.
#             labels: tensor of shape [N] with int class labels (>=0). Use -1 to ignore.
#             mask: optional [N, N] mask (bool) where mask[i,j]=1 means positive pair
#         Returns:
#             loss (tensor)
#         """
#         device = features.device  # Get device (CPU/GPU) where features are stored
        
#         if features.ndim == 3:  # Check if we have multiple views per sample
#             # collapse views: [N, K, D] -> [N*K, D]
#             N, K, D = features.shape  # N=batch_size, K=num_views, D=feature_dim
#             features = features.view(N * K, D)  # Flatten to treat each view as separate sample
#             labels = labels.unsqueeze(1).repeat(1, K).view(-1)  # Repeat labels for each view
            
#         features = F.normalize(features, p=2, dim=1)  # L2 normalize: each feature vector has unit length

#         batch_size = features.shape[0]  # Get total number of samples (N or N*K)
        
#         if labels is None and mask is None:  # Need either labels or explicit mask
#             raise ValueError("Either labels or mask must be provided")

#         # Compute logits (similarity scores between all pairs)
#         anchor_dot_contrast = torch.div(  # Compute similarity / temperature
#             torch.matmul(features, features.t()),  # [batch_size, batch_size] dot product matrix
#             self.temperature  # Divide by temperature to scale logits
#         )

#         # For numerical stability (prevents overflow/underflow)
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # Find max per row
#         logits = anchor_dot_contrast - logits_max.detach()  # Subtract max (detach prevents gradient flow)

#         # Mask (remove self-comparisons: sample shouldn't compare with itself)
#         diag = torch.eye(batch_size, dtype=torch.bool, device=device)  # Identity matrix [batch_size, batch_size]
#         logits_mask = ~diag  # Invert diagonal: True everywhere except diagonal

#         if mask is None:  # If no explicit mask, create from labels
#             labels = labels.contiguous().view(-1, 1)  # Reshape to column vector [batch_size, 1]
#             if labels.shape[0] != batch_size:  # Sanity check
#                 raise ValueError("Labels length must match features")
#             mask = torch.eq(labels, labels.T).float().to(device)  # [i,j]=1 if labels[i]==labels[j]
#         else:
#             mask = mask.float().to(device)  # Convert provided mask to float

#         # remove self (samples shouldn't be positive pairs with themselves)
#         mask = mask * logits_mask.float()  # Element-wise multiply: zeros out diagonal

#         # compute log_prob (log probability of each pair)
#         exp_logits = torch.exp(logits) * logits_mask.float()  # Exponentiate and mask diagonal
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)  # Log-softmax with stability epsilon

#         # mean of log-likelihood over positive (average over same-class pairs)
#         mask_sum = mask.sum(1)  # Count positive pairs per sample [batch_size]
        
#         # avoid division by zero: if an anchor has no positives, ignore it
#         valid = mask_sum > 0  # Boolean mask: True if sample has at least one positive
#         if valid.sum() == 0:  # If no valid samples at all
#             return features.new_tensor(0.0, requires_grad=True)  # Return zero loss

#         mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-12)  # Average log prob over positives

#         # loss (negative log-likelihood scaled by temperature ratio)
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # Negate to maximize similarity
#         loss = loss[valid].mean() * self.loss_weight  # Average over valid samples, apply weight
#         return loss