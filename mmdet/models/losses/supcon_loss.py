import torch
import torch.nn as nn
import torch.nn.functional as F 
from mmdet.utils import get_root_logger
from mmengine.model import BaseModule
from mmdet.registry import MODELS


class SupConLoss(BaseModule):

    """supervised contrastive loss. Khosla et al. https://arxiv.org/pdf/2004.11362.pdf
    Args:
        temperature (float): scaling temperature factor.
        base_temperature (float): used for scaling in SupConLoss
    
    """

    def __init__(self, temperature=0.07, base_temperature=0.07, loss_weight=1.0):
        super().__init__()
        self.temperature = temperature                     # Controls how concentrated similarity distribution is
        self.base_temperature = base_temperature           # Controls how concentrated similarity distribution is
        self.loss_weight = loss_weight                     # Multiplier for final loss value
    def forward(self, features, labels, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss.
        Args:
         features: tensor of shape(N, D) (embedddings) or (N, K, D) (multiple view embeddings)
         labels: tensor of shape [N] with class labels (>=0)
         mask: optional [N, N] mask (bool) where mask[i,j]=1 means positive pair
        Returns:
            loss: scalar tensor containing loss
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        """
        device = features.device
        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...],at least three view required")
        if features.ndim == 3:
            # collapse views [N, K, D] -> [N*K, D]
            N, K, D = features.shape # batch_size, num_views, features_dim
            features = features.view(N * K, D) # Flatten and treat each view as a separate sample
            labels = labels.unsqueeze(1).repeat(1, K).view(-1) # Repreat labels for view

        features = F.normalize(features, p=2, dim=1) # L2 Normalize features

        batch_size = features.shape[0] #total number of samples (N*K)

        if labels is None and mask is None:
            raise ValueError("Either 'labels' or 'mask' must be provided")
        
        # compute logit similarity between all pairs
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.t()), # batch_size x batch_size matrix of dot products
            self.temperature # Scaling by temperature
        )

        # for numerical overflow/underflow
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # find max per row
        logits = anchor_dot_contrast - logits_max.detach() # subtract max from each row


        #Mask - remove self comparisons
        diag = torch.eye(batch_size, dtype=torch.bool).to(device) #identity matrix
        logits_mask = ~diag #invert identity matrix

        if mask is None:
            labels = labels.contiguous().view(-1, 1) # make column vector (batch, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device) # create mask where (i,j)=1 if same class
        else:
            mask = mask.float().to(device)

        # remove self-comparisons from mask
        mask = mask * logits_mask.float()   # #Element-wise multiplication to zero out self-comparisons

        #compute log_prob
        exp_logits = torch.exp(logits) * logits_mask.float() #Exponentiate and maskdiagonal
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12) # Log softmax with stability epsilon

        # mean of log-likelihood over positive
        mask_sum = mask.sum(1) # number of positives per sample(batch)

        # avoid zero division: ignore no positive anchor
        valid = mask_sum > 0
        if valid.sum() == 0:
            return features.new_tensor(0.0, requires_grad=True) # zero loss
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-12) # average log-prob over positives

        # Loss: negative log likelihood scaled by temperature
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos # Negate to maximize simiilarity
        loss = loss[valid].mean() * self.loss_weight # average over valid samples
        return loss