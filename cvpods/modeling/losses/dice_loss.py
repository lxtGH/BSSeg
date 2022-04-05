import torch


def dice_loss(input, target):
    r"""
    Dice loss defined in the V-Net paper as:

    Loss_dice = 1 - D

            2 * sum(p_i * g_i)
    D = ------------------------------
         sum(p_i ^ 2) + sum(g_i ^ 2)

    where the sums run over the N mask pixels (i = 1 ... N), of the predicted binary segmentation
    pixel p_i ∈ P and the ground truth binary pixel g_i ∈ G.

    Args:
        input (Tensor): predicted binary mask, each pixel value should be in range [0, 1].
        target (Tensor): ground truth binary mask.

    Returns:
        Tensor: dice loss.
    """
    assert input.shape[-2:] == target.shape[-2:]
    input = input.view(input.size(0), -1).float()
    target = target.view(target.size(0), -1).float()

    d = (
        2 * torch.sum(input * target, dim=1)
    ) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4
    )

    return 1 - d


def weighted_dice_loss(
        prediction,
        target_seg,
        gt_num,
        index_mask,
        instance_num: int = 0,
        weighted_val: float = 1.0,
        weighted_num: int = 1,
        mode: str = "thing",
        reduction: str = "sum",
        eps: float = 1e-8,
):
    """
    Weighted version of Dice Loss used in PanopticFCN for multi-positive optimization.

    Args:
        prediction: prediction for Things or Stuff,
        target_seg: segmentation target for Things or Stuff,
        gt_num: ground truth number for Things or Stuff,
        index_mask: positive index mask for Things or Stuff,
        instance_num: instance number of Things or Stuff,
        weighted_val: values of k positives,
        weighted_num: number k for weighted loss,
        mode: used for things or stuff,
        reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum' : The output will be summed.
        eps: the minimum eps,
    """
    # avoid Nan
    if gt_num == 0:
        loss = prediction[0][0].sigmoid().mean() + eps
        return loss * gt_num

    n, _, h, w = target_seg.shape
    if mode == "thing":
        prediction = prediction.reshape(n, instance_num, weighted_num, h, w)
        prediction = prediction.reshape(-1, weighted_num, h, w)[index_mask, ...]
        target_seg = target_seg.unsqueeze(2).expand(n, instance_num, weighted_num, h, w)
        target_seg = target_seg.reshape(-1, weighted_num, h, w)[index_mask, ...]
        weighted_val = weighted_val.reshape(-1, weighted_num)[index_mask, ...]
        weighted_val = weighted_val / torch.clamp(weighted_val.sum(dim=-1, keepdim=True), min=eps)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(int(gt_num), weighted_num, h * w)
        target_seg = target_seg.reshape(int(gt_num), weighted_num, h * w)
    elif mode == "stuff":
        prediction = prediction.reshape(-1, h, w)[index_mask, ...]
        target_seg = target_seg.reshape(-1, h, w)[index_mask, ...]
        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(int(gt_num), h * w)
        target_seg = target_seg.reshape(int(gt_num), h * w)
    else:
        raise ValueError

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    return loss
