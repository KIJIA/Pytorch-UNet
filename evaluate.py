import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    pixel_accuracy = 0
    total_pixels = 0
    correct_pixels = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                # Compute Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

                # Compute IoU (Intersection over Union)
                intersection = torch.sum(mask_pred * mask_true)
                union = torch.sum(mask_pred) + torch.sum(mask_true) - intersection
                iou_score += intersection / (union + 1e-6)

                # Compute Pixel Accuracy
                correct_pixels += torch.sum(mask_pred == mask_true).item()
                total_pixels += mask_true.numel()
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # Convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # Compute Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # Compute IoU (ignoring background)
                for class_idx in range(1, net.n_classes):  # Ignore background (class 0)
                    intersection = torch.sum(mask_pred[:, class_idx] * mask_true[:, class_idx])
                    union = torch.sum(mask_pred[:, class_idx]) + torch.sum(mask_true[:, class_idx]) - intersection
                    iou_score += intersection / (union + 1e-6)

                # Compute Pixel Accuracy (ignoring background)
                mask_true_wo_bg = mask_true[:, 1:]  # Ignore background class
                mask_pred_wo_bg = mask_pred[:, 1:]  # Ignore background class
                correct_pixels += torch.sum(mask_pred_wo_bg == mask_true_wo_bg).item()
                total_pixels += mask_true_wo_bg.numel()

    net.train()

    # Compute the average scores
    dice_score /= max(num_val_batches, 1)
    iou_score /= max(num_val_batches, 1)
    pixel_accuracy = correct_pixels / total_pixels

    return dice_score, iou_score, pixel_accuracy
