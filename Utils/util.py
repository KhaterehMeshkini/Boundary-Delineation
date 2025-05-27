#from post_processing import *
import glob as gl
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pytorch_msssim import SSIM
#from piqa import SSIM
import torch.nn.functional as N
from skimage.morphology import skeletonize, dilation, square
from skimage.util import invert
from scipy.ndimage import convolve
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_cdt, label
from skimage.measure import regionprops
from sklearn.metrics import confusion_matrix


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if isinstance(item, str):
            item = np.array(Image.open(item))
        elif isinstance(item, Image.Image):
            item = np.array(item)
        elif isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy / len(np_ims[0].flatten())

def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    for index in range(batch_size):
        total_acc += accuracy_check(masks[index], predictions[index])
    return total_acc / batch_size

def compute_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union
    return iou

def thin_boundaries(binary_boundary, thickness=2):

    # Thin lines to 1-pixel width
    skeleton = skeletonize(binary_boundary)

    # Thicken lines to desired thickness
    uniform_width = dilation(skeleton, square(thickness))

    # Ensure binary output
    processed_pred = (uniform_width > 0).astype(np.uint8)

    return processed_pred



def postprocess_boundary(pred_np, complete=True, thickness=2, connect_thresh=10):

    binary_boundary = (pred_np == 1).astype(np.uint8)

    # 1. Skeletonize to thin to 1-pixel
    skeleton = skeletonize(binary_boundary)

    if complete:
        # Find endpoints in skeleton
        kernel = np.array([[1,1,1],
                           [1,10,1],
                           [1,1,1]])
        neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant')
        endpoints = (neighbors == 11)
        coords = np.column_stack(np.where(endpoints))

        # Build cost array for pathfinding
        cost_array = invert(skeleton).astype(np.uint8) + 1
        connected = skeleton.copy()

        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                p1 = tuple(coords[i])
                p2 = tuple(coords[j])
                if np.linalg.norm(np.array(p1) - np.array(p2)) < connect_thresh:
                    try:
                        path, _ = route_through_array(cost_array, p1, p2, fully_connected=True)
                        for r, c in path:
                            connected[r, c] = 1
                    except:
                        continue
        skeleton = connected

    # 2. Thicken to ~2-pixel lines
    processed_pred = dilation(skeleton, square(thickness)).astype(np.uint8)
    return processed_pred


class CombinedCELSSIMLoss(nn.Module):
    def __init__(self, alpha=0.5, class_weights=None):
        super(CombinedCELSSIMLoss, self).__init__()
        self.alpha = alpha
        self.class_weights = class_weights  # Store for later move to device
        self.ce_loss = None  # Delayed creation

    def forward(self, logits, targets):
        if self.ce_loss is None:
            # Move weights to same device as logits
            self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)

        ce = self.ce_loss(logits, targets)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prob = N.softmax(logits, dim=1).float()
        pred_mask = torch.argmax(prob, dim=1).unsqueeze(1).float()
        true_mask = targets.unsqueeze(1).float()

        ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
  
        loss = ssim_loss(pred_mask, true_mask)

        ssim_loss = 1.0 - loss

        combined = (1 - self.alpha) * ce + self.alpha * ssim_loss
        return combined
    

def count_matched_objects(ref_labeled, pred_labeled, min_iou_threshold=0.3):
    ref_ids = np.unique(ref_labeled)
    ref_ids = ref_ids[ref_ids != 0]  # remove background
    pred_ids = np.unique(pred_labeled)
    pred_ids = pred_ids[pred_ids != 0]

    matched_refs = set()

    for ref_id in ref_ids:
        ref_mask = (ref_labeled == ref_id)

        for pred_id in pred_ids:
            pred_mask = (pred_labeled == pred_id)

            intersection = np.logical_and(ref_mask, pred_mask).sum()
            union = np.logical_or(ref_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0

            if iou >= min_iou_threshold:
                matched_refs.add(ref_id)
                break  # Only count the first match

    num_matched = len(matched_refs)
    total_ref = len(ref_ids)

    return num_matched, total_ref    
    
def compute_iou_per_matched_pair(ref_labeled, pred_labeled, min_iou_threshold=0.5):
    ref_ids = np.unique(ref_labeled)
    ref_ids = ref_ids[ref_ids != 0]  # remove background
    matched_pairs = []
    iou_scores = []

    for ref_id in ref_ids:
        ref_mask = (ref_labeled == ref_id)
        pred_ids = np.unique(pred_labeled[ref_mask])
        pred_ids = pred_ids[pred_ids != 0]  # ignore background

        for pred_id in pred_ids:
            pred_mask = (pred_labeled == pred_id)
            intersection = np.logical_and(ref_mask, pred_mask).sum()
            union = np.logical_or(ref_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0

            matched_pairs.append((ref_id, pred_id))
            iou_scores.append(iou)

    return matched_pairs, iou_scores    

def evaluate_boundary_quality(pred_boundary, ref_boundary, pixel_size=10):
    # Ensure binary
    pred_binary = (pred_boundary == 1).astype(np.uint8)
    ref_binary = (ref_boundary == 1).astype(np.uint8)

    # Distance from predicted boundary pixels to reference boundary
    dist_map = distance_transform_cdt(1 - ref_binary, metric='chessboard')
    pred_coords = np.argwhere(pred_binary)

    distances = dist_map[pred_coords[:, 0], pred_coords[:, 1]] * pixel_size
    avg_distance = np.mean(distances) if len(distances) > 0 else np.nan

    pct_within_10 = np.mean(distances <= 10) * 100 if len(distances) > 0 else 0
    pct_within_20 = np.mean(distances <= 20) * 100 if len(distances) > 0 else 0

    ref_labeled, _ = label(ref_binary)
    pred_labeled, _ = label(pred_binary)


    num_matched, total_ref = count_matched_objects(ref_labeled, pred_labeled, min_iou_threshold=0.5)



    return {
        "avg_distance_m": avg_distance,
        "pct_within_10m": pct_within_10,
        "pct_within_20m": pct_within_20,
        "total_ref_objects": total_ref,
        "num_matched_objects": num_matched
    }





class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: float or list  class weights: [non-boundary, boundary]
        gamma: focusing parameter for hard examples
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
        else:
            self.alpha = None  # No class weighting
            
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, 2, H, W)  raw model outputs
        targets: (B, H, W)    ground truth labels (0 or 1)
        """
        B, C, H, W = logits.shape
        # Cross entropy loss per pixel (no reduction)
        ce_loss = N.cross_entropy(
            logits, targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            reduction='none'  # shape: (B, H, W)
        )

        # Get the probability for the target class: pt = softmax(logits)[target]
        probs = N.softmax(logits, dim=1)  # (B, 2, H, W)
        targets_one_hot = N.one_hot(targets, num_classes=2)  # (B, H, W, 2)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, 2, H, W)
        pt = (probs * targets_one_hot).sum(1)  # (B, H, W)

        # Focal Loss: (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss  # (B, H, W)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
"""
def accuracy_compare(prediction_folder, true_mask_folder):
    ''' Output average accuracy of all prediction results and their corresponding true masks.
    Args
        prediction_folder : folder of the prediction results
        true_mask_folder : folder of the corresponding true masks
    Returns
        a tuple of (original_accuracy, posprocess_accuracy)
    '''

    # Bring in the images
    all_prediction = gl.glob(prediction_folder)
    all_mask = gl.glob(true_mask_folder)

    # Initiation
    num_files = len(all_prediction)
    count = 0
    postprocess_acc = 0
    original_acc = 0

    while count != num_files:

        # Prepare the arrays to be further processed.
        prediction_processed = postprocess(all_prediction[count])
        prediction_image = Image.open(all_prediction[count])
        mask = Image.open(all_mask[count])

        # converting the PIL variables into numpy array
        prediction_np = np.asarray(prediction_image)
        mask_np = np.asarray(mask)

        # Calculate the accuracy of original and postprocessed image
        postprocess_acc += accuracy_check(mask_np, prediction_processed)
        original_acc += accuracy_check(mask_np, prediction_np)
        # check individual accuracy
        print(str(count) + 'th post acc:', accuracy_check(mask_np, prediction_processed))
        print(str(count) + 'th original acc:', accuracy_check(mask_np, prediction_np))

        # Move onto the next prediction/mask image
        count += 1

    # Average of all the accuracies
    postprocess_acc = postprocess_acc / num_files
    original_acc = original_acc / num_files

    return (original_acc, postprocess_acc)
"""

# Experimenting
if __name__ == '__main__':
    '''
    predictions = 'result/*.png'
    masks = '../data/val/masks/*.png'

    result = accuracy_compare(predictions, masks)
    print('Original Result :', result[0])
    print('Postprocess result :', result[1])
    '''
