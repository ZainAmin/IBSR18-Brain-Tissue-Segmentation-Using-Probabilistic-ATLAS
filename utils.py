import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from typing import List
from scipy.spatial.distance import directed_hausdorff

def mutual_information(vol1: np.ndarray, vol2: np.ndarray):
    """Computes the mutual information between two images/volumes
    Args:
        vol1 (np.ndarray): First of two image/volumes to compare
        vol2 (np.ndarray): Second of two image/volumes to compare
    Returns:
        (float): Mutual information
    """
    # Get the histogram
    hist_2d, x_edges, y_edges = np.histogram2d(
        vol1.ravel(), vol2.ravel(), bins=255)
    # Get pdf
    pxy = hist_2d / float(np.sum(hist_2d))
    # Marginal pdf for x over y
    px = np.sum(pxy, axis=1)
    # Marginal pdf for y over x
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def min_max_norm(img: np.ndarray, max_val: int = None, dtype: str = None):
    """
    Scales images to be in range [0, 2**bits]

    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.
        dtype (str, optional): Output datatype

    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if max_val is None:
        max_val = np.iinfo(img.dtype).max
    img = (img - img.min()) / (img.max() - img.min()) * max_val
    if dtype is not None:
        return img.astype(dtype)
    else:
        return img


def save_segementations(
    volume: np.ndarray, reference: sitk.Image, filepath: Path
):
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray): Volume to store as in Nifty format
        reference (sitk.Image): Reference image to get the spatial parameters from.
        filepath (Path): Where to save the volume.
    """
    # Save image
    img = sitk.GetImageFromArray(volume)
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    sitk.WriteImage(img, str(filepath))


def complete_figure(data_path: Path, img_names: List[Path], titles: List[str]):
    """Plots a huge figure with all image names from all modalities.
    Args:
        data_path (Path): Common path among the images
        img_names (List[Path]): File paths to read the images from
        titles (List[str]): Modality
    """
    img = sitk.ReadImage(str(data_path/img_names[0]))
    img_array = sitk.GetArrayFromImage(img)
    img_array = img_array[::-1, ::-1, ::-1]

    fig, ax = plt.subplots(7, len(img_names), figsize=(10, 14))
    for i, slice_n in enumerate(np.linspace(70, (img_array.shape[0] - 70), 7).astype('int')):
        for j, img_name in enumerate(img_names):
            img = sitk.ReadImage(str(data_path/img_name))
            img_array = sitk.GetArrayFromImage(img)
            if i == 6:
                ax[i, j].set_xlabel(titles[j])
            if i == 0:
                ax[i, j].set_title(titles[j])
            if j in [0, 1]:
                ax[i, j].imshow(img_array[slice_n, :, :], cmap='gray')
            else:
                ax[i, j].imshow(img_array[slice_n, :, :])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if j == 0:
                ax[i, j].set_ylabel(f'Slice {slice_n}')
    plt.show()

def dice_score(groundtruth_image, segmented_image):
    """
    Compute the Dice score for each label in the images and return them in a dictionary.
    """
    labels = [0, 1, 2, 3]  # Labels for Background, CSF, GM, and WM
    scores = {}

    for label in labels:
        gt = groundtruth_image == label
        seg = segmented_image == label

        intersection = np.sum(gt & seg)
        size_gt = np.sum(gt)
        size_seg = np.sum(seg)

        if size_gt + size_seg == 0:
            scores[label] = 1
        else:
            scores[label] = 2.0 * intersection / (size_gt + size_seg)

    return scores



def hausdorff_distance(ground_truth, segmentation):
    """
    Calculate the Hausdorff Distance for each label in 3D binary images, including the background.

    Parameters:
    ground_truth (np.ndarray): Ground truth 3D binary image.
    segmentation (np.ndarray): Segmented 3D binary image.

    Returns:
    dict: Hausdorff Distances for each label.
    """
    labels = np.unique(ground_truth)  # Include all labels, including background
    distances = {}

    for label in labels:
        max_hd = 0
        for slice_idx in range(ground_truth.shape[0]):
            seg1_slice = (ground_truth[slice_idx] == label).astype(int)
            seg2_slice = (segmentation[slice_idx] == label).astype(int)

            if np.any(seg1_slice) and np.any(seg2_slice):  # Proceed if label is present in the slice
                hd1 = directed_hausdorff(seg1_slice, seg2_slice)[0]
                hd2 = directed_hausdorff(seg2_slice, seg1_slice)[0]
                max_hd = max(max_hd, max(hd1, hd2))

        distances[f'Label_{label}'] = max_hd

    return distances



def volumetric_difference(groundtruth_image, segmented_image):
    groundtruth_image = np.array(groundtruth_image)
    segmented_image   = np.array(segmented_image)

    if groundtruth_image.shape != segmented_image.shape:
        raise ValueError("Segmented and ground truth images must have the same shape")

    avd_scores = {}
    for label in [0, 1, 2, 3]:  # Labels for Background, CSF, GM, WM
        segmented_volume = np.sum(segmented_image == label)
        groundtruth_volume = np.sum(groundtruth_image == label)

        if groundtruth_volume > 0:
            avd_score = abs(segmented_volume - groundtruth_volume) / groundtruth_volume * 100
        else:
            avd_score = 0

        avd_scores[label] = avd_score

    return avd_scores




def save_atlas(volume: np.ndarray, reference: sitk.Image, filepath: Path):
    # Save image
    img = sitk.GetImageFromArray(volume)
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    sitk.WriteImage(img, str(filepath))

    
def segmentation_tissue_model(test_image, test_mask, tissue_model):   
    
    masked_test_image_data = test_image[test_mask == 1].flatten()
    n_classes              = tissue_model.shape[0]
    preds                  = np.zeros((n_classes, len(masked_test_image_data)))

    for c in range(n_classes):
        preds[c, :] = tissue_model[c, masked_test_image_data]

    preds = np.argmax(preds, axis=0)
    predictions = test_mask.flatten()
    predictions[predictions == 1] = preds + 1
    segmented_test_image = predictions.reshape(test_image.shape)
    
    return segmented_test_image