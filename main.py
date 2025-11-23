from waterpixels import *

# -----------------------------------------------------------
# Test images
# Lac ./BSDS300/images/train/176035.jpg ./BSDS300/human/gray/1130/176035.seg
# Loutre ./BSDS300/images/train/173036.jpg ./BSDS300/human/color/1107/173036.seg
# Cheval ./BSDS300/images/train/187003.jpg ./BSDS300/human/color/1105/187003.seg
# -----------------------------------------------------------

# Read image and display
im = read_image('./BSDS300/images/train/176035.jpg')
viewimage(im, gray=False)

# Compute waterpixels and display
labels = waterpixels(im, 8, k=0.5, gradient_method='lab', grid='hexagonal', markers='centers', distance_alg='chanfrein', watershed_alg='fast')
viewimage(labels, gray=True)

# Reconstruct the image of the human segmentation from the .seg file 
human_seg = open('./BSDS300/human/gray/1130/176035.seg', 'r')
gt = gt_borders(human_seg)
viewimage(gt, gray=True)

# Evaluate segmentation
eval = evaluate_waterpixels_measures(labels, gt)
print(f"BR: {eval[0]:.4f}, CD: {eval[1]:.4f}, MF: {eval[2]:.4f}")

# Display segmentation boundaries on top of original image with evaluation metrics
overlap = im.copy()
borders = segmentation_borders(labels)
viewimage(borders, gray=True)
overlap[borders == 1] = [255, 255, 255]
results = f"BR: {eval[0]:.4f}, CD: {eval[1]:.4f}, MF: {eval[2]:.4f}"
viewimage(overlap, titre=results)