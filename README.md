# Stage 1: Underwater Image Enhancement

## Objective
Enhance low-light underwater images by correcting color distortion,
reducing haze, and improving contrast while preserving structural details.

## Model
- Architecture: Supervised U-Net (Encoder–Decoder with skip connections)
- Input: RGB underwater image
- Output: Enhanced RGB image

## Loss Function
Total Loss = L1 Loss + 0.5 × (1 − SSIM)

## Dataset
- Training images: 5000
- Validation images: 599
- Paired dataset (input → ground truth)

## Evaluation Metrics
- PSNR
- SSIM
- LPIPS
- UCIQE

## Result
The model is trained to beat the baseline metrics provided in the problem
statement and produce visually and quantitatively improved images.

## Folder Structure
- data/: Training and validation datasets
- models/: Saved model checkpoints
- outputs/: Enhanced images
- src/: Source code for model and training
