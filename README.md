Task 1 – Low-Light Image Enhancement
Problem Statement
Low-light images often suffer from poor visibility, low contrast, color distortion, and loss of structural details. These issues negatively affect both human perception and downstream vision tasks such as object detection and tracking.
The objective of Task-1 is to enhance low-light images while preserving structural fidelity and improving perceptual quality.

Our Approach
We propose a CNN–Transformer hybrid architecture designed specifically for stable and metric-aware low-light image enhancement.
The core idea is to:
Use CNNs to preserve local structure, edges, and textures
Use a Transformer only at the bottleneck to model global illumination and color consistency
By restricting the Transformer to the lowest spatial resolution, we avoid hallucinations and maintain pixel alignment, which is critical for quantitative metrics.

Model Architecture
The network follows a U-Net style encoder–decoder structure:
CNN Encoder
Gradually downsamples the input image
Extracts hierarchical spatial and semantic features
Preserves edges and textures
Transformer Bottleneck
Operates on low-resolution feature maps
Captures global context such as illumination and color distribution
Does not directly manipulate pixels
CNN Decoder
Reconstructs the enhanced image using skip connections
Preserves spatial structure and fine details
The final output is a normalized RGB image.

Loss Function

A composite loss is used to balance fidelity, structure, perception, and color consistency:
L1 Loss – pixel-level accuracy
SSIM Loss – structural preservation
Perceptual Loss (VGG-based) – perceptual similarity
Color Consistency Loss – corrects color shifts

Final loss formulation:
0.8 × L1 + 0.1 × SSIM + 0.05 × Perceptual + 0.05 × Color

This design ensures stable training without adversarial components.

Training Strategy :
Deterministic train/validation split
High-performance tf.data pipeline
Adam optimizer with adaptive learning rate scheduling
Early stopping to prevent overfitting
The training pipeline is fully modular and reproducible.

Evaluation Metrics

Task-1 performance is evaluated using multiple complementary metrics:
PSNR – pixel fidelity
SSIM – structural similarity
LPIPS – perceptual similarity
UCIQE – colorfulness and contrast

Using multiple metrics ensures balanced enhancement rather than over-optimization of a single criterion.

Key Design Choices
No adversarial training for stabilit
Transformer limited to bottleneck to avoid hallucination
Moderate parameter count to reduce overfitting

Clean modular code structure for reproducibility

This Task-1 solution focuses on stable, perceptually consistent low-light image enhancement using a carefully designed CNN–Transformer architecture.
The approach improves visibility and color quality while preserving structural details, making it suitable for both quantitative evaluation and real-world usage.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
