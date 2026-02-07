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
# YOLOv8 + ByteTrack Ground Truth Generator

This repository provides a **robust and reproducible pipeline** for generating **frame-wise ground truth bounding boxes** from video sequences using **YOLOv8 for object detection** and **ByteTrack for multi-object tracking**.

The system processes videos stored as image frames and outputs a **single `ground_truth.txt` file per video**, containing normalized bounding box coordinates for all detected objects across frames.

---

## 📁 Project Structure

### Input (ZIP upload compatible with Google Colab)

data/
├── Video_0001/
│ └── imgs/
│ ├── 0001.jpg
│ ├── 0002.jpg
│ └── ...
├── Video_0002/
│ └── imgs/
│ ├── 0001.jpg
│ └── ...


### Output

output/
├── Video_0001/
│ └── ground_truth.txt
├── Video_0002/
│ └── ground_truth.txt


---

## 📄 Ground Truth File Format

Each `ground_truth.txt` contains bounding box data for **all frames** in the corresponding video.

### Format


- `frame_id` : Frame index (derived from image filename)
- `x_center` : Normalized x-coordinate of bounding box center
- `y_center` : Normalized y-coordinate of bounding box center
- `width`    : Normalized bounding box width
- `height`   : Normalized bounding box height

All values are normalized to `[0, 1]`.

### Example

0001 0.482959 0.350738 0.067204 0.222873
0002 0.483381 0.375187 0.079682 0.264794
0048 0.456942 0.570568 0.084394 0.237360
0048 0.525873 0.521943 0.030812 0.047081

> Multiple entries with the same `frame_id` indicate **multiple objects detected in that frame**.

---

## ⚙️ Methodology

### Object Detection
- **YOLOv8** is used for real-time object detection.
- Anchor-free architecture ensures stable bounding boxes.
- Confidence and IoU thresholds are applied to remove noisy detections.

### Object Tracking
- **ByteTrack** associates detections across frames.
- Uses Kalman filtering and IoU-based matching.
- Retains low-confidence detections to improve tracking continuity.

### Bounding Box Normalization
Pixel coordinates are converted into normalized form:

x_center = x_pixel / image_width
y_center = y_pixel / image_height
width = box_width / image_width
height = box_height / image_height


This makes the output resolution-independent and dataset-agnostic.

---

## 🚀 Why YOLOv8 + ByteTrack?

While more complex models exist, YOLOv8 + ByteTrack was chosen because it offers:

- High detection accuracy
- Stable bounding boxes across frames
- Low ID-switch rate
- Computational efficiency
- Reproducibility and ease of deployment

The goal of this project is **reliable ground truth generation**, not benchmark optimization.

---

## 🧪 Applications

- Ground truth generation for tracking datasets
- Motion and trajectory analysis
- Conversion to MOT / COCO formats
- Evaluation of tracking algorithms
- Research and academic projects

---

## 🛠 Requirements

- Python 3.8+
- ultralytics
- OpenCV

Install dependencies:

```bash
pip install ultralytics opencv-python
