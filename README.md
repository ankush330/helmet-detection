Helmet Detection
A binary image classification model that detects whether a person is wearing a helmet or not, built using Transfer Learning on MobileNetV2.

 Project Pipeline
- **Data**: 764 images from Kaggle (andrewmvd/helmet-detection)
- **Preprocessing**: Extracted 1254 head crops from XML annotations (PASCAL VOC format)
- **Model**: MobileNetV2 (Transfer Learning, pretrained on ImageNet)
- **Training**: 10 epochs, NVIDIA RTX 2050 GPU
- **Result**: 88% accuracy on validation set

 Results
| Metric | Score |
|---|---|
| Overall Accuracy | 88% |
| Helmet F1 Score | 0.91 |
| No Helmet F1 Score | 0.79 |
| Total Crops | 1254 (1006 train / 248 val) |

## How to Run

### 1. Clone the repo
git clone https://github.com/ankush330/helmet-detection.git

### 2. Install dependencies
pip install torch torchvision streamlit pillow scikit-learn

### 3. Run the app
streamlit run app.py
## Tech Stack
- Python
- PyTorch + TorchVision
- MobileNetV2
- Streamlit
- scikit-learn
- OpenCV

## Dataset
- Source: [Kaggle - Helmet Detection by andrewmvd](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
- 764 images with PASCAL VOC XML annotations
- 2 classes: `With Helmet` / `Without Helmet`

## Limitations
- Works best on cropped head images, struggles with full scene images
- When multiple people are in the frame, gives only ONE prediction for the whole image — cannot detect helmet/no_helmet per person separately
- Does not distinguish between helmets and other headwear (caps, hats, turbans)
- Live webcam detection is inaccurate because the model was trained on cropped head regions, not full video frames — real-time detection requires a two-stage pipeline (head detector + classifier)

## Future Improvements
- Integrate a head/face detector (e.g. YOLO) to crop each person's head first, then classify — this would fix both multi-person and live video limitations
- Collect more no_helmet samples to fix class imbalance (698 helmet vs 308 no_helmet)
- Add a third class for other headwear (caps, hats)
- Train on full scene images for better real-world performance
- Deploy on Streamlit Cloud for public access
