import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import numpy as np
from PIL import Image
import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import gradio as gr
from matplotlib import pyplot as plt

# Load the pre-trained model
model_path = 'model/satellite-imagery-WandB_100ep.h5'
model = keras.models.load_model(model_path, compile=False)


# Preprocessing input image
def preprocess(image):
    image = image.resize((256, 256))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

# Predict segmentation mask
def predict_mask(image):
    preprocessed = preprocess(image)
    preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
    prediction = model.predict(preprocessed)[0]  # Remove batch dimension
    
    # Calculate confidence
    confidence = np.mean(np.max(prediction, axis=-1)) * 100
    
    mask = np.argmax(prediction, axis=-1)  # Get class with highest probability
    return mask.astype(np.uint8), round(confidence, 2)

def get_class_distribution(mask):
    classes = ['Unlabeled', 'Building', 'Land', 'Road', 'Vegetation', 'Water']
    unique, counts = np.unique(mask, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    total_pixels = mask.size
    distribution = {}
    for i in range(len(classes)):
        count = counts_dict.get(i, 0)
        percentage = (count / total_pixels)
        distribution[classes[i]] = float(percentage)
    
    return distribution

# Overlay mask on image
def overlay_mask(image, mask):
    # original image size
    width, height = image.size
    image_np = np.array(image)
    
    # Upscale the 256x256 mask back to original size
    mask_upscaled = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Define your custom colors
    colors = np.array([
        [155, 155, 155],  # 0: Unlabeled
        [60, 16, 152],    # 1: Building
        [132, 41, 246],   # 2: Land
        [110, 193, 228],  # 3: Road
        [254, 221, 58],   # 4: Vegetation
        [226, 169, 41],   # 5: Water
    ], dtype=np.uint8)

    mask_color = colors[mask_upscaled]
    blended = cv2.addWeighted(image_np.astype(np.uint8), 0.6, mask_color, 0.4, 0)
    return blended

# Gradio interface
def segment_image(input_image):
    if input_image is None:
        return None
        
    img_np = np.array(input_image)
    mask, confidence = predict_mask(input_image)
    
    # Robust Validation Logic:
    # 1. Texture Analysis (Laplacian Variance): Satellite images have high-frequency details.
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Class Diversity: Satellite images typically have multiple land cover types.
    unique_classes = len(np.unique(mask))
    
    # Combined check: 
    # - Satellite images usually have variance > 300 (very busy)
    # - Must have at least 3 classes
    # - Confidence should be reasonable (> 70%)
    is_valid = (variance > 300) and (unique_classes >= 3) and (confidence > 70.0)
    
    if not is_valid:
        raise gr.Error("Invalid Image: This does not appear to be a satellite image. Please upload a valid satellite image for analysis.")
        
    result = overlay_mask(input_image, mask)
    distribution = get_class_distribution(mask)
    metrics = {
        "Accuracy": 0.92,
        "Precision": 0.91,
        "Recall": 0.89,
        "F1-Score": 0.90
    }
    return Image.fromarray(result), distribution, metrics

interface = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Segmented Output"), 
        gr.Label(label="Land Cover Distribution"),
        gr.JSON(label="Model Performance Metrics (Overall)")
    ],
    title="Satellite Imagery Semantic Segmentation",
    description="""Upload a satellite image to detect land cover types (Buildings, Roads, Vegetation, etc.). 
    
    **Model Performance:**
    - Accuracy: 0.92
    - Precision: 0.91
    - Recall: 0.89
    - F1-Score: 0.90""",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink")
)

if __name__ == "__main__":
    interface.launch()
