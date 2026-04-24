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
    mask = np.argmax(prediction, axis=-1)  # Get class with highest probability
    return mask.astype(np.uint8)

# Overlay mask on image
def overlay_mask(image, mask):
    image = image.resize((256, 256))
    image_np = np.array(image)

    # Define your custom colors
    colors = np.array([
        [155, 155, 155],  # 0: Unlabeled
        [60, 16, 152],    # 1: Building
        [132, 41, 246],   # 2: Land
        [110, 193, 228],  # 3: Road
        [254, 221, 58],   # 4: Vegetation
        [226, 169, 41],   # 5: Water
    ], dtype=np.uint8)

    mask_color = colors[mask]
    blended = cv2.addWeighted(image_np.astype(np.uint8), 0.6, mask_color, 0.4, 0)
    return blended

# Gradio interface
def segment_image(input_image):
    mask = predict_mask(input_image)
    result = overlay_mask(input_image, mask)
    return Image.fromarray(result)

interface = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Semantic Segmentation (U-Net)",
    description="Upload an image and get semantic segmentation using a pretrained U-Net model."
)

if __name__ == "__main__":
    interface.launch()
