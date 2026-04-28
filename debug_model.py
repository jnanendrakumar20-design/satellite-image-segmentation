import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

model_path = r'c:\Users\My world\Desktop\4 sem project\modified\code\segmentation_app\satellite-imagery-WandB_100ep.h5'

print(f"Checking if file exists: {os.path.exists(model_path)}")

try:
    model = keras.models.load_model(model_path, compile=False)
    print("SUCCESS: Model loaded successfully in test script")
except Exception as e:
    print(f"ERROR: Error loading model in test script: {e}")
