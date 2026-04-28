import os
from django.conf import settings
from tensorflow.keras.models import load_model

# Assuming the model file is in the 'model' directory
try:
    model = load_model(os.path.join('model', 'satellite-imagery-WandB_100ep.h5'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
