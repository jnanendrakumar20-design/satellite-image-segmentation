import os
from django.conf import settings
from django.shortcuts import render, redirect
from .models import UserRegistrationModel
from django.contrib import messages

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html') 


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")





import os
import numpy as np
from PIL import Image
import cv2
from django.shortcuts import render
from django.conf import settings
from .forms import ImageUploadForm
from tensorflow import keras

# Load model
model_path = os.path.join(settings.BASE_DIR, 'segmentation_app',  'satellite-imagery-WandB_100ep.h5')

if not os.path.exists(model_path):
    print(f"❌ Model file not found at: {model_path}")
    model = None
else:
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def preprocess(image):
    image = image.resize((256, 256))
    return np.array(image) / 255.0

def predict_mask(image):
    if model is None:
        raise ValueError("Model not loaded. Cannot make prediction.")
    preprocessed = preprocess(image)
    preprocessed = np.expand_dims(preprocessed, axis=0)
    prediction = model.predict(preprocessed)[0]
    
    # Calculate confidence as the average of maximum probabilities across all pixels
    confidence = np.mean(np.max(prediction, axis=-1)) * 100
    
    mask = np.argmax(prediction, axis=-1).astype(np.uint8)
    return mask, round(confidence, 2)

def overlay_mask(image, mask):
    image = image.resize((256, 256))
    image_np = np.array(image)
    colors = np.array([
        [155, 155, 155],  # Unlabeled
        [60, 16, 152],    # Building
        [132, 41, 246],   # Land
        [110, 193, 228],  # Road
        [254, 221, 58],   # Vegetation
        [226, 169, 41],   # Water
    ], dtype=np.uint8)
    mask_color = colors[mask]
    blended = cv2.addWeighted(image_np.astype(np.uint8), 0.6, mask_color, 0.4, 0)
    return Image.fromarray(blended)
from django.shortcuts import render
from django.conf import settings
from PIL import Image
import os

def index1(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            try:
                img = Image.open(request.FILES['image']).convert("RGB")
                mask, confidence = predict_mask(img)
                result = overlay_mask(img, mask)

                output_path = os.path.join(settings.MEDIA_ROOT, 'result.png')
                result.save(output_path)

                return render(request, 'users/detection.html', {
                    'result_image': os.path.join(settings.MEDIA_URL, 'result.png'),
                    'confidence': confidence
                })
            except Exception as e:
                return render(request, 'users/index1.html', {
                    'error': f'Error: {str(e)}'
                })
        else:
            return render(request, 'users/index1.html', {
                'error': 'No image was uploaded.'
            })

    return render(request, 'users/index1.html')
