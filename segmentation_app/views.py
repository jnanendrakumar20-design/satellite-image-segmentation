import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
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
import segmentation_models as sm

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

def get_class_distribution(mask):
    classes = ['Unlabeled', 'Building', 'Land', 'Road', 'Vegetation', 'Water']
    unique, counts = np.unique(mask, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    total_pixels = mask.size
    distribution = []
    for i in range(len(classes)):
        count = counts_dict.get(i, 0)
        percentage = (count / total_pixels) * 100
        distribution.append(round(percentage, 2))
    
    return distribution

def overlay_mask(image, mask):
    # original image size
    width, height = image.size
    image_np = np.array(image)
    
    # Upscale the 256x256 mask back to original size
    mask_upscaled = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    colors = np.array([
        [155, 155, 155],  # Unlabeled
        [60, 16, 152],    # Building
        [132, 41, 246],   # Land
        [110, 193, 228],  # Road
        [254, 221, 58],   # Vegetation
        [226, 169, 41],   # Water
    ], dtype=np.uint8)
    
    mask_color = colors[mask_upscaled]
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
                img_np = np.array(img)
                mask, confidence = predict_mask(img)
                
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
                    return render(request, 'users/index1.html', {
                        'error': 'Invalid Image: This does not appear to be a satellite image. Please upload a valid satellite image.'
                    })

                result = overlay_mask(img, mask)
                distribution = get_class_distribution(mask)

                output_path = os.path.join(settings.MEDIA_ROOT, 'result.png')
                result.save(output_path)

                return render(request, 'users/detection.html', {
                    'result_image': os.path.join(settings.MEDIA_URL, 'result.png'),
                    'confidence': confidence,
                    'distribution': distribution
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

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from io import BytesIO

@csrf_exempt
def api_predict(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            try:
                img_file = request.FILES['image']
                img = Image.open(img_file).convert("RGB")
                img_np = np.array(img)
                
                # Perform prediction
                mask, confidence = predict_mask(img)
                
                # Overlay mask
                result_img = overlay_mask(img, mask)
                
                # Get distribution
                distribution = get_class_distribution(mask)
                
                # Convert result image to base64
                buffered = BytesIO()
                result_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return JsonResponse({
                    'success': True,
                    'confidence': confidence,
                    'distribution': distribution,
                    'result_image': f"data:image/png;base64,{img_str}",
                    'classes': ['Unlabeled', 'Building', 'Land', 'Road', 'Vegetation', 'Water']
                })
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)
        return JsonResponse({'success': False, 'error': 'No image uploaded'}, status=400)
    return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
