from django.shortcuts import render, HttpResponse
from django.contrib import messages
from segmentation_app.models import UserRegistrationModel

from django.shortcuts import render


def index(request):
    return render(request, 'index.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})

