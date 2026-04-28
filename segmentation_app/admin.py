from django.contrib import admin

# Register your models here.
from .models import UserRegistrationModel
list_display = ('name', 'loginid', 'password', 'mobile', 'email', 'locality', 'address', 'city', 'state', 'status')
admin.site.register(UserRegistrationModel)