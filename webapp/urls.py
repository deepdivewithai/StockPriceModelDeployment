from django.urls import path
from . import views

app_name = 'webapp'

urlpatterns = [
    path('', views.home, name='home'),
    path('predictor', views.predictor, name='predictor')
]
