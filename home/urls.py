from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from home import views

urlpatterns = [
    path('',views.index,name='index'),
    path('about',views.about,name='about'),
    path('getstarted/',views.getstarted,name='getstarted'),
    path('getstarted/result/', views.result,name='result')
    ]
urlpatterns += staticfiles_urlpatterns()