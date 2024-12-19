from django.urls import path
from .views import register,login_view,dashboard,videocall

urlpatterns = [
    path('register/',register, name='register'),
    path("login/",login_view, name='login'),
    path("dashboard/",dashboard,name='dashboard'),
    path("videocall/",videocall,name='videocall')

]