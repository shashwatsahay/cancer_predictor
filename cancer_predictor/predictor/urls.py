from django.urls import path
from django.conf.urls import url
from . import views
 
urlpatterns=[
	path('', views.home, name='home'),
	path('documentation', views.documentation, name='documentation'),
	url(r'^predictor/$', views.predictor, name='predictor'),
	path('predict_asjson/', views.predict_asjson, name='predict_asjson')	
]




