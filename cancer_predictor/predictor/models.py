from django.db import models
import uuid, datetime, os
from cancer_predictor.settings import MEDIA_ROOT, CLASS_ID

# Create your models here.

class UserInput(models.Model):
	id = models.CharField(max_length=100, primary_key=True, default=uuid.uuid4)
	def k(self, filename):
		year, month, day = str(datetime.date.today()).split('-')
		returnpath= MEDIA_ROOT +'/' + year + '/' + day + '_' + month + '/' + str(self.id) + '.csv'
		return returnpath
	file_name = models.FileField(upload_to=k)
	processed = models.BooleanField(default=False)
	
	def getfilename(self):
		return str(self.file_name)
		
	def getid(self):
		return str(self.id)

class OUTPUT(models.Model):
	sample=models.CharField(max_length=100)
	label=models.CharField(max_length=100)
	def __init__(self, sample, label):
		self.sample=sample
		self.label=CLASS_ID[str(label)]
