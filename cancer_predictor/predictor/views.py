from django.shortcuts import render, redirect
from django.http import HttpResponse

from cancer_predictor.settings import MEDIA_ROOT, FILTER_LABEL, SVC_MODEL, SCALER_MODEL
from predictor.models import UserInput, OUTPUT
from predictor.form import UploadForm
from cancer_predictor.settings import CLASS_ID
import json
import numpy as np
# Create your views here.
def home(request):
    return render(request, 'predictor/home.html', {})

def documentation(request):
    return render(request, 'predictor/documentation.html', {})

	
def predictor(request):
	form = UploadForm()
	if request.method=='POST':
		form=UploadForm(request.POST, request.FILES)
		if form.is_valid():
			userin=form.save()
			userin.save()
			return_label=True
			return redirect('/predictor?id='+userin.getid())
	return render(request, 'predictor/predictor.html', {'Upload_Form':form})


		
def predict_asjson(request):
	to_predict=request.GET.get('id', 1)
	qs=UserInput.objects.get(id=to_predict)
	return_label=class_predict(qs.getfilename())
	json_data=json.dumps(return_label)
	return HttpResponse({json_data}, content_type='application/json')



	
def class_predict(filename):
	handle=open(filename)
	gene_labels=handle.readline().strip().split(",")
	indexes=list()
	sample_data=list()
	sample_id=list()
	for label in FILTER_LABEL:
		indexes.append(gene_labels.index(label))
	for line in handle:
		line=line.strip().split(',')
		sample_id.append(line[0])
		line=[float(line[int(index)]) for index in indexes]
		sample_data.append(line)
		#print(','.join(line))
	sample_data=np.array(sample_data)
	sample_data=SCALER_MODEL.fit_transform(sample_data)
	y_pred=SVC_MODEL.predict(sample_data)
	return_labels=list()
	for i in range(0, len(sample_id)):
		out=dict()
		out["sample"]=sample_id[i]
		out["label"]=CLASS_ID[str(y_pred[i])]
		return_labels.append(out)
	return return_labels
