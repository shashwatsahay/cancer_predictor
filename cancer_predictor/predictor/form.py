from django import forms
from predictor.models import UserInput
class UploadForm(forms.ModelForm):
    file_name = forms.FileField(required=True, label='Browse File*')
    class Meta:
        model = UserInput
        fields = ('file_name',)
