from django import forms

class SentimentPredictionForm(forms.Form):
    review = forms.CharField(label='Review')
