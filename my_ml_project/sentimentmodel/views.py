import os
import joblib
import spacy

from django.shortcuts import render
from .forms import SentimentPredictionForm

def predict_sentiment(request):
    if request.method == 'POST':
        form = SentimentPredictionForm(request.POST)
        if form.is_valid():
            # Load the trained model and vectorizer
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'sgdc_model.sav')
            model = joblib.load(model_path)

            vect_path = os.path.join(os.path.dirname(__file__), 'model', 'vectorizer.sav')
            vect = joblib.load(vect_path)

            # Extract input data from the form
            new_data = list(form.cleaned_data.values())

            # Perform prediction
            pred = model.predict(vect.transform(new_data))
            prediction = ""
            if pred == [0]:
                prediction = "Negative sentiment."
            else:
                prediction = "Positive sentiment."

            # Prepare the response
            context = {
                'form': form,
                'sentiment': prediction,
            }
            return render(request, 'index.html', context)
    else:
        form = SentimentPredictionForm()

    context = {'form': form}
    return render(request, 'index.html', context)
