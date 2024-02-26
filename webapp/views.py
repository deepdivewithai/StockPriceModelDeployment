from django.shortcuts import render, redirect
from keras.models import load_model
import joblib

import pandas as pd
import numpy as np
import yfinance as yf
from .forms import StockForm
import datetime as dt

model = load_model("model/stocksprediction.keras")
scaler = joblib.load("model/scaler.pkl")

# Create your views here.
def home(request):
    return render(request, 'webapp/index.html')

def predictor(request):
    form = StockForm()

    if request.method == "POST":
        stock_name = request.POST["stock_name"]
        date_for_prediction = request.POST['date_for_prediction']

        form = StockForm(request.POST)

        if form.is_valid():
            print(form.cleaned_data)

            tickers = {'Google':"GOOGL",
                       'Apple': "AAPL"}
            
            # Get the quote
            stock_quote = yf.download(tickers= tickers[stock_name], start='2012-01-01', end=date_for_prediction)
            new_df = stock_quote.filter(["Close"])
            last_60_days = new_df[-60:].values
            last_60_days_scaled = scaler.transform(last_60_days)
            X_test = []
            X_test.append(last_60_days_scaled)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            pred_price = model.predict(X_test)
            pred_price = scaler.inverse_transform(pred_price)
            
            start_date = dt.datetime.strptime(date_for_prediction, '%Y-%m-%d')
            end_date = start_date + dt.timedelta(days=1)
            end_date = end_date.strftime('%Y-%m-%d')

            print(end_date)
            actual_price = yf.download(tickers= tickers[stock_name], start=date_for_prediction, end=end_date)
            actual_price = actual_price["Close"].values

            context = {
                'stock': stock_name,
                'actual_price': actual_price[0],
                'prediction': pred_price[0][0],
                'date': date_for_prediction
                }

            return render(request, "webapp/result.html", context)
        
    context = {
        'form': form,
    }
        
    return render(request, "webapp/stocks.html", context)
