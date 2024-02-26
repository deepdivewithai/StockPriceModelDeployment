from django import forms

class StockForm(forms.Form):
    stock_name = forms.CharField(max_length=200)
    date_for_prediction = forms.DateField()