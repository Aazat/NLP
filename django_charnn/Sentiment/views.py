from django.shortcuts import render
from . import engine


def home(request):
    p = 0
    r = "Nothing received"
    if request.GET:
        a = request.GET["comment"]
        if (a == "" or a == " "):
            pass
        else:
            r, p = engine.get_label([a])
            r=r[0]
    return render(request, 'Sentiment/home.html', {'sample': r, 'pred_value' : p})