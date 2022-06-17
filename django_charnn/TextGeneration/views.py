from requests import request
from . import engine
# from .models import Post
from django.shortcuts import render

def home(request):
    return render(request, 'TextGeneration/about.html', {"title" : "Title"})

def test(request):
    finput = request.GET["finput"]
    n_chars = int(request.GET["n_chars"])
    result = engine.complete_text(finput)
    return render(request, 'TextGeneration/test.html', {'finput': finput, 'result':result})


# Create your views here.
# except ImportError:
#     print("Import Error Recieved")
