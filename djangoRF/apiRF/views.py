from django.shortcuts import render

# Creamos una vista para renderizar el formulario HTML
def formulario(request):
    return render(request, 'index.html')