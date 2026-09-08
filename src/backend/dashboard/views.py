"""
Phase 11 — dashboard template views.
Session-authenticated (Django's normal login), so a person logs in once
and both the dashboard pages and the JS fetch() calls to the DRF API
(which also accepts SessionAuthentication, per Phase 10's settings) work
off the same login.
"""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def home(request):
    return render(request, "dashboard/home.html")


@login_required
def transactions(request):
    return render(request, "dashboard/transactions.html")


@login_required
def alerts(request):
    return render(request, "dashboard/alerts.html")


@login_required
def models_page(request):
    return render(request, "dashboard/models.html")