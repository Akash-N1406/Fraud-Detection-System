"""
Phase 11 — dashboard page routing.
"""
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="dashboard-home"),
    path("transactions/", views.transactions, name="dashboard-transactions"),
    path("alerts/", views.alerts, name="dashboard-alerts"),
    path("models/", views.models_page, name="dashboard-models"),
]