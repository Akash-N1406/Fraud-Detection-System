"""
Phase 10 — URL routing for the fraud_api app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r"transactions", views.TransactionViewSet, basename="transaction")
router.register(r"predictions", views.PredictionViewSet, basename="prediction")
router.register(r"alerts", views.FraudAlertViewSet, basename="alert")
router.register(r"model-metrics", views.ModelMetricViewSet, basename="model-metric")

urlpatterns = [
    path("", include(router.urls)),
    path("dashboard/kpis/", views.DashboardKPIView.as_view(), name="dashboard-kpis"),
]