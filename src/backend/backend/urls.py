"""
Phase 10/11 — project URL configuration.
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),  # login/logout
    path("api/auth/login/", obtain_auth_token, name="api-login"),
    path("api/", include("fraud_api.urls")),
    path("dashboard/", include("dashboard.urls")),
    path("", RedirectView.as_view(url="dashboard/", permanent=False)),
]