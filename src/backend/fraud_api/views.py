"""
Phase 10 — API views: read-only endpoints for transactions/predictions/
model metrics, a status-updatable endpoint for fraud alerts (analysts
triaging alerts is a real write operation, unlike the other tables which
are only ever written by the Kafka consumer), and a dashboard KPI
aggregate endpoint matching the SRS's Phase 11 dashboard requirements.
"""
from django.db.models import Avg, Count, Q
from rest_framework import viewsets, mixins
from rest_framework.views import APIView
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter

from .models import Transaction, Prediction, FraudAlert, ModelMetric
from .serializers import (
    TransactionSerializer, PredictionSerializer,
    FraudAlertSerializer, ModelMetricSerializer,
)


class TransactionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer
    filter_backends = [DjangoFilterBackend, SearchFilter]
    filterset_fields = ["type", "is_fraud_actual"]
    search_fields = ["transaction_id", "name_orig", "name_dest"]


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Prediction.objects.select_related("transaction").all()
    serializer_class = PredictionSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["risk_level"]


class FraudAlertViewSet(
    mixins.ListModelMixin, mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin, viewsets.GenericViewSet,
):
    """
    List/retrieve alerts, and allow analysts to update `status` as they
    triage (New -> Under Investigation -> Confirmed Fraud / False Positive
    -> Resolved, per the SRS's alert lifecycle). No create/delete — alerts
    are only ever created by the consumer when a transaction is flagged.
    """
    queryset = FraudAlert.objects.select_related("transaction").all()
    serializer_class = FraudAlertSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["status", "alert_level"]


class ModelMetricViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelMetric.objects.all()
    serializer_class = ModelMetricSerializer


class DashboardKPIView(APIView):
    """
    GET /api/dashboard/kpis/ — the summary numbers the SRS's dashboard
    page calls for: total transactions, fraud detected, detection rate,
    high-risk count, average transaction amount.
    """

    def get(self, request):
        total_transactions = Transaction.objects.count()
        total_fraud_alerts = FraudAlert.objects.count()
        total_actual_fraud = Transaction.objects.filter(is_fraud_actual=True).count()
        high_risk_count = Prediction.objects.filter(
            risk_level__in=["High", "Critical"]
        ).count()
        avg_amount = Transaction.objects.aggregate(avg=Avg("amount"))["avg"] or 0

        fraud_detection_rate = (
            total_fraud_alerts / total_transactions if total_transactions else 0
        )

        return Response({
            "total_transactions": total_transactions,
            "total_fraud_detected": total_fraud_alerts,
            "total_actual_fraud": total_actual_fraud,  # ground truth, demo/eval only
            "fraud_detection_rate": round(fraud_detection_rate, 6),
            "high_risk_transactions": high_risk_count,
            "average_transaction_amount": round(float(avg_amount), 2),
        })