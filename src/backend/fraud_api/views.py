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


class FraudByTypeView(APIView):
    """GET /api/dashboard/fraud-by-type/ — total vs. fraud count per
    transaction type. Stands in for the SRS's "fraud by merchant category"
    chart, since PaySim has transaction type, not merchant category."""

    def get(self, request):
        data = (
            Transaction.objects.values("type")
            .annotate(
                total=Count("id"),
                fraud=Count("id", filter=Q(is_fraud_actual=True)),
            )
            .order_by("-total")
        )
        return Response(list(data))


class HourlyPatternView(APIView):
    """GET /api/dashboard/hourly-pattern/ — transaction and fraud counts by
    hour of day, derived from `step` (1 step = 1 simulated hour), matching
    the SRS's "hourly fraud patterns" chart."""

    def get(self, request):
        from django.db.models import F
        data = (
            Transaction.objects.annotate(hour=F("step") % 24)
            .values("hour")
            .annotate(
                total=Count("id"),
                fraud=Count("id", filter=Q(is_fraud_actual=True)),
            )
            .order_by("hour")
        )
        return Response(list(data))


class RiskDistributionView(APIView):
    """GET /api/dashboard/risk-distribution/ — prediction count per risk
    level, matching the SRS's "risk score distribution" chart."""

    def get(self, request):
        data = Prediction.objects.values("risk_level").annotate(count=Count("id"))
        return Response(list(data))


class AmountDistributionView(APIView):
    """GET /api/dashboard/amount-distribution/ — transaction count per
    amount bucket, matching the SRS's "transaction amount distribution"
    chart. Bucketed server-side rather than shipping every raw amount to
    the browser for client-side binning."""

    def get(self, request):
        from django.db.models import Case, When, Value, CharField
        buckets = (
            Transaction.objects.annotate(
                bucket=Case(
                    When(amount__lt=10000, then=Value("0-10K")),
                    When(amount__lt=100000, then=Value("10K-100K")),
                    When(amount__lt=500000, then=Value("100K-500K")),
                    When(amount__lt=1000000, then=Value("500K-1M")),
                    default=Value("1M+"),
                    output_field=CharField(),
                )
            )
            .values("bucket")
            .annotate(count=Count("id"))
        )
        # Fixed order so the chart's x-axis is always in ascending bucket
        # order rather than whatever order the DB happens to return
        order = ["0-10K", "10K-100K", "100K-500K", "500K-1M", "1M+"]
        by_bucket = {row["bucket"]: row["count"] for row in buckets}
        return Response([{"bucket": b, "count": by_bucket.get(b, 0)} for b in order])