"""
Phase 10 — DRF serializers for the four Phase 9 tables.
"""
from rest_framework import serializers
from .models import Transaction, Prediction, FraudAlert, ModelMetric


class TransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transaction
        fields = [
            "transaction_id", "step", "type", "amount", "name_orig",
            "old_balance_org", "new_balance_orig", "name_dest",
            "old_balance_dest", "new_balance_dest", "is_fraud_actual",
            "received_at",
        ]


class PredictionSerializer(serializers.ModelSerializer):
    transaction_id = serializers.CharField(source="transaction.transaction_id", read_only=True)

    class Meta:
        model = Prediction
        fields = [
            "id", "transaction_id", "fraud_probability", "risk_level",
            "model_version", "latency_ms", "prediction_time",
        ]


class FraudAlertSerializer(serializers.ModelSerializer):
    transaction_id = serializers.CharField(source="transaction.transaction_id", read_only=True)
    amount = serializers.DecimalField(
        source="transaction.amount", max_digits=18, decimal_places=2, read_only=True
    )

    class Meta:
        model = FraudAlert
        fields = [
            "id", "transaction_id", "amount", "alert_level",
            "fraud_probability", "status", "created_at", "updated_at",
        ]


class ModelMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetric
        fields = [
            "id", "model_name", "accuracy", "precision_score",
            "recall_score", "f1_score", "roc_auc", "pr_auc", "training_date",
        ]