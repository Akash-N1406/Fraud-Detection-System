"""
Phase 10 — Django models mapped to the existing Phase 9 PostgreSQL schema.

managed = False on every model: these tables already exist and are being
written to by kafka/consumer/transaction_consumer.py via raw SQL. Django
only reads and serves this data through the API — letting Django "manage"
these tables (i.e. run its own migrations against them) would risk
migration conflicts with a schema that already has live data flowing
into it. The consumer remains the single source of truth for writes.
"""
from django.db import models


class Transaction(models.Model):
    transaction_id = models.CharField(max_length=32, unique=True)
    step = models.IntegerField()
    type = models.CharField(max_length=20)
    amount = models.DecimalField(max_digits=18, decimal_places=2)
    name_orig = models.CharField(max_length=32, db_column="name_orig")
    old_balance_org = models.DecimalField(max_digits=18, decimal_places=2, db_column="old_balance_org")
    new_balance_orig = models.DecimalField(max_digits=18, decimal_places=2, db_column="new_balance_orig")
    name_dest = models.CharField(max_length=32, db_column="name_dest")
    old_balance_dest = models.DecimalField(max_digits=18, decimal_places=2, db_column="old_balance_dest")
    new_balance_dest = models.DecimalField(max_digits=18, decimal_places=2, db_column="new_balance_dest")
    is_fraud_actual = models.BooleanField(null=True, db_column="is_fraud_actual")
    received_at = models.DateTimeField(db_column="received_at")

    class Meta:
        managed = False
        db_table = "transactions"
        ordering = ["-received_at"]

    def __str__(self):
        return self.transaction_id


class Prediction(models.Model):
    transaction = models.ForeignKey(
        Transaction, on_delete=models.DO_NOTHING,
        to_field="transaction_id", db_column="transaction_id",
        related_name="predictions",
    )
    fraud_probability = models.DecimalField(max_digits=6, decimal_places=5, db_column="fraud_probability")
    risk_level = models.CharField(max_length=10, db_column="risk_level")
    model_version = models.CharField(max_length=50, db_column="model_version")
    latency_ms = models.DecimalField(max_digits=8, decimal_places=2, null=True, db_column="latency_ms")
    prediction_time = models.DateTimeField(db_column="prediction_time")

    class Meta:
        managed = False
        db_table = "predictions"
        ordering = ["-prediction_time"]

    def __str__(self):
        return f"{self.transaction_id} -> {self.risk_level}"


class FraudAlert(models.Model):
    STATUS_CHOICES = [
        ("New", "New"),
        ("Under Investigation", "Under Investigation"),
        ("Confirmed Fraud", "Confirmed Fraud"),
        ("False Positive", "False Positive"),
        ("Resolved", "Resolved"),
    ]

    transaction = models.ForeignKey(
        Transaction, on_delete=models.DO_NOTHING,
        to_field="transaction_id", db_column="transaction_id",
        related_name="alerts",
    )
    alert_level = models.CharField(max_length=10, db_column="alert_level")
    fraud_probability = models.DecimalField(max_digits=6, decimal_places=5, db_column="fraud_probability")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="New")
    created_at = models.DateTimeField(db_column="created_at")
    updated_at = models.DateTimeField(db_column="updated_at")

    class Meta:
        managed = False
        db_table = "fraud_alerts"
        ordering = ["-created_at"]

    def save(self, *args, **kwargs):
        # managed=False means Django won't apply any DEFAULT/trigger
        # behavior automatically — set updated_at explicitly on every save
        # so PATCH requests (analyst status changes) actually bump it.
        from django.utils import timezone
        self.updated_at = timezone.now()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Alert: {self.transaction_id} ({self.status})"


class ModelMetric(models.Model):
    model_name = models.CharField(max_length=50, db_column="model_name")
    accuracy = models.DecimalField(max_digits=6, decimal_places=5, null=True)
    precision_score = models.DecimalField(max_digits=6, decimal_places=5, null=True, db_column="precision_score")
    recall_score = models.DecimalField(max_digits=6, decimal_places=5, null=True, db_column="recall_score")
    f1_score = models.DecimalField(max_digits=6, decimal_places=5, null=True, db_column="f1_score")
    roc_auc = models.DecimalField(max_digits=6, decimal_places=5, null=True, db_column="roc_auc")
    pr_auc = models.DecimalField(max_digits=6, decimal_places=5, null=True, db_column="pr_auc")
    training_date = models.DateTimeField(db_column="training_date")

    class Meta:
        managed = False
        db_table = "model_metrics"
        ordering = ["-training_date"]

    def __str__(self):
        return self.model_name