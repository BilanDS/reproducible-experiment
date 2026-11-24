import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score

# Ініціалізація експерименту в MLflow
mlflow.set_experiment("reproducibility-benchmark")

with mlflow.start_run(run_name="if_baseline"):
    # Фіксація гіперпараметрів експерименту
    n_estimators = 100
    contamination = 0.02
    random_state = 42

    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("contamination", contamination)
    mlflow.log_param("random_state", random_state)

    # Генерація відтворюваного табличного датасету з 2% аномалій
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=6,
        weights=[0.98, 0.02],
        random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Навчання моделі виявлення аномалій
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(X_train)

    # Оцінка метрик якості на валідаційній вибірці
    y_pred = (model.predict(X_val) == -1).astype(int)
    anomaly_scores = -model.decision_function(X_val)

    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, anomaly_scores)

    mlflow.log_metric("f1_score", round(float(f1), 4))
    mlflow.log_metric("roc_auc", round(float(roc_auc), 4))

    # Збереження серіалізованого артефакту моделі
    mlflow.sklearn.log_model(model, "isolation_forest_model")

    print(f"Експеримент завершено успішно: F1-score = {f1:.4f}, ROC-AUC = {roc_auc:.4f}")