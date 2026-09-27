import argparse
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class CreditScorePreprocessor:
    """Menangani pembersihan data, split train/val/test, encoding target,
    dan pembuatan preprocessing pipeline (persis alur notebook)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

        self.drop_cols = [
            "Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"
        ]

        self.num = [
            "Age", "Annual_Income", "Monthly_Inhand_Salary",
            "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
            "Num_of_Loan", "Delay_from_due_date",
            "Num_of_Delayed_Payment", "Changed_Credit_Limit",
            "Num_Credit_Inquiries", "Outstanding_Debt",
            "Credit_Utilization_Ratio", "Total_EMI_per_month",
            "Amount_invested_monthly", "Monthly_Balance"
        ]

        self.obj = [
            "Month", "Occupation", "Type_of_Loan", "Credit_Mix",
            "Credit_History_Age", "Payment_of_Min_Amount",
            "Payment_Behaviour"
        ]

        self.target_encoder = LabelEncoder()

    def handle_invalid_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mengubah nilai yang tidak masuk akal secara bisnis menjadi NaN
        agar ditangani oleh imputer (bukan dianggap outlier biasa)."""
        data = data.copy()

        data.loc[(data["Age"] < 18) | (data["Age"] > 100), "Age"] = np.nan
        data.loc[data["Num_Bank_Accounts"] < 0, "Num_Bank_Accounts"] = np.nan
        data.loc[data["Num_Credit_Card"] > 50, "Num_Credit_Card"] = np.nan
        data.loc[data["Interest_Rate"] > 100, "Interest_Rate"] = np.nan
        data.loc[(data["Num_of_Loan"] < 0) | (data["Num_of_Loan"] > 20), "Num_of_Loan"] = np.nan

        data.loc[
            (data["Num_of_Delayed_Payment"] < 0) |
            (data["Num_of_Delayed_Payment"] > 100),
            "Num_of_Delayed_Payment"
        ] = np.nan

        data.loc[data["Num_Credit_Inquiries"] > 100, "Num_Credit_Inquiries"] = np.nan
        data.loc[data["Monthly_Balance"] < -1e10, "Monthly_Balance"] = np.nan

        return data

    def clean_and_split(self, data_path: str | Path):
        """Load CSV, perbaiki tipe data numerik, drop kolom identitas,
        lalu split Train 80% / Validation 10% / Test 10% (stratify),
        encode target, dan terapkan invalid-value handling di tiap subset.
        """

        df = pd.read_csv(Path(data_path))
        logger.info("Data dimuat: %d baris, %d kolom", df.shape[0], df.shape[1])

        for col in self.num:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("_", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.drop(columns=self.drop_cols, inplace=True)

        X = df.drop("Credit_Score", axis=1)
        y = df["Credit_Score"]

        # Train 80% / Temp 20%
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.20,
            random_state=self.random_state,
            stratify=y
        )

        # Temp -> Validation 10% / Test 10%
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=self.random_state,
            stratify=y_temp
        )

        logger.info(
            "Train: %s | Validation: %s | Test: %s",
            X_train.shape, X_val.shape, X_test.shape
        )

        # Encode target (fit di train, transform di val & test)
        y_train = self.target_encoder.fit_transform(y_train)
        y_val = self.target_encoder.transform(y_val)
        y_test = self.target_encoder.transform(y_test)

        # Invalid value handling (setelah split, sebelum imputasi)
        X_train = self.handle_invalid_values(X_train)
        X_val = self.handle_invalid_values(X_val)
        X_test = self.handle_invalid_values(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_transformer(self) -> ColumnTransformer:
        """Pipeline preprocessing: median imputer + scaler untuk numerik,
        most_frequent imputer + OneHotEncoder untuk kategorikal."""

        numeric_pipeline = Pipeline([
            ("num_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformer = ColumnTransformer([
            ("numPreprocess", numeric_pipeline, self.num),
            ("catPreprocess", categorical_pipeline, self.obj)
        ])

        return transformer


class CreditScoreModelTrainer:
    """Melatih, membandingkan (di validation set), dan melacak (MLflow)
    beberapa model Credit Score, lalu memilih & mengevaluasi final model
    terbaik di test set — persis alur notebook.

    Model yang dibandingkan: Logistic Regression, Decision Tree,
    Random Forest. (Gradient Boosting sengaja tidak diikutkan karena
    terlalu lambat — training sequential + butuh dense array dari data
    one-hot yang besar.)
    """

    TARGET_NAMES = ["Good", "Poor", "Standard"]  # urutan alfabetis LabelEncoder

    def __init__(
        self,
        experiment_name: str = "Credit Score Prediction",
        artifact_path: str = "artifacts",
        random_state: int = 42,
    ):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.random_state = random_state

        self.preprocessor = CreditScorePreprocessor(random_state=random_state)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

        self.best_pipeline: Pipeline | None = None
        self.best_model_name: str | None = None

    def get_candidate_models(self) -> dict:
        """Kandidat model, sama seperti notebook."""
        return {
            "LogisticRegression": {
                "estimator": LogisticRegression(
                    max_iter=1000, random_state=self.random_state
                ),
            },
            "DecisionTree": {
                "estimator": DecisionTreeClassifier(
                    random_state=self.random_state
                ),
            },
            "RandomForest": {
                "estimator": RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
            },
        }

    def build_pipeline(self, estimator) -> Pipeline:
        transformer = self.preprocessor.get_transformer()

        return Pipeline([
            ("preprocessing", transformer),
            ("classifier", estimator),
        ])

    def evaluate(self, y_true, y_pred) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        }

    def run(self, data_path: str | Path):
        x_train, x_val, x_test, y_train, y_val, y_test = self.preprocessor.clean_and_split(data_path)

        candidates = self.get_candidate_models()
        comparison_rows = []
        best_score = -1.0

        with mlflow.start_run(run_name="model_comparison") as parent_run:
            mlflow.log_param("split", "train80_val10_test10")
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("num_imputer", "median")
            mlflow.log_param("cat_imputer", "most_frequent")
            mlflow.log_param("encoder", "OneHotEncoder")
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_param("target_encoder", "LabelEncoder")

            # === 1. Training & perbandingan model di VALIDATION set ===
            for name, cfg in candidates.items():
                logger.info("Training %s...", name)

                with mlflow.start_run(run_name=name, nested=True):
                    pipeline = self.build_pipeline(cfg["estimator"])
                    pipeline.fit(x_train, y_train)

                    y_val_pred = pipeline.predict(x_val)
                    metrics = self.evaluate(y_val, y_val_pred)

                    mlflow.log_param("model", name)
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"val_{metric_name}", value)

                    logger.info(
                        "%s (validation) -> accuracy=%.4f | f1_macro=%.4f",
                        name, metrics["accuracy"], metrics["f1_macro"]
                    )

                    comparison_rows.append({"Model": name, **metrics})

                    if metrics["f1_macro"] > best_score:
                        best_score = metrics["f1_macro"]
                        self.best_pipeline = pipeline
                        self.best_model_name = name

            comparison_df = pd.DataFrame(comparison_rows).sort_values(
                "f1_macro", ascending=False
            )
            logger.info(
                "Perbandingan model (validation set):\n%s",
                comparison_df.to_string(index=False)
            )

            comparison_path = self.artifact_dir / "model_comparison_validation.csv"
            comparison_df.to_csv(comparison_path, index=False)
            mlflow.log_artifact(str(comparison_path))

            mlflow.log_param("best_model", self.best_model_name)
            mlflow.log_metric("best_val_f1_macro", best_score)

            # === 2. Evaluasi final model terbaik di TEST set ===
            y_test_pred = self.best_pipeline.predict(x_test)
            test_metrics = self.evaluate(y_test, y_test_pred)

            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            report = classification_report(
                y_test, y_test_pred, target_names=self.TARGET_NAMES
            )
            logger.info(
                "Model terbaik: %s | Evaluasi TEST set:\n%s",
                self.best_model_name, report
            )

            report_path = self.artifact_dir / "classification_report_test.txt"
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path))

            cm = confusion_matrix(y_test, y_test_pred)
            cm_path = self.artifact_dir / "confusion_matrix_test.csv"
            pd.DataFrame(
                cm, index=self.TARGET_NAMES, columns=self.TARGET_NAMES
            ).to_csv(cm_path)
            mlflow.log_artifact(str(cm_path))

            # === 3. Simpan model & preprocessing artifacts ===
            model_file_path = self.artifact_dir / "credit_score_pipeline.pkl"
            joblib.dump(self.best_pipeline, model_file_path)
            mlflow.log_artifact(str(model_file_path))

            target_encoder_path = self.artifact_dir / "target_encoder.pkl"
            joblib.dump(self.preprocessor.target_encoder, target_encoder_path)
            mlflow.log_artifact(str(target_encoder_path))

            
            mlflow.sklearn.log_model(
                        self.best_pipeline,
                            name="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            logger.info(
                "✅ Model terbaik (%s) disimpan ke %s", self.best_model_name, model_file_path
            )
            logger.info("MLflow run_id: %s", parent_run.info.run_id)

            return parent_run.info.run_id, x_test, y_test
            

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training & perbandingan model Credit Score Prediction (train/val/test 80/10/10)."
    )
    parser.add_argument(
        "--data-path", type=str, default="data_D.csv",
        help="Path ke file CSV dataset (default: data_D.csv)"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="Credit Score Prediction",
        help="Nama experiment MLflow"
    )
    parser.add_argument(
        "--artifact-path", type=str, default="artifacts",
        help="Folder untuk menyimpan model & artifact hasil training"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random state untuk reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = CreditScoreModelTrainer(
        experiment_name=args.experiment_name,
        artifact_path=args.artifact_path,
        random_state=args.random_state,
    )

    trainer.run(args.data_path)


if __name__ == "__main__":
    main()
