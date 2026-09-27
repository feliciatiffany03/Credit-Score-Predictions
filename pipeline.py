from pathlib import Path

from data_ingestion import DataIngestion
from train import CreditScoreModelTrainer
from evaluation import ModelEvaluator


class CreditScorePipeline:
    """
    Master pipeline orchestrating:
    Data Ingestion
    Training
    Evaluation
    Deployment Approval
    """

    def __init__(
        self,
        raw_data_path: str | Path,
        accuracy_threshold: float = 0.70
    ):

        self.base_dir = Path(__file__).parent

        self.raw_data_path = Path(raw_data_path)

        self.ingested_dir = (
            self.base_dir / "ingested"
        )

        self.accuracy_threshold = (
            accuracy_threshold
        )

        # Core Components
        self.ingestor = DataIngestion(
            self.raw_data_path,
            self.ingested_dir
        )

        self.trainer = (
            CreditScoreModelTrainer()
        )

        self.evaluator = (
            ModelEvaluator()
        )

    def execute(self):

        print(
            "🚀 Executing Credit Score Pipeline..."
        )

        # Step 1
        ingested_file_path = (
            self.ingestor.run()
        )

        # Step 2
        run_id, x_test, y_test = (
            self.trainer.run(
                ingested_file_path
            )
        )

        # Step 3
        accuracy, precision, recall, f1 = (
            self.evaluator.run(
                run_id,
                x_test,
                y_test
            )
        )

        # Step 4
        print(
            "\n--- Deployment Approval Decision ---"
        )

        if accuracy >= self.accuracy_threshold:

            print(
                "🎉 Success: Model approved for deployment!"
            )

        else:

            print(
                f"❌ Rejected: Accuracy ({accuracy:.3f}) "
                f"below threshold ({self.accuracy_threshold})"
            )


if __name__ == "__main__":

    DATA_INPUT = (
        Path(__file__).parent /
        "data_D.csv"
    )

    credit_pipeline = (
        CreditScorePipeline(
            raw_data_path=DATA_INPUT,
            accuracy_threshold=0.70
        )
    )

    credit_pipeline.execute()

