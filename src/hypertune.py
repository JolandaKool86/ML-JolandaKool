# hypertune.py
import torch
from torch import nn
from datasets import HeartDataset1D, HeartDataset2D
from models import CNN, Transformer
from metrics import Accuracy, F1Score, Precision, Recall
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow
from mltrainer import Trainer, TrainerSettings, ReportTypes
import json
import matplotlib.pyplot as plt
import sys
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from loguru import logger
from typing import Dict

# def load_config(model_type):
#     with open('config.json', 'r') as f:
#         all_configs = json.load(f)
#     return all_configs[model_type]

def train(config):
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load datasets
    trainfile = Path(config['trainfile']).resolve()
    testfile = Path(config['testfile']).resolve()
    traindataset = HeartDataset2D(trainfile, target="target", shape=tuple(config['shape']))
    testdataset = HeartDataset2D(testfile, target="target", shape=tuple(config['shape']))
    traindataset.to(device)
    testdataset.to(device)

    # Load streamers
    trainstreamer = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=config['batch_size'])
    teststreamer = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=config['batch_size'])

    # Initialize model
    model = CNN(config)
    model.to(device)

    # Define metrics
    metrics = [
        Accuracy(),
        F1Score(average='micro'),
        F1Score(average='macro'),
        Precision('micro'),
        Recall('macro')
    ]

    # MLFlow setup
    mlflow.set_tracking_uri(config['database_uri'])
    mlflow.set_experiment(config['experiment_name'])

    # Training settings
    settings = TrainerSettings(
        epochs=config['epochs'],
        metrics=metrics,
        logdir=config['logdir'],
        train_steps=len(trainstreamer),
        valid_steps=len(teststreamer),
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs=None,
        earlystop_kwargs=None
    )

    # Training
    with mlflow.start_run():
        optimizer = torch.optim.Adam
        loss_fn = nn.CrossEntropyLoss()

        mlflow.set_tag("model", config['model_name'])
        mlflow.set_tag("dataset", config['dataset_name'])
        mlflow.log_param("scheduler", "None")
        mlflow.log_param("earlystop", "None")
        mlflow.log_params(config)
        mlflow.log_param("epochs", settings.epochs)
        mlflow.log_param("shape0", config.get('shape', [0])[0])
        mlflow.log_param("optimizer", str(optimizer))

        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=None,
            device=device
        )
        trainer.loop()

if __name__ == "__main__":
    ray.init()

    data_dir = Path("../data/heart_train.parq").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()

    config = {
        "input_size": 3,
        "output_size": 20,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.3),
        "num_layers": tune.randint(2, 5),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(config["tune_dir"]),
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()