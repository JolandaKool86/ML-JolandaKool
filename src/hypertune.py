# hypertune.py
import torch
from torch import nn, optim
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
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor

def load_config(model_type):
    with open('config.json', 'r') as f:
        all_configs = json.load(f)
    return all_configs[model_type]

def train_model(config, model_type):
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    trainfile = Path(config['trainfile']).resolve()
    testfile = Path(config['testfile']).resolve()
    traindataset = HeartDataset1D(trainfile, target="target") if model_type == "transformer" else HeartDataset2D(trainfile, target="target", shape=tuple(config['shape']))
    testdataset = HeartDataset1D(testfile, target="target") if model_type == "transformer" else HeartDataset2D(testfile, target="target", shape=tuple(config['shape']))
    traindataset.to(device)
    testdataset.to(device)

    # Load streamers
    trainstreamer = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=config['batch_size'])
    teststreamer = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=config['batch_size'])

    # Initialize model
    model = Transformer(config) if model_type == "transformer" else CNN(config)
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
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs=None,
        earlystop_kwargs=None
    )

    # Training
    with mlflow.start_run():
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
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

    # Evaluation
    y_true = []
    y_pred = []

    testdata = teststreamer.stream()
    for _ in range(len(teststreamer)):
        X, y = next(testdata)
        yhat = model(X)
        yhat = yhat.argmax(dim=1)
        y_pred.append(yhat.cpu().tolist())
        y_true.append(y.cpu().tolist())

    yhat = [x for y in y_pred for x in y]
    y = [x for y in y_true for x in y]

    cfm = confusion_matrix(y, yhat)
    plot = sns.heatmap(cfm, annot=True, fmt=".3f", cmap='viridis')
    plot.set(xlabel="Predicted", ylabel="Target")
    plt.show()

    accuracy_score = Accuracy()(y, yhat)
    tune.report(accuracy=accuracy_score)

def hypertune(model_type):
    # Load initial configuration
    config = load_config(model_type)

    # Ensure local_dir is an absolute path
    storage_path = Path(config["logdir"]).resolve()

    # Define search space for hyperparameters
    search_space = {
        "batch_size": tune.choice([16, 32, 64]),
        "learning_rate": tune.loguniform(1e-5, 1e-2)
    }

    if model_type == "cnn":
        search_space.update({
            "hidden": tune.randint(16, 128),
            "num_layers": tune.randint(1, 5),
        })
    else:
        search_space.update({
            "hidden": tune.randint(64, 256),
            "num_heads": tune.randint(2, 8),
            "num_blocks": tune.randint(1, 6),
            "dropout": tune.uniform(0.1, 0.5),
        })

    reporter = CLIReporter(metric_columns=["accuracy", "training_iteration"])
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )
    bohb_search = TuneBOHB()

    analysis = tune.run(
        tune.with_parameters(train_model, model_type=model_type),
        resources_per_trial={"cpu": 2, "gpu": 0},
        metric="accuracy",
        mode="max",
        config=search_space,
        num_samples=50,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        progress_reporter=reporter,
        storage_path=str(storage_path)  # Convert to string
    )

    print(f"Best hyperparameters found: {analysis.best_config}")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["cnn", "transformer"]:
        print("Usage: python hypertune.py <cnn|transformer>")
        sys.exit(1)

    model_type = sys.argv[1]
    ray.init()
    hypertune(model_type)
    ray.shutdown()