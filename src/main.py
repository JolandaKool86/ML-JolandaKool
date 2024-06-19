# main.py
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

def load_config(model_type):
    with open('config.json', 'r') as f:
        all_configs = json.load(f)
    return all_configs[model_type]

def train(config, model_type):
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["cnn", "transformer"]:
        print("Usage: python main.py <cnn|transformer>")
        sys.exit(1)

    model_type = sys.argv[1]
    config = load_config(model_type)
    train(config, model_type)