# main.py
import torch
from torch import nn, optim
from datasets import HeartDataset2D
from models import CNN
from metrics import Accuracy, F1Score, Precision, Recall
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow
from mltrainer import Trainer, TrainerSettings, ReportTypes
import json
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

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
    reporttypes=[getattr(ReportTypes, rt) for rt in config['report_types']],
    scheduler_kwargs=None,
    earlystop_kwargs=None
)

# Training
with mlflow.start_run():
#    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer = torch.optim.Adam
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_tag("model", config['model_name'])
    mlflow.set_tag("dataset", config['dataset_name'])
    mlflow.log_param("scheduler", "None")
    mlflow.log_param("earlystop", "None")
    mlflow.log_params(config)
    mlflow.log_param("epochs", settings.epochs)
    mlflow.log_param("shape0", config['shape'][0])
    mlflow.log_param("optimizer", str(optimizer))

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optimizer,
        traindataloader=trainstreamer.stream(),
        validdataloader=teststreamer.stream(),
        scheduler=None,
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