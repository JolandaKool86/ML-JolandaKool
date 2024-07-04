from mads_datasets.base import BaseDatastreamer
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics
import models
import datasets
import metrics
from pathlib import Path
import torch
import mlflow
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from mltrainer.preprocessors import BasePreprocessor
from loguru import logger
from metrics import ThresholdedRecall
from typing import Dict

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

# Define the training function for Ray Tune
def train(config: Dict):
    device = "cpu"
    shape = (16, 12)

    data_dir = config["data_dir"]
    trainfile = data_dir/"heart_train.parq"
    testfile = data_dir/"heart_train.parq"
    traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=shape)
    testdataset = datasets.HeartDataset2D(testfile, target="target", shape=shape)

    trainstreamer = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
    teststreamer = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)

    model = models.CNN(config)
    model.to(device)

    # metrics
    f1micro = metrics.F1Score(average='micro')
    f1macro = metrics.F1Score(average='macro')
    precision = metrics.Precision('micro')
    recall = metrics.Recall('macro')
    accuracy = metrics.Accuracy()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    
    settings = TrainerSettings(
        epochs=10,
    #    metrics=[accuracy, f1micro, f1macro, precision, recall],
        metrics=[accuracy, f1micro, f1macro, precision, recall, ThresholdedRecall(threshold=0.2, average='micro')],
    #    logdir=f"heart2D/{config['num_layers']}layers_{config['hidden']}hidden",
        logdir=Path("."),
        train_steps=len(trainstreamer),
        valid_steps=len(teststreamer),
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None
    )
    
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optimizer,
        traindataloader=trainstreamer.stream(),
        validdataloader=teststreamer.stream(),
        scheduler=scheduler
    )
    
    trainer.loop()

if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    data_dir = Path("data").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()

    config = {
   #     "num_layers": tune.randint(2, 5),
        "num_layers": 1,
        "hidden": tune.randint(16, 256),
        "num_classes": 2,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "dropout": tune.uniform(0.0, 0.5),
        "shape": (16, 12),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("f1micro")
    reporter.add_metric_column("f1macro")
    reporter.add_metric_column("precision")
    reporter.add_metric_column("Recall")

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
        local_dir=str(config["tune_dir"]),
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()



# Shutdown Ray
#tune.shutdown()