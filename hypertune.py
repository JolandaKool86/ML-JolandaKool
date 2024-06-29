from mads_datasets.base import BaseDatastreamer
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics
import models
import datasets
import metrics
from pathlib import Path
import torch
import mlflow
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
#from src import models
#from src import datasets, metrics
from mltrainer.preprocessors import BasePreprocessor
import ray
from loguru import logger
# Define dataset paths and shape
trainfile = Path('../data/heart_train.parq').resolve()
testfile = Path('../data/heart_test.parq').resolve()
shape = (16, 12)

# Define datasets and datastreamers
traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=shape)
testdataset = datasets.HeartDataset2D(testfile, target="target", shape=shape)

# Determine device
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = "cpu"

traindataset.to(device)
testdataset.to(device)

# Create datastreamers
trainstreamer = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
teststreamer = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)

# Define the training function for Ray Tune
def train(config):
    model = models.CNN(config)
    model.to(device)
    
    f1micro = metrics.F1Score(average='micro')
    f1macro = metrics.F1Score(average='macro')
    precision = metrics.Precision('micro')
    recall = metrics.Recall('macro')
    accuracy = metrics.Accuracy()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    settings = TrainerSettings(
        epochs=config['epochs'],
        metrics=[accuracy, f1micro, f1macro, precision, recall],
    #    logdir=f"heart2D/{config['num_layers']}layers_{config['hidden']}hidden",
        logdir=Path("."),
        train_steps=len(trainstreamer),
        valid_steps=len(teststreamer),
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs=None,
        earlystop_kwargs=None
    )
    
    mlflow.set_tag("model", "Conv2D")
    mlflow.set_tag("dataset", "heart_small_binary")
    mlflow.log_param("scheduler", "None")
    mlflow.log_param("earlystop", "None")
    mlflow.log_params(config)
    mlflow.log_param("epochs", settings.epochs)
    mlflow.log_param("shape0", shape[0])
    mlflow.log_param("optimizer", str(optimizer))
    mlflow.log_params(settings.optimizer_kwargs)
    
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optimizer,
        traindataloader=trainstreamer.stream(),
        validdataloader=teststreamer.stream(),
        scheduler=None
    )
    
    trainer.loop()

# Define the hyperparameter search space
config_space = {
    'hidden': tune.randint(16, 128),
    'dropout_rate': 0.25,
#    'num_layers': tune.randint(1, 6),
#    'epochs': tune.choice([5, 10, 20])
}

# Set up Ray Tune
reporter = CLIReporter()
reporter.add_metric_column("accuracy")

bohb_hyperband = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=50,
    reduction_factor=3,
    stop_last_trials=False,
)

bohb_search = TuneBOHB()

# Run hyperparameter search with Ray Tune
analysis = tune.run(
    train,
    config=config_space,
    metric="accuracy",
    mode="max",
    progress_reporter=reporter,
 #   storage_path="tune_results",
    num_samples=50,
    search_alg=bohb_search,
    scheduler=bohb_hyperband,
    verbose=1,
)


if __name__ == "__main__":
    ray.init()

    data_dir = Path('../data/heart_test.parq').resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()

    config = {
        "output_size": 2,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.3),
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
        local_dir=str(config["tune_dir"]),
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()



# Shutdown Ray
tune.shutdown()