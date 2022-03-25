from datasource import download_raw_dataset
from preprocess import preprocess_data
from data_split import generate_data_split
from model import MultiTemporalFeatureMap
from train import train, evaluate
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from flytekit import task, workflow


@task
def download_dataset(target_folder: str, download: bool) -> None:
    """
    This function looks for dataset in the target folder
    Default target folder is .
    This function intends to download the dataset because its around 1.2 GB
    """
    download_raw_dataset(target_folder=target_folder, download_dataset=download)


@task
def data_preprocessing() -> None:
    """
    This function preprocesses the data already downloaded and
    saves the data into a data directory
    """
    preprocess_data()


@task
def split_dataset(
    train_split_pc: float, validation_split_pc: float, test_split_pc: float
) -> (DataLoader, DataLoader, DataLoader):
    return generate_data_split(
        train_split_pc=train_split_pc,
        validation_split_pc=validation_split_pc,
        test_split_pc=test_split_pc,
    )


@task
def train_model(
    epochs: int,
    max_learning_rate: float,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimization_function: any,
    weight_decay: float,
    gradient_clip: float,
) -> MultiTemporalFeatureMap:

    model = train(
        epochs=epochs,
        max_learning_rate=max_learning_rate,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimization_function=optimization_function,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
    )
    return model


@task
def evaluate_model(
    model: MultiTemporalFeatureMap, test_dataloader: DataLoader
) -> dict[str, any]:
    return evaluate(model=model, test_dataloader=test_dataloader)


@workflow
# def music_genre_classifier(training_params: dict[str, any]) -> MultiTemporalFeatureMap:
def music_genre_classifier() -> MultiTemporalFeatureMap:
    training_params = {
        "epochs": 1,
        "max_learning_rate": 0.005,
        "optimization_function": torch.optim.Adam,
        "weight_decay": 1e-4,
        "gradient_clip": 0.1,
    }
    download_dataset(target_folder=".", download=True)
    data_preprocessing()
    train_dataloader, validation_dataloader, test_dataloader = split_dataset(
        train_split_pc=0.01, validation_split_pc=0.01, test_split_pc=0.98
    )
    training_data, model = train_model(
        epochs=training_params["epochs"],
        max_learning_rate=training_params["max_learning_rate"],
        training_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimization_function=training_params["optimization_function"],
        weight_decay=training_params["weight_decay"],
        gradient_clip=training_params["gradient_clip"],
    )

    # evaluation_data = evaluate_model(model=model, validation_dataloader=test_dataloader)
    evaluation_data = evaluate_model(
        model=model, validation_dataloader=validation_dataloader
    )

    model_artifacts = {
        "training_data": training_data,
        "evaluation_data": evaluation_data,
    }

    # for sending to model tracker
    print(model_artifacts)

    return model


if __name__ == "__main__":
    # training_params = {
    #     "epochs": 1,
    #     "max_learning_rate": 0.005,
    #     "optimization_function": torch.optim.Adam,
    #     "weight_decay": 1e-4,
    #     "gradient_clip": 0.1,
    # }

    # genre_classification_model = music_genre_classifier(training_params=training_params)
    genre_classification_model = music_genre_classifier()
