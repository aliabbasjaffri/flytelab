from datasource import download_raw_dataset
from preprocess import preprocess_data
from data_split import generate_data_split
from model import MusicGenreClassificationModel
from train import train, evaluate
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import torch
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
from flytekit import task, workflow


@dataclass_json
@dataclass
class TrainingParameters(object):
    epochs: int = 1
    max_learning_rate: float = 0.005
    optimization_function = torch.optim.Adam
    weight_decay: float = 1e-4
    gradient_clip: float = 0.1


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
) -> (List[any], MusicGenreClassificationModel):

    return train(
        epochs=epochs,
        max_learning_rate=max_learning_rate,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimization_function=optimization_function,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
    )


@task
def evaluate_model(model: MusicGenreClassificationModel, test_dataloader: DataLoader):
    return evaluate(model=model, test_dataloader=test_dataloader)


@workflow
# def music_genre_classifier() -> (Dict[str, any], MultiTemporalFeatureMap):
def music_genre_classifier() -> MusicGenreClassificationModel:
    training_params = TrainingParameters()
    download_dataset(target_folder=".", download=True)
    data_preprocessing()
    train_dataloader, validation_dataloader, test_dataloader = split_dataset(
        train_split_pc=0.01, validation_split_pc=0.01, test_split_pc=0.98
    )
    training_data, model = train_model(
        epochs=training_params.epochs,
        max_learning_rate=training_params.max_learning_rate,
        training_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimization_function=training_params.optimization_function,
        weight_decay=training_params.weight_decay,
        gradient_clip=training_params.gradient_clip
    )

    evaluation_data = evaluate_model(model=model, test_dataloader=validation_dataloader)
    # evaluation_data = evaluate_model(model=model, test_dataloader=test_dataloader)

    # for sending to model tracker
    model_artifacts: dict = {
        "training_data": training_data,
        "evaluation_data": evaluation_data,
    }

    return model


if __name__ == "__main__":
    genre_classification_model = music_genre_classifier()
