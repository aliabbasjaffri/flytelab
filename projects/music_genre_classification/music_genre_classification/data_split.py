from torch.utils.data import DataLoader, random_split

from dataset import MusicDataset
from projects.music_genre_classification.music_genre_classification.preprocess import (
    preprocessed_data_dir,
)
from utils import get_default_device, DeviceDataLoader


def generate_data_split(
    batch_size: int = 150,
    train_split_pc: float = 0.6,
    validation_split_pc: float = 0.2,
    test_split_pc: float = 0.2,
) -> (DataLoader, DataLoader, DataLoader):
    _dataset = MusicDataset(preprocessed_data_dir)

    assert (train_split_pc + validation_split_pc + test_split_pc) == 1

    train_split_size = int(len(_dataset) * train_split_pc)
    validation_split_size = int(len(_dataset) * validation_split_pc)
    test_split_size = int(len(_dataset) * test_split_pc)

    train_dataset, validation_dataset, test_dataset = random_split(
        _dataset, [train_split_size, validation_split_size, test_split_size]
    )

    print(
        f"Training Dataset size: {len(train_dataset)}, "
        f"Validation Dataset size: {len(validation_dataset)}, "
        f"Test Dataset size: {len(test_dataset)}"
    )

    device = get_default_device()

    train_dataloader = DeviceDataLoader(
        dataloader=DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        ),
        device=device,
    )

    validation_dataloader = DeviceDataLoader(
        dataloader=DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size * 2,
            num_workers=4,
            pin_memory=True,
        ),
        device=device,
    )

    test_dataloader = DeviceDataLoader(
        dataloader=DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        ),
        device=device,
    )

    return train_dataloader, validation_dataloader, test_dataloader


if __name__ == "__main__":
    _train_dataloader, _validation_dataloader, _test_dataloader = generate_data_split()

    (
        __train_dataloader,
        __validation_dataloader,
        __test_dataloader,
    ) = generate_data_split(
        train_split_pc=0.01, validation_split_pc=0.01, test_split_pc=0.98
    )
