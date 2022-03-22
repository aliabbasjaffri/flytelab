import torch
from torchaudio.datasets import GTZAN

# from torchvision.datasets import ImageFolder
import torchaudio.transforms as tt
from torch.utils.data import random_split

# from torchaudio.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MusicDataset
from preprocess import data_dir
from utils import get_default_device, DeviceDataLoader

random_seed = 42
torch.manual_seed(random_seed)

matplotlib.rcParams["figure.facecolor"] = "#ffffff"


def download_raw_dataset():
    _dataset = GTZAN(root=".", download=True)
    return _dataset


def generate_data_split(
    batch_size: int = 150,
    train_split_pc: float = 0.6,
    validation_split_pc: float = 0.2,
    test_split_pc: float = 0.2,
) -> (DataLoader, DataLoader, DataLoader):
    _dataset = MusicDataset(data_dir)

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
    dataset = download_raw_dataset()

    waveform, samplerate, label = dataset[0]
    print(
        f"shape of waveform is {waveform.size()}, sample rate with {samplerate} Hz and its label is {label}"
    )

    print("Waveform: {}\n".format(waveform))
    plt.figure()
    plt.plot(waveform.t().numpy())

    spectrogram = tt.Spectrogram()(waveform)
    print("shape of spectrogram {}".format(spectrogram.size()))

    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram.log2()[0, :, :].numpy(), cmap="magma")

    # plt.show()

    train_dataloader, validation_dataloader, test_dataloader = generate_data_split()
