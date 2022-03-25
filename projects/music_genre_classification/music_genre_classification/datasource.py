import torch
from torchaudio.datasets import GTZAN
import torchaudio.transforms as tt
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

random_seed = 42
torch.manual_seed(random_seed)
matplotlib.rcParams["figure.facecolor"] = "#ffffff"


def download_raw_dataset(
    target_folder: str = ".", download_dataset: bool = True
) -> Dataset:
    _dataset = GTZAN(root=target_folder, download=download_dataset)
    return _dataset


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
