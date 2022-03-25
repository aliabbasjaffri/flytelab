import torch
import errno
import pathlib


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("running to cuda device...")
        return torch.device("cuda")
    else:
        print("running on CPU")
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)


def generate_directory(directory_path: str) -> None:
    try:
        pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def parse_genres(filename):
    parts = filename.split("_")
    return " ".join(parts[:-1])
