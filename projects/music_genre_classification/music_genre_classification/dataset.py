import os
import torchaudio
from utils import parse_genres
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, root: str = "data/"):
        super().__init__()
        self.root = root
        self.files = [
            filename for filename in os.listdir(root) if filename.endswith(".wav")
        ]
        self.classes = list(set(parse_genres(filename) for filename in self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        fpath = os.path.join(self.root, filename)
        audio = torchaudio.load(fpath)[0]
        class_idx = self.classes.index(parse_genres(filename))
        return audio, class_idx


if __name__ == "__main__":
    dataset = MusicDataset()
    print(f"Length of dataset: {len(dataset)}")
    print(dataset[0])
