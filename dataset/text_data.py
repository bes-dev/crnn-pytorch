from torch.utils.data import Dataset
import json
import os
import cv2

def text_to_seq(text, abc):
    seq = []
    for c in text:
        seq.append(abc.find(c) + 1)
    return seq

class TextDataset(Dataset):
    def __init__(self, data_path, mode="train", transform=None):
        super().__init__()
        self.data_path = data_path
        config = json.load(open(os.path.join(data_path, "desc.json")))
        self.names = config[mode]
        self.abc = config["abc"]
        self.transform = transform
        self.seq_len = 5

    def abc_len(self):
        return len(self.abc)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_path, "data", self.names[idx]["name"]))
        seq = text_to_seq(self.names[idx]["text"], self.abc)
        sample = {"img": img, "seq": seq, "seq_len": len(seq)}
        if self.transform:
            sample = self.transform(sample)
        return sample
