from glob import glob
import os.path as osp

class IMDB:
    def __init__(self, mode="train"):
        data_paths = glob(f"data/{mode}/neg/*.txt") + glob(f"data/{mode}/pos/*.txt")
        self.data = [self.get_sample(p) for p in data_paths]
        self.idx = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx < len(self.data):
            result = self.data[self.idx]
            self.idx += 1
            return result
        else:
            self.idx = 0
            raise StopIteration

    @staticmethod
    def get_sample(path):
        with open(path, "r", encoding="utf-8") as fr:
            txt = fr.read()
        basename = osp.basename(path)[:-4]
        label = int(basename.split("_")[-1])
        return label, txt