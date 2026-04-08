import os
from PIL import Image
from torch.utils.data import Dataset

class reiddts(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        for img in os.listdir(root):
            if not img.endswith(".jpg"):
                continue

            pid = img.split('_')[0]

            if pid in ['0000', '-1']:
                continue

            img_path = os.path.join(self.root, img)

            self.samples.append((pid, img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pid, img_path = self.samples[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return img, pid