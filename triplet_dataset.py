from torch.utils.data import Dataset
from collections import defaultdict
import os
import random
from PIL import Image
from torchvision import transforms

class market1501(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images_list = defaultdict(list)

        for image in os.listdir(root):
            if not image.endswith(".jpg"):
                continue

            img_id = image.split('_')[0]

            if img_id in ['0000', '-1']:
                continue

            image_path = os.path.join(root, image)
            self.images_list[img_id].append(image_path)

        self.ids_list = [pid for pid in self.images_list.keys() if len(self.images_list[pid]) >= 2]

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        person_id = self.ids_list[idx]

        anchor_path, positive_path = random.sample(self.images_list[person_id], 2)

        negative_id = random.choice(self.ids_list)

        while negative_id == person_id:
            negative_id = random.choice(self.ids_list)

        negative_path = random.choice(self.images_list[negative_id])

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img




