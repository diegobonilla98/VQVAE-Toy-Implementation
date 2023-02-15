import glob
import matplotlib
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, size=(64, 64), train=True):
        self.size = size
        images_list = sorted(glob.glob('D:\\FFHQ\\images1024x1024\\images1024x1024\\*\\*'))
        s = len(images_list)
        if train:
            self.images_list = images_list[: int(s * 0.8)]
        else:
            self.images_list = images_list[-int(s * 0.2):]

        self.transformation = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda l: l - 0.5)
        ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        image = self.transformation(image)
        return image


if __name__ == '__main__':
    dl = CustomDataset()
    training_dataloader = DataLoader(dl, batch_size=8, shuffle=True,
                                     num_workers=2, pin_memory=True)
    for i, image in enumerate(training_dataloader):
        print()
    print()
