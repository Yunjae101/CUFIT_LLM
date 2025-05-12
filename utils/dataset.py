import json
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision import datasets as dset
import torchvision
from torch.utils.data import Dataset, DataLoader
from .aptos import APTOS2019

def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return train_transform, test_transform

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = torchvision.datasets.ImageFolder(path + '/train', train_transform)
    np.random.seed(seed)
    
    new_data = []
    noise_imgs = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(7))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
            noise_imgs.append([train_data.samples[i][0], train_data.samples[i][1], new_label])
    train_data.samples = new_data

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))
    
    with open('noise_imgs.txt', 'w') as f:
        for i in range(len(noise_imgs)):
            f.write('{} {} {}\n'.format(noise_imgs[i][0], noise_imgs[i][1], noise_imgs[i][2]))

    valid_data = torchvision.datasets.ImageFolder(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_aptos_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = APTOS2019(path, train=True, transforms = train_transform)

    np.random.seed(seed)
    new_data = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(5))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
    train_data.samples = new_data

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = APTOS2019(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_mnist_noise_dataset(dataname, noise_rate = 0.2, batch_size = 32, seed = 0):
    # from medmnist import NoduleMNIST3D
    from medmnist import PathMNIST, BloodMNIST, OCTMNIST, TissueMNIST, OrganCMNIST
    train_transform, test_transform = get_transform()

    if dataname == 'pathmnist':
        train_data = PathMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = PathMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 9
    if dataname == 'bloodmnist':
        train_data = BloodMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = BloodMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'octmnist':
        train_data = OCTMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OCTMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 4
    if dataname == 'tissuemnist':
        train_data = TissueMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = TissueMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'organcmnist':
        train_data = OrganCMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OrganCMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 11

    np.random.seed(seed)
    # new_imgs = []
    new_labels =[]
    for i in range(len(train_data.imgs)):
        if np.random.rand() > noise_rate: # clean sample:
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(train_data.labels[i][0])
        else:
            label_index = list(range(num_classes))
            label_index.remove(train_data.labels[i])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(new_label)
    # train_data.imgs = new_imgs
    train_data.labels = new_labels

    new_labels = []
    for i in range(len(test_data.labels)):
        new_labels.append(test_data.labels[i][0])
    test_data.labels = new_labels

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

# def read_noise_datalist(path, batch_size=1, seed=0):
#     with open(path, 'r') as f:

class NoiseDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (str): noise_imgs.txt 파일의 경로.
                각 줄이 "<img_path> <origin_label> <noise_label>" 형태로 기록되어 있음.
            transform (callable, optional): 이미지에 적용할 변환 함수.
        """
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue  # 형식이 올바르지 않은 줄은 건너뜁니다.
                img_path, origin_label, noise_label = parts
                self.samples.append((img_path, int(origin_label), int(noise_label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, origin_label, noise_label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # 평가 시에 모델의 예측과 함께 원래 라벨과 노이즈 라벨 모두를 활용할 수 있도록 반환합니다.
        return image, origin_label, noise_label

# noise_imgs.txt 파일에 기록된 데이터만 로드하는 dataloader를 반환하는 함수
def read_noise_datalist(path, batch_size=32, num_workers=8):
    """
    Args:
        path (str): noise_imgs.txt 파일이 위치한 디렉토리의 경로.
        batch_size (int): 배치 사이즈.
        num_workers (int): DataLoader에서 사용하는 worker 수.
    Returns:
        DataLoader: noise_imgs.txt에서 읽은 데이터를 로드하는 DataLoader.
    """
    # 테스트 시에 사용할 transform을 가져옵니다.
    _, test_transform = get_transform()
    noise_txt = path + '/noise_imgs.txt'
    dataset = NoiseDataset(noise_txt, transform=test_transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,     # 평가에서는 보통 순서를 그대로 유지합니다.
                        pin_memory=True,
                        num_workers=num_workers)
    return loader

def get_clean_dataset(path, batch_size = 32):
    train_transform, test_transform = get_transform()
    train_data = torchvision.datasets.ImageFolder(path + '/train', train_transform)
    valid_data = torchvision.datasets.ImageFolder(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_clean_aptos_dataset(path, batch_size = 32):
    train_transform, test_transform = get_transform()
    train_data = APTOS2019(path, train=True, transforms = train_transform)
    valid_data = APTOS2019(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

# if __name__=='__main__':
#     noise_loader = read_noise_datalist('/SSDc/yunjae_heo/CUFIT_LLM', batch_size=1)
#     # 모델 평가 시:
#     for imgs, origin_labels, noise_labels in noise_loader:
#     #     outputs = model(imgs)  # 모델의 예측 결과
#     #     # 예를 들어, 예측 결과와 원래 라벨, 노이즈 라벨 비교:
#         print("원래 라벨:", origin_labels)
#         print("노이즈 라벨:", noise_labels)
#         exit()
#         # print("예측 라벨:", outputs.argmax(dim=1))