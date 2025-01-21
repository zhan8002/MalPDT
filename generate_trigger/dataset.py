import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100, size=4096):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        # self.files_list = glob(os.path.join(self.data_path, '*.txt'))
        self.files_list = os.listdir(self.data_path)

    def __getitem__(self, idx):
        bytes_cover_path = self.data_path + '/' + self.files_list[idx]

        # with open(bytes_cover_path, "rb") as f:
        #     data = f.read()
        #     data_array = bytearray(data)
        #     payload= np.frombuffer(data_array,dtype=np.uint8)


        with open(bytes_cover_path,'rb') as f:
            tmp = [i for i in f.read()[:self.size]]
            tmp = tmp+[256]*(self.size-len(tmp))

        bytes_cover = np.array(tmp)
        # bytes_cover = torch.from_numpy(bytes_cover/256).float()
        bytes_cover = torch.from_numpy(bytes_cover).float()
        bytes_cover = bytes_cover.unsqueeze(0)
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.

        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()
        secret = secret.unsqueeze(0)

        return bytes_cover, secret

    def __len__(self):
        return len(self.files_list)


if __name__ == '__main__':

    # define the path of benign byte segments 
    dataset = StegaData(data_path='./benign_section_content', secret_size=100, size=4096)

    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    print(image_input.max())
