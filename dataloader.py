import os
from torch.utils.data import Dataset
import torch
#from torchvision import transforms


class PokemonDataset(Dataset):
    def __init__(self, mask_dir, img_dir):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.image_list = os.listdir(img_dir)
        self.mask_list = os.listdir(mask_dir)
        
#        self.transform = transforms.Compose([
#            transforms.Grayscale(num_output_channels=1),  # Force single-channel
#            transforms.ToTensor(),  # Converts to float and scales to [0, 1]
#        ])
        
        print(len(self.image_list))
        print(len(self.mask_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.img_dir, image_name)
        mask_name = self.mask_list[idx]
        
        mask_path =  os.path.join(self.mask_dir, mask_name)
        
        image = torch.load(image_path)
        
        image = image[1:,1:]
        #image=torch.nn.functional.pad(image, pad=(1,1), mode='constant', value=None)
        image = image.unsqueeze(0)
        #image_single = image[:, 0:1, :, :]  # Keep only the first channel

        img = image.to(torch.float32).to("cuda")
        
        label = torch.load(mask_path)
        
        label = label[1:,1:]
       # label_single = label[:, 0:1, :, :]  # Keep only the first channel

        lbl = label.to(torch.float32).to("cuda")

#        image = torch.load(image_path).to("cuda")
#        label = torch.load(mask_path).to("cuda")
        return img, lbl
