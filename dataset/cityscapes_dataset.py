import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torch.utils import data
from PIL import Image
import json
from utils import encode_segmap


class cityscapesDataSet(data.Dataset):
    
    
    def __init__(self, root, list_path, crop_size=(1024,512), ignore_label=255, use_pseudolabels = False, pseudo_path=None, encodeseg= 1):

        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = [] 
        self.info = json.load(open(os.path.join(root, 'info.json'), 'r'))
        self.id_to_trainid = {k:v for k, v in self.info.get('label2train')}
        self.mean = self.info.get('mean')
        self.std = self.info.get('std')
        self.use_pseudolabels = use_pseudolabels
        self.pseudo_path = pseudo_path
        self.encodeseg = encodeseg
       
        for name in self.img_ids:
            
            # Select only the name without "_leftImg8bit.png"
            name = name[:-16]
            img_file = osp.join (root, 'images' , name + '_leftImg8bit.png')
            
            if self.use_pseudolabels == False:
              label_file = osp.join(root,'labels', name + '_gtFine_labelIds.png')
            else:
              label_file = osp.join(pseudo_path,'labels', name + '.png') #_gtFine_labelIds.png

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]) 
        name = datafiles["name"]
        
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        
        if self.encodeseg == 1:
            label = encode_segmap(label, self.id_to_trainid, self.ignore_label)
        
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    dst = cityscapesDataSet("./data/Cityscapes", list_path = "./data/Cityscapes/train.txt")
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, _, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()