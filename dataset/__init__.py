from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
  def __init__(self, images_path, labels, transform=None):

    self.images_path = images_path
    self.labels = labels
    self.transform = transform

  def __len__(self):
   
    return len(self.images_path)

  def __get_item__(self, index):
    
    img = self.images_path[index]
    lbl = self.labels[index]

    
    img = Image.open(img).convert("RGB")
    
   
    if self.transform is not None: 
      img = self.transform(img)
    
   
    return img, lbl