import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from utils import rgb_label, upload_model
from arguments import get_args
from dataset.cityscapes_dataset import cityscapesDataSet
from model.build_BiSeNet import BiSeNet
from model.discriminator_dsc import DSCDiscriminator

def create_pseudo_labels(model, args, targetloader):

    print("Creating pseudo labels....")
    
    # Create directory if not exixts
    if not os.path.exists(args.meta_pseudo_path): #creo cartella pseudo labels
        os.makedirs(args.meta_pseudo_path + "/labels")
    if not os.path.exists(args.meta_pseudo_path + "/labels_rgb"): #creo cartella pseudo labels a colori
        os.makedirs(args.meta_pseudo_path + "/labels_rgb")

    model.eval() 
    model.cuda()
    
    target_train_loader = iter(targetloader)

    for _ in range(len(targetloader)):
      image_t, _, _, name_t = next(target_train_loader)        
      for index, (image, name) in enumerate(zip(image_t, name_t)):
        if image is not None: 
          image = image.unsqueeze(0) #Returns a new tensor with a dimension of size one inserted at the specified position. In our case 0 so only one array
          
          outputs = model(image.cuda()) #predictions from Bisnet
          outputs = F.softmax(outputs, dim=1) #Applies a softmax function. It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
          output = F.upsample(outputs, (512, 1024), mode='nearest')[0] 
          
          #Upsamples the input to either the given (512, 1024), algorithm used for upsampling: 'nearest'Copies the value from the nearest pixel
          outputs = torch.transpose(output,2,1)
          outputs = torch.transpose(outputs,0,2)
          label,prob = torch.argmax(outputs, axis =2), torch.max(outputs, axis=2)[0]
         
          
          predicted_label = torch.clone(label)
          predicted_prob = torch.clone(prob)
            
        THRESHOLDS = []
        for i in range(19):
          mask = predicted_label.eq(i) #predicted_prob[predicted_label==i]
          x = torch.masked_select(predicted_prob, mask)

          if len(x) == 0:
            THRESHOLDS.append(0)  

          x = torch.sort(x)[0]
          q = torch.median(x)
          
          THRESHOLDS.append(q.detach().cpu())
          

        THRESHOLDS = np.array(THRESHOLDS)
        THRESHOLDS[THRESHOLDS>0.9]=0.9
       

        name = name.replace("leftImg8bit", "gtFine_labelIds")

        
        
        for i in range(19):
            mask1 = predicted_prob.lt(THRESHOLDS[i])
            mask2 = predicted_label.eq(i)
            
            mask = torch.logical_and(mask1,mask2)
            predicted_label= predicted_label.masked_fill(mask, 255) 
          
        output = np.array(predicted_label.detach().cpu(), dtype=np.uint8)
        rgb_image = rgb_label(output)
        output = Image.fromarray(output)
         
             
        saving_file_name = args.meta_pseudo_path + "/labels/" + name + ".png" 
        saving_file_name_rgb = args.meta_pseudo_path + "/labels_rgb/" + name + ".png" 
        rgb_image.save(saving_file_name_rgb)
        output.save(saving_file_name)

    print('pseudo label created')      
        

def main(params):
    
    args, img_mean = get_args(params)
    cropSize= (args.crop_width , args.crop_height)
    cropSizeGTA5 = (1280,720)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    
    if torch.cuda.is_available() and args.use_gpu:
      model = torch.nn.DataParallel(model).cuda()

    model_D= DSCDiscriminator(num_classes=args.num_classes) 

    if torch.cuda.is_available() and args.use_gpu:
      model_D = torch.nn.DataParallel(model_D).cuda()  

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rateD, betas=(0.9, 0.99))
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
   
   
    model, model_D, optimizer, optimizer_D, epoch_start = upload_model(args, model, model_D, optimizer, optimizer_D) 
    train_dataset_target= cityscapesDataSet(args.dataset, args.data_train, crop_size=(args.crop_width , args.crop_height), encodeseg=0) 
    targetloader= DataLoader(train_dataset_target, batch_size=1, shuffle=True, num_workers=1)
    create_pseudo_labels(model, args,targetloader)
    print("PSEUDOLABEL CREATE")

    
    
    
if __name__ == '__main__':
    params = [
        '--save_dir_plabels', '/content/drive/MyDrive/dataset/pseudolabels', 
        '--pseudo_path', './dataset/pseudolabels/labels',
        '--num_epochs', '50',
        '--learning_rate', '2.5e-2',
        '--data_train', './dataset/data/Cityscapes/train.txt',
        '--data_val', './dataset/data/Cityscapes/val.txt',
        '--num_workers', '4',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_101_sgd',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        '--Discriminator', '1',
        '--use_pretrained_model','1',
        '--checkpoint_name_save','model_output.pth',
        '--checkpoint_name_load','model_output_ssl_best_model_.pth'
    ]


main(params)