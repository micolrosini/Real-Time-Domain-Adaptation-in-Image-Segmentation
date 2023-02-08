import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from model.build_BiSeNet import BiSeNet
from torch.autograd import Variable
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
import torch.cuda.amp as amp
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.gta5_dataset import gta5DataSet
from model.discriminator import FCDiscriminator
from model.discriminator_dsc import DSCDiscriminator
from utils import upload_model, best_model
from arguments import get_args

def val(args, model, dataloader):
    print('start val!')
    
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))

        for i, (data,label,_,_) in enumerate(dataloader):
            
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou


def train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val, model_D, optimizer_D, IMG_MEAN, cropSize):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    scaler = amp.GradScaler()
    discriminator_scaler = amp.GradScaler()
    # Loss
    bce_loss = torch.nn.BCEWithLogitsLoss()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    source_label = 0
    target_label = 1    
    max_miou = 0
    step = 0

    for epoch in range(args.num_epochs):

          
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs, power = args.power) 
        discriminator_lr = poly_lr_scheduler(optimizer_D, args.learning_rateD, iter=epoch, max_iter=args.num_epochs, power = args.power)        
        
        
        model.train()
        model_D.train()

        total=len(dataloader_source) * args.batch_size
        tq = tqdm(total=total)
        tq.set_description('epoch %d, lr %f'% (epoch , lr))
        
        
        loss_record_source = []
        loss_record_target = []
        loss_D_record = []

        source_iter = enumerate(dataloader_source)
        target_iter = enumerate(dataloader_target)

        for batch_source, batch_target in zip(source_iter, target_iter):
            
            _, (data_source, label_source, _, _) = batch_source
            
            
            _, (data_target, label_target, _, _) = batch_target
            
                
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # Train Segmentation network
            for param in model_D.parameters():
                param.requires_grad = False

            # Train with source
            data_source = data_source.cuda()
            label_source = label_source.long().cuda()

            with amp.autocast():
                output, output_sup1, output_sup2 = model(data_source)
                loss1 = loss_func(output, label_source)
                loss2 = loss_func(output_sup1, label_source)
                loss3 = loss_func(output_sup2, label_source)
                loss_segmentation_source = loss1 + loss2 + loss3             #LOSS SEGMENTATION
            
            scaler.scale(loss_segmentation_source).backward()
            

            # Train with target
            data_target = data_target.cuda()

            if args.use_pseudolabels==1:
                label_target = label_target.long().cuda()

            with amp.autocast(): 
                output_target, output_sup1_t, output_sup2_t = model(data_target)
                if args.use_pseudolabels==1:
                    loss1_t = loss_func(output_target, label_target)
                    loss2_t = loss_func(output_sup1_t, label_target)
                    loss3_t = loss_func(output_sup2_t, label_target)
                    loss_seg_target = loss1_t + loss2_t + loss3_t

                else:
                    loss_seg_target = 0

           
                D_out=model_D(F.softmax(output_target, dim=1))  
                loss_adversarial = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())   
                loss_target = args.lambda_adv * loss_adversarial + loss_seg_target                                                   #LOSS ADVERSARIAL      
   
            scaler.scale(loss_target).backward()
            

            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # Train D with source
            with amp.autocast():
              output_source = output.detach()
              D_out = model_D(F.softmax(output_source, dim =1))
              loss_D_source = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

             # Train D with target
            with amp.autocast():
              output_target = output_target.detach()
              D_out = model_D(F.softmax(output_target, dim=1))
              loss_D_target = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())      
    
              loss_D = loss_D_source/2 + loss_D_target/2

            discriminator_scaler.scale(loss_D).backward()

            discriminator_scaler.step(optimizer_D)
            scaler.step(optimizer)

            discriminator_scaler.update()
            scaler.update()


            tq.update(args.batch_size)
            

            tq.set_postfix(loss_segmentation_source='%.6f' % loss_segmentation_source, loss_target='%.6f' % loss_target, loss_D='%.6f' % loss_D)
            step += 1
            writer.add_scalar('loss_seg_source_step', loss_segmentation_source, step)
            writer.add_scalar('loss_target_step', loss_target, step)
            writer.add_scalar('loss_D_step', loss_D, step)
            
            loss_record_source.append(loss_segmentation_source.item())
            loss_record_target.append(loss_target.item())
            loss_D_record.append(loss_D.item())

        
        tq.close() 
        loss_train_mean_source = np.mean(loss_record_source)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean_source), epoch)
        print('loss for train source : %f' % (loss_train_mean_source))

        loss_train_mean_target = np.mean(loss_record_target)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean_target), epoch)
        print('loss for train target : %f' % (loss_train_mean_target))

        loss_D_mean = np.mean(loss_D_record)
        writer.add_scalar('epoch/loss_', float(loss_D_mean), epoch)
        print('loss for discriminator : %f' % (loss_D_mean))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                best_model(args, model, model_D, optimizer, optimizer_D, epoch, "best_model")
                
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
        
        
        

def main(params):
    args, IMG_MEAN = get_args(params)

    #sistema
    cropSize= (args.crop_width , args.crop_height)
    cropSizeGTA5 = (1280,720)
   
    # Create dataset train GTA 
    dataset_train_source = gta5DataSet(args.source, args.path_source, crop_size=cropSizeGTA5)
    dataloader_source = DataLoader(dataset_train_source,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers = args.num_workers)
  
    if args.use_pseudolabels == 1:
        args.checkpoint_name_save = args.checkpoint_name_save.replace(".pth", "_ssl.pth")
    

    if args.use_pseudolabels == 0:
        dataset_train_target = cityscapesDataSet(args.dataset, args.data_train,  crop_size=cropSize)

        dataloader_target = DataLoader(dataset_train_target,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers = args.num_workers
                            )
    else:  
        print('entrato nel dataset_train_target delle pseudo')
        dataset_train_target = cityscapesDataSet(args.dataset, args.data_train,  crop_size=cropSize, pseudo_path= args.pseudo_path, use_pseudolabels = 1, encodeseg= 0)
      
        dataloader_target = DataLoader(dataset_train_target,
                            batch_size= args.batch_size,
                            shuffle=True,
                            num_workers = args.num_workers,    
                        )

    dataset_val = cityscapesDataSet(args.dataset, args.data_val, crop_size=cropSize, use_pseudolabels=0, encodeseg =1 )
    dataloader_val = DataLoader(dataset_val,
                            batch_size= 1,
                            shuffle=True,
                            num_workers = args.num_workers,                           
                          ) 

    
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    
    if(args.Discriminator==0): 
      print('entrato Discriminator 0')  
      model_D = FCDiscriminator(num_classes=args.num_classes)
    else: #uso quello light weight 
      print('entrato Discriminator 1')
      model_D= DSCDiscriminator(num_classes=args.num_classes) 

    if torch.cuda.is_available() and args.use_gpu:
      model_D = torch.nn.DataParallel(model_D).cuda()
      model = torch.nn.DataParallel(model).cuda()
  
    
    
    # build optimizers
    
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rateD, betas=(0.9, 0.99)) 
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    

    
    if args.use_pretrained_model ==1 : 
      model, model_D, optimizer, optimizer_D, epoch_start = upload_model(args, model, model_D, optimizer, optimizer_D)   
    else:
      train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val,  model_D, optimizer_D, IMG_MEAN, cropSize)
    
    
    val(args, model, dataloader_val)
    
if __name__ == '__main__':
    params = [
        '--use_pseudolabels','0',
        '--save_dir_plabels', '/content/drive/MyDrive/dataset/pseudolabels',
        '--pseudo_path', './dataset/pseudolabels/labels',
        '--num_epochs', '50',
        '--learning_rate', '2.5e-4',
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
        '--use_pretrained_model','0',
        '--checkpoint_name_save','model_output.pth',
        '--checkpoint_name_load','model_output_best.pth'

    ]


main(params)