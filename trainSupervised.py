import argparse
from torch.utils.data import DataLoader
from torch.utils import data
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, compute_loss
from loss import DiceLoss
import torch.cuda.amp as amp
from dataset.cityscapes_dataset import cityscapesDataSet
from PIL import Image
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
            predict = reverse_one_hot(predict) #Transform into a 2D array with only 1 channel, where each pixel value is the classified class key
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label) #accuracy over all classes given the prediction and the label
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            precision_record.append(precision)
        
        precision = np.mean(precision_record) # mean over all precisions
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    scaler = amp.GradScaler()

    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        
        for i, (data,label,_,_) in enumerate(dataloader_train):
            
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad() #sets the gradients of all optimized to zero.
            
            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)
                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
            
def main(params):
    args, img_mean = get_args(params)

    cropSize= (args.crop_width , args.crop_height)

    dataset_train = cityscapesDataSet(args.dataset, args.data, max_iters= args.num_epochs* args.num_iter*args.batch_size,  crop_size=cropSize,  ignore_label=255, encodeseg = 1)
    dataset_val = cityscapesDataSet(args.dataset, args.val, max_iters= args.num_epochs* args.num_iter*args.batch_size ,  crop_size=cropSize, encodeseg=1)
    
    # Define HERE your dataloaders:
    dataloader_train = DataLoader(dataset_train,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers = args.num_workers, 
                            drop_last=True
                          ) 
    dataloader_val = DataLoader(dataset_val,
                            batch_size=1,
                            shuffle=True ,
                            num_workers = args.num_workers
                            ) 

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)
    
    # final test
    val(args, model, dataloader_val)
    
if __name__ == '__main__':
    params = [
        '--num_epochs', '100',
        '--learning_rate', '2.5e-2',
        '--data', './dataset/data/Cityscapes/train.txt',
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_101_sgd',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'rmsprop'
    ]

main(params)
