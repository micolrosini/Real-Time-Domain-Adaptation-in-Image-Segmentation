import os
import wandb
from torch.autograd import Variable
import numpy as np
import torch.cuda.amp as amp
import time 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.cityscapes_dataset import cityscapesDataSet
from utils import poly_lr_scheduler,reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, best_model, create_meta_pseudo_labels, build_pretrained_model, compute_loss, uda_loss
from dataset.gta5_dataset import gta5DataSet
from arguments import get_args
from model.build_BiSeNet import BiSeNet
from model.discriminator_dsc import DSCDiscriminator
import torch.optim as optim


# Meta Pseudo Label training function

def train(args, teacher_G, teacher_G_opt, teacher_G_scaler, 
                teacher_D, teacher_D_opt, teacher_D_scaler,
                student_G, student_G_opt, student_G_scaler,
                student_D, student_D_opt, student_D_scaler,
                stud_lr, teacher_lr, epoch, 
                training_dataloader_source, training_dataloader_target, dataloader_val):

    # Loss definitions
    bce_loss = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    source_iter = enumerate(training_dataloader_source)
    target_iter = enumerate(training_dataloader_target)

    sup_stud_loss = list()           
    unsup_stud_loss = list() 
    sup_teacher_loss = list()
    unsup_teacher_loss = list() 
    teacher_tot_loss = list()

    source_label = 0
    target_label = 1
    i=0

    
    for batch_target, batch_source in zip(target_iter,source_iter):
        i+=1
        if (i*4) % 100 == 0:
            print(f"Epoch:{epoch}, Batch: {i*4}/500")
            
        teacher_G_opt.zero_grad()
        teacher_D_opt.zero_grad()
        student_G_opt.zero_grad()
        student_D_opt.zero_grad()

        # Sample an unlabelled example x_u and sample a labelled example (x_l, y_l) 
        _, (x_source, y_source, _, _) = batch_source
        _, (x_target, _, _, name) = batch_target
        
        # Put everything on cuda
        x_target = x_target.cuda()
        x_source = x_source.cuda()
        y_source = y_source.long().cuda()
        y_source = y_source.detach()

        # 1) Create pseudolabels
        y_pl = create_meta_pseudo_labels(teacher_G, args, x_target, name, epoch)
        y_pl = y_pl.long().cuda()
        y_pl = y_pl.detach()

        # Set networks in training mode
        student_G.train()
        teacher_G.train()

        # Freeze discriminators
        for param_stud, param_teacher in zip(student_D.parameters(), teacher_D.parameters()):
            param_stud.requires_grad = False
            param_teacher.requires_grad = False

        # 2) Compute supervised loss student
        seg_loss_source_stud, output_stud_source = compute_loss(student_G, x_source, y_source, loss_fn) 
        student_G_scaler.scale(seg_loss_source_stud).backward()

        # 3) Compute unsupervised loss student      
        seg_loss_target_stud, output_stud_target = compute_loss(student_G, x_target, y_pl, loss_fn)

        with amp.autocast():
            D_out = student_D(F.softmax(output_stud_target, dim=1))  
            loss_adversarial = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())   
        
            loss_target = args.lambda_adv * loss_adversarial + seg_loss_target_stud                                                  #LOSS ADVERSARIAL      
   
        # Update Student parameters
        student_G_scaler.scale(loss_target).backward()

        # 4) Train discriminator 
        for param in student_D.parameters():
            param.requires_grad = True

        with amp.autocast():
              output_source = output_stud_source.detach()
              D_out = student_D(F.softmax(output_source, dim =1))
              loss_D_source_stud = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

              output_target = output_stud_target.detach()
              D_out = student_D(F.softmax(output_target, dim=1))
              loss_D_target_stud = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())      
    
              loss_D = loss_D_source_stud/2 + loss_D_target_stud/2

        student_D_scaler.scale(loss_D).backward()
        
        student_D_scaler.step(student_D_opt)
        student_G_scaler.step(student_G_opt) 

        student_G_scaler.update()
        student_D_scaler.update()



        # 5) Compute H 
        H = stud_lr * seg_loss_target_stud.detach() * seg_loss_source_stud.detach()

        alpha = np.exp(epoch/40) - 1

        if epoch == 0:
            alpha = 1/H
        
        if epoch > 25:
            alpha = 1
        
        H = alpha * H
        
        # 6) Compute supervised loss teacher
        seg_loss_source_teacher, output_teacher_source = compute_loss(teacher_G, x_source, y_source, loss_fn) 
       
        # 7) Compute unsupervised loss teacher      
        seg_loss_target_teacher, output_teacher_target = compute_loss(teacher_G, x_target, y_pl, loss_fn)
        
        H = H.detach()
        #print(H.item())
        #teacher_G_scaler.scale(seg_loss_source_teacher).backward()
        uda_loss_teacher = uda_loss(teacher_G, x_target, y_pl, loss_fn)
        with amp.autocast():
            D_out = teacher_D(F.softmax(output_teacher_target, dim=1))  
            loss_adversarial = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())   
        
            loss_teacher = seg_loss_source_teacher + loss_adversarial + (H * seg_loss_target_teacher) + uda_loss_teacher                                                  #LOSS ADVERSARIAL      
        
        
        # 3) Update Teacher parameters 
        teacher_G_scaler.scale(loss_teacher).backward()

        # Train discriminator 
        for param in teacher_D.parameters():
            param.requires_grad = True

        with amp.autocast():
              output_source = output_stud_source.detach()
              D_out = teacher_D(F.softmax(output_source, dim =1))
              loss_D_source_teacher= bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

              output_target = output_stud_target.detach()
              D_out = teacher_D(F.softmax(output_target, dim=1))
              loss_D_target_teacher = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())      
              
              
              loss_D = loss_D_source_teacher/2 + loss_D_target_teacher/2

        
        teacher_D_scaler.scale(loss_D).backward()
        
        teacher_G_scaler.step(teacher_G_opt) 
        teacher_D_scaler.step(teacher_D_opt)

        teacher_G_scaler.update()
        teacher_D_scaler.update()
       

        sup_stud_loss.append(seg_loss_source_stud.item())
        unsup_stud_loss.append(seg_loss_target_stud.item()) 
        sup_teacher_loss.append(seg_loss_target_teacher.item())
        unsup_teacher_loss.append(seg_loss_target_teacher.item())
        teacher_tot_loss.append(loss_teacher.item())

          
    # Return stats of training
    update_info = {
            'epoch': epoch,
            'stud_sup_loss': np.array(sup_stud_loss).mean(),
            'stud_unsup_loss': np.array(unsup_stud_loss).mean(), 
            'teacher_sup_loss': np.array(sup_teacher_loss).mean(), 
            'teacher_unsup_loss': np.array(unsup_teacher_loss).mean(), 
            'teacher_tot_loss': np.array(teacher_tot_loss).mean()
            }
    
    return update_info, teacher_G, teacher_G_opt, teacher_D, teacher_D_opt, student_G, student_G_opt, student_D, student_D_opt



# Function to test the model 
def test(args, model, dataloader):
    print("Start validation!")
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
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou





# The main function
def main(params):
    args, img_mean = get_args(params)   
    max_miou = 0.3
    run_id = int(time.time())

    # Use wandb to store stats
    wandb.init(project="segmentation",
                name=f'Segmentation-MetapseudoLabels-{str(run_id)}',
                group=f'Segmentation-MetapseudoLabels',
                config=args)

    # Instanciate dataloaders
    training_dataset_target = cityscapesDataSet(args.dataset, args.data_train, crop_size=(args.crop_width , args.crop_height), encodeseg=0)
    training_dataloader_target = DataLoader(training_dataset_target,
                            batch_size= args.batch_size,
                            shuffle=True,
                            num_workers = args.num_workers, 
                            drop_last=True )

    training_dataset_source = gta5DataSet(args.source, args.path_source, crop_size=(1280,720))
    training_dataloader_source = DataLoader(training_dataset_source,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers = args.num_workers, 
                            drop_last=True)
    
    dataset_val = cityscapesDataSet(args.dataset, args.val, crop_size=(args.crop_width , args.crop_height), encodeseg=1)
    dataloader_val = DataLoader(dataset_val,
                            shuffle=True ,
                            num_workers = args.num_workers,
                            batch_size=1
                            )

    # Instanciate scalers
    student_G_scaler = amp.GradScaler()
    student_D_scaler = amp.GradScaler()
    teacher_G_scaler = amp.GradScaler()
    teacher_D_scaler = amp.GradScaler() 

    #student not pretreined, teacher pretreined
    # Instanciate models
    student_G = BiSeNet(args.num_classes, args.context_path)
    student_D = DSCDiscriminator(num_classes=args.num_classes) 
    
    if torch.cuda.is_available() and args.use_gpu:
      student_G = torch.nn.DataParallel(student_G).cuda() 
      student_D = torch.nn.DataParallel(student_D).cuda()              
                            
    student_D_opt = optim.Adam(student_D.parameters(), lr=args.learning_rateD, betas=(0.9, 0.99)) 
    student_G_opt = torch.optim.SGD(student_G.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)

     
    teacher_G, teacher_G_opt, teacher_D, teacher_D_opt  = build_pretrained_model(args)
    
    teacher_G_opt = torch.optim.SGD(teacher_G.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)

    teacher_D_opt = optim.Adam(teacher_D.parameters(), lr=args.learning_rateD, betas=(0.9, 0.99)) #torch.optim.SGD(student_G.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)

    print("------Training started------")
    for epoch in range(args.num_epochs):
        stud_lr = poly_lr_scheduler(student_G_opt, 
                                    args.learning_rate, 
                                    iter=epoch, 
                                    max_iter=args.num_epochs)

        teacher_lr =  poly_lr_scheduler(teacher_G_opt, 
                                    1.25e-2, 
                                    iter=epoch, 
                                    max_iter=args.num_epochs)

        train_info, teacher_G, teacher_G_opt, teacher_D, teacher_D_opt, student_G, student_G_opt, student_D, student_D_opt =train(args, teacher_G, teacher_G_opt, teacher_G_scaler, 
                teacher_D, teacher_D_opt, teacher_D_scaler,
                student_G, student_G_opt, student_G_scaler,
                student_D, student_D_opt, student_D_scaler,
                stud_lr, teacher_lr, epoch, 
                training_dataloader_source, training_dataloader_target, dataloader_val)

         
        wandb.log(train_info)
        
        if (epoch + 1) % 1 == 0 and epoch>0:
      
            prec, miou = test(args, student_G, dataloader_val)
            test_info = {"epoch": epoch, "precision": prec, "miou": miou}
            wandb.log(test_info)

            if miou > max_miou:
                print("Model saved!")
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                best_model(args, student_G, student_G_opt, student_D, student_D_opt, epoch + 50, "best_stud")
                best_model(args, teacher_G, teacher_G_opt, teacher_D, teacher_D_opt, epoch + 50, "best_teacher")
    
    print("------Training finished------")


# Entry point of the script
if __name__ == "__main__":
    params = [
    '--use_meta_pseudo_labels',' 0',
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
    '--checkpoint_name_save','model_output.pth',
    '--checkpoint_name_load','model_output_ssl_best_model_.pth'
    ]
    main(params)