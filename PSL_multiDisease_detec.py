import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import math
# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

from model.PSScreen import PSScreen
from loss.my_method_loss import partial_BCELoss, Partial_MultiLabelKLDivergenceLoss, MMDLoss
from utils.helper import get_word_file
from utils.metrics import AverageMeter
from utils.checkpoint import load_pretrained_model, save_checkpoint
from datasets.retinal_disease_image import Balanced_dataloader, test_dataloader, valid_dataloader
from utils.validation import evaluate
from mydata_config import arg_parse, logger, show_args

def main():
    # Argument Parse
    args = arg_parse()

    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = 'exp/log/{}.log'.format(args.post)

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")  #load train,test, and valid dataset
    train_loader = Balanced_dataloader(args)
    test_loader = test_dataloader(args)
    valid_loader = valid_dataloader(args)
    logger.info("==> Done!\n")

    # Load the network
    logger.info("==> Loading the network...")

    WordFile = get_word_file()  #load disease description embedding
    model = PSScreen(WordFile,
                 classNum=args.classNum,alpha=args.alpha)   #load PSScreen

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model = load_pretrained_model(model, args)  #load pretrained backbone

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location='cpu')    
        best_f1, args.startEpoch = checkpoint['best_f1'], checkpoint['best_epoch']
        model.load_state_dict(checkpoint['state_dict'])                              # Load the saved checkpoint
        logger.info("==> Checkpoint epoch: {0}, f1: {1}".format(args.startEpoch, best_f1))

    model.cuda()
    logger.info("==> Done!\n")

    criterion = {
                 'MMDLoss' : MMDLoss().cuda(),                   #Feature distillation loss
                 'BCELoss': partial_BCELoss(reduce=True, size_average=True).cuda(),  #classification loss for GT and pseudo labels
                 'KLloss': Partial_MultiLabelKLDivergenceLoss().cuda(),                 #self-distillation loss
                 }

    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weightDecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepEpoch, gamma=0.1)
    if args.evaluate:
        evaluate(test_loader, model, args, False, True)
        return

    logger.info('Total: {:.3f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024.0 ** 3))


    best_f1=0
    isBest=None

    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):
        
    
        Train(train_loader, model, criterion, optimizer, args, scheduler, epoch)
        

        seen_dataset_overall_result = evaluate(valid_loader, model, args, True, False)
        
        if epoch>=args.gen_psl_epoch:
            valid_overall_f1 = seen_dataset_overall_result['f1']
            isBest, best_f1 = valid_overall_f1 > best_f1, max(valid_overall_f1, best_f1)
            if isBest:
                best_epoch = epoch
            save_checkpoint(args, {'best_epoch': best_epoch, 'state_dict': model.state_dict(), 'best_f1': best_f1},
                            isBest)
        
            logger.info('[Best] [Epoch{0}]: Best f1 is {1:.3f}'.format((best_epoch), best_f1))

        scheduler.step()


    best_checkpoints = 'exp/checkpoint/{0}/Checkpoint_Best.pth'.format(args.post)
    logger.info('... loading pretrained weights from %s' % best_checkpoints)
    checkpoint = torch.load(best_checkpoints, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['best_epoch']  
    logger.info(f'model performs best on validset in epoch{best_epoch}')
    evaluate(test_loader, model, args, False, True)


def Train(train_loader, model, criterion, optimizer, args, scheduler,epoch):

    model.train()
    loss,loss1,loss2,loss3,loss4 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")
    end = time.time()
    best_f1 = 0.0
    for batchIndex, (input, target) in enumerate(train_loader):
        model.train()  
        input, target = input.cuda(), target.float().cuda()
        data_time.update(time.time() - end)
        with torch.autograd.set_detect_anomaly(True):
            if epoch<args.gen_psl_epoch:   # The pseudo-label loss is not computed in the early training stage
                output,output_aug,semantic_feature,semantic_feature_aug = model(input)
                
                loss1_ = criterion['BCELoss'](output, target)
                loss2_ = 0 * loss1_
                loss3_ = criterion['KLloss'](output_aug,output,target)
                loss4_ = criterion['MMDLoss'](semantic_feature,semantic_feature_aug)
            else:
                output,output_aug,semantic_feature, semantic_feature_aug, pseudolabel = model(input,target)
      
                loss1_ = criterion['BCELoss'](output, target)
                loss2_ = criterion['BCELoss'](output_aug, pseudolabel.detach())
                loss3_ = criterion['KLloss'](output_aug,output,target)
                loss4_ = criterion['MMDLoss'](semantic_feature,semantic_feature_aug)

        loss_ = loss1_+ args.lam*loss2_ +loss3_ + 0.05*loss4_
        loss.update(loss_.item(), input.size(0))
        loss1.update(loss1_.item(), input.size(0))
        loss2.update(loss2_.item(), input.size(0))
        loss3.update(loss3_.item(), input.size(0))
        loss4.update(loss4_.item(), input.size(0))

        # Backward
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batchIndex % args.printFreq == 0:
            logger.info('[Train] [Epoch {0}]: [{1}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} Learn Rate {lr:.6f}\n'
                        '\t\t\t\t\toverall Loss {loss.avg:.4f} real label Loss {loss1.avg:.4f} pseudo label Loss {loss2.avg:.4f}  consistency loss {loss3.avg:.8f} MMD loss {loss4.avg:.8f} '.format(
                        epoch, batchIndex, len(train_loader), batch_time=batch_time, data_time=data_time, lr=optimizer.param_groups[0]['lr'],
                        loss=loss,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4))
            sys.stdout.flush()
    
        if batchIndex==len(train_loader):
            break




if __name__ == "__main__":
    main()

