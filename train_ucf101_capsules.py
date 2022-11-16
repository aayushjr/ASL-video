import sys
import os
import utils_caps as utils
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torch import optim
import time
from torch.nn.modules.loss import _Loss
import datetime
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
import numpy as np


class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=24):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        target = target.long()
        # target comes in as class number like 23
        # x comes in as a length 64 vector of averages of all locations
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + r
        #margin = self.m_min + (self.m_max - self.m_min) * r
        # print('predictions', x[0])
        # print('target',target[0])
        # print('margin',margin)
        # print('target',target.size())

        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
            # print('an at value',x[i][lb])
        at = at.view(b, 1).repeat(1, E)
        # print('at shape',at.shape)
        # print('at',at[0])

        zeros = x.new_zeros(x.shape)
        # print('zero shape',zeros.shape)
        absloss = torch.max(.9 - (at - x), zeros)
        loss = torch.max(margin - (at - x), zeros)
        # print('loss',loss.shape)
        # print('loss',loss)
        absloss = absloss ** 2
        loss = loss ** 2
        absloss = absloss.sum() / b - .9 ** 2
        loss = loss.sum() / b - margin ** 2
        # loss = loss.sum()/b

        return loss, absloss


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, labels, classes):
        # print('labels',labels[0])
        # print('predictions',classes[0])
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return margin_loss


def get_accuracy(predicted_actor, actor):
    max, prediction = torch.max(predicted_actor, 1)
    prediction = prediction.view(-1, 1)
    actor = actor.view(-1, 1)
    correct = torch.sum(actor == prediction.float()).item()
    accuracy = correct / float(prediction.shape[0])
    return accuracy


def get_accuracy2(predicted_actor, actor):
    # This gets the f-measure of our network
    predictions = predicted_actor > 0.5

    tp = ((predictions + actor) > 1).sum()
    tn = ((predictions + actor) < 1).sum()
    fp = (predictions > actor).sum()
    fn = (predictions < actor).sum()

    return (tp + tn) / (tp + tn + fp + fn)




def model_interface(minibatch, criterion5, r=0, masking=False):
    data = minibatch['data']
    action = minibatch['action']
    action = action.cuda()
    segmentation = minibatch['segmentation']
    segmentation = segmentation
    data = data.type(torch.cuda.FloatTensor)
    output, predicted_action = model(data, action)
    
    if masking:
        mask_cls = minibatch['mask_cls']
        mask_cls = mask_cls.cuda()
        output = output * mask_cls
    # print('prediction',predicted_actor)

    #criterion5 = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    class_loss, abs_class_loss = criterion5(predicted_action, action, r)

    criterion1 = nn.BCEWithLogitsLoss(size_average=True)
    loss1 = criterion1(output, segmentation.float().cuda())

    
    seg_loss = loss1    # * 0.0002
    total_loss =  seg_loss + class_loss

    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss)



def train(model, train_loader, optimizer, criterion, epoch, r, save_path, short=False, masking=False):
    start_time = time.time()
    steps = len(train_loader)
    # print('training: batch size ',TRAIN_BATCH_SIZE,' ',N_EPOCHS,'epochs', steps,' steps ')
    model.train(mode=True)
    model.training = True
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    class_loss_sent = []
    accuracy_sent = []
    print('epoch  step    loss   seg    class  accuracy')
    #optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=0, eps=1e-6)
    
    start_time = time.time()
    for batch_id, minibatch in enumerate(train_loader):
        if short:
            if batch_id > 40:
                break

        optimizer.zero_grad()

        #if (batch_id + 1) % 100 == 0:
        #    r = (1. * batch_id + (epoch - 1) * steps) / (30 * steps)

        output, predicted_action, segmentation, action, loss, s_loss, c_loss = model_interface(minibatch, criterion, r, masking=masking)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        seg_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        accuracy.append(get_accuracy(predicted_action, action))

        report_interval = 20
        if (batch_id + 1) % report_interval == 0:
            r_total = np.array(total_loss).mean()
            r_seg = np.array(seg_loss).mean()
            r_class = np.array(class_loss).mean()
            r_acc = np.array(accuracy).mean()
            print('%d/%d  %d/%d  %.3f  %.3f  %.3f  %.3f'%(epoch,N_EPOCHS,batch_id + 1,steps,r_total,r_seg,r_class,r_acc))
            sys.stdout.flush()
    
    r_total = np.array(total_loss).mean()
    r_seg = np.array(seg_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    print('%d/%d  %d/%d  %.3f  %.3f  %.3f  %.3f'%(epoch,N_EPOCHS,batch_id + 1,steps,r_total,r_seg,r_class,r_acc))
    sys.stdout.flush()
    
    end_time = time.time()
    train_epoch_time = end_time - start_time
    print("Training time: ", train_epoch_time)
    
    #file = 'weights' + str(epoch)
    if epoch > 10 and epoch % 2 == 0:
        torch.save(model.state_dict(), save_path)
        print('saved weights to ', save_path)
    
    #return r


def validate(model, val_data_loader, criterion, epoch, short=False):
    steps = len(val_data_loader)
    # print('validation: batch size ', VAL_BATCH_SIZE, ' ', N_EPOCHS, 'epochs', steps, ' steps ')
    model.eval()
    model.training = False
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    total_IOU = 0
    validiou = 0
    print('validating...')
    start_time = time.time()
    
    with torch.no_grad():
        
        for batch_id, minibatch in enumerate(val_data_loader):
            if short:
                if batch_id > 40:
                    break
            
            output, predicted_action, segmentation, action, loss, s_loss, c_loss = model_interface(minibatch, criterion, r, masking=False)
            total_loss.append(loss.item())
            seg_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            accuracy.append(get_accuracy(predicted_action, action))


            maskout = output.cpu()
            maskout_np = maskout.data.numpy()
            # utils.show(maskout_np[0])

            # use threshold to make mask binary
            maskout_np[maskout_np > 0] = 1
            maskout_np[maskout_np < 1] = 0
            # utils.show(maskout_np[0])

            truth_np = segmentation.cpu().data.numpy()
            for a in range(minibatch['data'].shape[0]):
                iou = utils.IOU2(truth_np[a], maskout_np[a])
                if iou == iou:
                    total_IOU += iou
                    validiou += 1
                else:
                    print('bad IOU')
    
    val_epoch_time = time.time() - start_time
    print("Time taken: ", val_epoch_time)
    
    r_total = np.array(total_loss).mean()
    r_seg = np.array(seg_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    average_IOU = total_IOU / validiou
    print('Validation %d  %.3f  %.3f  %.3f  %.3f IOU %.3f' % (epoch, r_total, r_seg, r_class, r_acc, average_IOU))
    sys.stdout.flush()

if __name__ == '__main__':
    USE_CUDA = True if torch.cuda.is_available() else False
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    N_EPOCHS = 80
    LR = 0.00005
    LR_step_size = 60
    LR_SCHEDULER = False     # True  
    IS_MASKING = True        # False
    pretrained_load = True   # False
    load_previous_weights = False # True
    
    
    MODEL_TYPE = 'i3d'  # 'c3d' 'i3d'
    
    if MODEL_TYPE == 'i3d':
        from capsules_ucf101 import CapsNet
        vid_size = [224, 224]
    elif MODEL_TYPE == 'c3d':
        from caps_c3d_ucf101_up2 import CapsNet
        vid_size = [112, 112]
    else:
        print("Invalid model type ", MODEL_TYPE)
        exit(0)
    
    
    percent = str(sys.argv[1])
    
    print("="*40)
    print("TRAIN_BATCH_SIZE: ", TRAIN_BATCH_SIZE)
    print("VAL_BATCH_SIZE: ", VAL_BATCH_SIZE)
    print("N_EPOCHS: ", N_EPOCHS)
    print("Learning Rate: ", LR)
    print("Masking mode: ", IS_MASKING)
    print("pretrained_load: ", pretrained_load)
    print("load previous weights: ", load_previous_weights)
    print("Percent: ", percent)
    print("-"*40)
    
    from load_ucf101_pytorch_gaus import UCF101DataLoader      # Currently has gaussian interpolation weights 
    trainset = UCF101DataLoader('train', vid_size, TRAIN_BATCH_SIZE, percent=percent, use_random_start_frame=False)
    validationset = UCF101DataLoader('validation',vid_size, VAL_BATCH_SIZE, use_random_start_frame=False)
    
    train_data_loader = DataLoader(
        dataset=trainset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=5,
        shuffle=True
    )

    val_data_loader = DataLoader(
        dataset=validationset,
        batch_size=VAL_BATCH_SIZE,
        num_workers=5,
        shuffle=False
    )
    
    model = CapsNet(pretrained_load=pretrained_load)
    if load_previous_weights:
        model.load_previous_weights('./trained/active_learning/checkpoints_ucf101_capsules_i3d/i3d-ucf101-5e-05_8_ADAM_capsules_Multi_5perRand_prune1wUN_total5per_GausInterp_flipDouble_slowLR_Dropout_PreTrainedCharades_RGB_Spread_BCE_epoch_030.pth')
        
    if USE_CUDA:
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=[0.5, 0.999], weight_decay=0, eps=1e-6)
    if LR_SCHEDULER:
        print("Scheduler on for step on ", LR_step_size)
        scheduler = StepLR(optimizer, step_size=LR_step_size, gamma=0.1)
    
    save_root = os.path.join('trained', 'active_learning', 'checkpoints_ucf101_capsules_i3d')
    training_specs = 'Multi_'+percent+'perRand_GausInterp_Dropout_PreTrainedCharades' 
    print("Save for :", training_specs)
    
    criterion = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)

    r = 0     
    num_epoch_to_eval = 2
    
    for e in range(1, N_EPOCHS + 1):
        name_prefix = '{}-ucf101-{}_8_ADAM_capsules_{}_RGB_Spread_BCE_epoch_{:03d}.pth'.format(MODEL_TYPE, LR, training_specs, e)
        save_path = os.path.join(save_root, name_prefix)
        train(model, train_data_loader, optimizer, criterion, e, r, save_path, short=False, masking=IS_MASKING)                
        
        if e > 10 and e % num_epoch_to_eval == 0:
            validate(model, val_data_loader, criterion, e, short=False)
        
        if LR_SCHEDULER:
            scheduler.step()


