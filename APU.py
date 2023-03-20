import os
import time
import numpy as np
import random
from threading import Thread
from scipy.io import loadmat
from skvideo.io import vread
import pdb
import sys 

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from capsules_ucf101 import CapsNet
import torch.nn.functional as F

from cust_losses import *
import cv2
import pickle
from scipy.stats import norm

'''

Loads in videos for the 24 class subset of UCF-101.

The data is assumed to be organized in a folder (dataset_dir):
-Subfolder UCF101_vids contains the videos
-Subfolder UCF101_Annotations contains the .mat annotation files

UCF101DataLoader loads in the videos and formats their annotations on seperate threads.
-Argument train_or_test determines if you want to load in training or testing data
-Argument sec_to_wait determines the amount of time to wait to load in data
-Argument n_threads determines the number of threads which will be created to load data

Calling .get_video() returns (video, bboxes, label)
-The videos are in the shape (F, H, W, 3).
-The bounding boxes are loaded in as heat maps of size (F, H, W, 1) where 1 is forground and 0 is background.
-The label is an integer corresponding to the action class.

'''

class UCF101DataLoader(Dataset):
    'Generates UCF101-24 data'
    def __init__(self, name, clip_shape, batch_size, use_random_start_frame=False):
      self._dataset_dir = '/home/Datasets/UCF101'
      
      if name == 'train':
          self.vid_files = self.get_det_annots_prepared()          
          print("TRAINING EVAL MODE !!!!")
      else:
          print("Should not run in test mode!")
          exit()

      self._height = clip_shape[0]
      self._width = clip_shape[1]      
      self._batch_size = batch_size
      self._size = len(self.vid_files)
      self.indexes = np.arange(self._size)

        
    def get_det_annots_prepared(self):
        import pickle     
        training_annot_file = 'training_annots_multi_1perRand.pkl'        
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        return training_annotations
            
            
    def __len__(self):
        'Denotes the number of videos per epoch'
        return int(self._size)


    def __getitem__(self, index):
        v_name, anns = self.vid_files[index]
        clip, bbox_clip, label, annots = self.load_video(v_name, anns)
        if clip is None:
            print("Video none ", v_name)
            return None, None, None, None, None, None
        
        # Center crop
        frames, h, w, ch = clip.shape        
        margin_h = h - 224
        h_crop_start = int(margin_h/2)
        margin_w = w - 296
        w_crop_start = int(margin_w/2)
        
        clip = clip[:, h_crop_start:h_crop_start+224, w_crop_start:w_crop_start+296, :]
        bbox_clip = bbox_clip[:, h_crop_start:h_crop_start+224, w_crop_start:w_crop_start+296, :]
        clip_resize = np.zeros((frames, self._height, self._width, ch))
        bbox_resize = np.zeros((frames, self._height, self._width, 1), dtype=np.uint8)
        for i in range(frames):
            img = clip[i]
            img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
            clip_resize[i] = img / 255.
            
            bbox = bbox_clip[i, :, :, :]
            bbox = cv2.resize(bbox, (224, 224), interpolation=cv2.INTER_NEAREST)
            bbox_resize[i, bbox>0, :] = 1
        
        return v_name, anns, clip_resize, bbox_resize, label, annots


    def load_video(self, video_name, annotations):
        video_dir = os.path.join(self._dataset_dir, 'UCF101_Videos/%s.avi' % video_name)
        try:
            video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
        except:
            return None, None, None, None
            
        # creates the bounding box annotation at each frame
        n_frames, h, w, ch = video.shape
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label = -1
        
        multi_frame_annot = []
        for ann in annotations:
            start_frame, end_frame, label = ann[0], ann[1], ann[2]      # Label is from 0 in annotations
            multi_frame_annot.extend(ann[4])
            for f in range(start_frame, min(n_frames, end_frame+1)):
                try:
                    x, y, w, h = ann[3][f-start_frame]
                    bbox[f, y:y+h, x:x+w, :] = 1
                except:
                    print('ERROR LOADING ANNOTATIONS')
                    print(start_frame, end_frame)
                    print(video_dir)
                    exit()
        multi_frame_annot = list(set(multi_frame_annot))
        
        return video, bbox, label, multi_frame_annot


def get_thresholded_arr(i_arr, threshold = 0.5):
    # b x 1 x h x w x 1  (FG)
    # b x 1 x h x w x 25 (CLS)
    arr = np.copy(i_arr)
    if arr.shape[-1] > 1:
        arr_max = (arr == np.max(arr,-1,keepdims=True)).astype(float)
        arr *= arr_max
        arr[arr>0] = 1. 
    else:
        arr[arr>threshold] = 1.
        arr[arr<=threshold] = 0.  
    return arr


def get_uncertainty_logx(frame):
    
    frame_th = get_thresholded_arr(frame, threshold = 0.4)
    if frame_th.sum() == 0:
        return 1000.
    frame_th = frame_th.astype(np.bool)
    frame[frame == 0] = 1e-8
    frame = -np.log(frame)
    uncertainty = frame[frame_th].sum() / frame_th.sum()
    return uncertainty
    
    

if __name__ == '__main__':
    name='train'
    clip_shape=[224,224]
    channels=3
    batch_size = 1
    select_percent = 0.05   # % of total annots to add to previous file (so 15% added to 5% will be 20% total)
    
    # Load the trained model for next iteration of APU based frame selection
    # Set trained model file here 
    model_file_path = './trained/active_learning/checkpoints_ucf101_capsules_i3d/'
    
    model = CapsNet()
    model.load_previous_weights(model_file_path)
    print("Model loaded from: ", model_file_path)
    
    model = model.to('cuda')
    model.eval()
    model.training = False
    
    # Enable dropout in eval mode 
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
    
    dataloader = UCF101DataLoader(name, clip_shape, batch_size)
    
    iou_thresh = np.arange(0, 1, 0.1)
    frame_tp = np.zeros((24, iou_thresh.shape[0]))
    frame_fp = np.zeros((24, 1))
    
    clip_span = 16
    num_vids = len(dataloader)
    clip_batch_size = 14
    num_forward_passes = 10
    
    print("Total vids: ", num_vids)
    new_training_annotations = []
    
    with torch.no_grad():
        for i in range(num_vids):
            v_name, anns, video, bbox, label, annots = dataloader.get_item(i)
            if video is None:
                continue
                
            num_frames = video.shape[0]
            if num_frames == 0:
                print("Video has no frames.")
                continue
            
            vid_scores = np.zeros((num_frames)) + 1000
            # prepare batches of this video, get results from model, stack np arrays for results 
            batches = 0
            bbox_pred_fg = np.zeros((num_frames, clip_shape[0], clip_shape[1], 1))
            while True:
                batch_frames = np.zeros((1,8,224,224,3))
                for j in range(8):
                    ind = (batches * clip_span) + (j * 2)
                    if ind >= num_frames:
                        if j > 0:
                            batch_frames[0,j] = batch_frames[0,j-1]
                    else:
                        batch_frames[0,j] = video[ind]
                
                data = np.transpose(np.array(batch_frames), [0, 4, 1, 2, 3])
                data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
                action_tensor = np.ones((len(batch_frames),1), np.int) * label
                action_tensor = torch.from_numpy(action_tensor).cuda()
                
                segmentation_np = np.zeros((len(batch_frames), 1, 8, clip_shape[0], clip_shape[1]))
                for j in range(num_forward_passes):
                    segmentation, pred = model(data, action_tensor)
                    segmentation = F.sigmoid(segmentation)
                    segmentation_np += segmentation.cpu().data.numpy()   # B x C x F x H x W -> B x 1 x 8 x 224 x 224
                segmentation_np = segmentation_np / num_forward_passes
                segmentation_np = np.transpose(segmentation_np, [0, 2, 3, 4, 1]) 
                
                output_fg = segmentation_np[0]      # F x H x W x C
                output_fg = np.repeat(output_fg, 2, axis=0)
                
                end_idx = (batches+1) * clip_span
                if end_idx > num_frames:
                    end_idx = num_frames
                start_idx = batches * clip_span
                bbox_pred_fg[start_idx : end_idx] = output_fg[0:(end_idx - start_idx)]
                
                if end_idx >= num_frames:
                    break
                    
                batches += 1
            
            for f_idx in range(num_frames):
                if f_idx in annots:
                    vid_scores[f_idx] = 10000.   # Already in annot list 
                    continue
                    
                output_fg = bbox_pred_fg[f_idx]
                if np.sum(output_fg) <= 0.1:
                    vid_scores[f_idx] = 10000.   # Make very low to ignore in selection time 
                    continue
                
                # cost function
                uncertainty_score = get_uncertainty_logx(output_fg)
                closest_frame_dist = np.min(np.abs(np.array(annots)-f_idx))
                frame_score = uncertainty_score - (norm.pdf(closest_frame_dist, 0, 8/3.) / norm.pdf(0, 0, 8/3.))                
                vid_scores[f_idx] = frame_score     # Higher is better
            
            len_ann = np.count_nonzero(np.sum(bbox,(1,2,3)))
            select_frames = int(np.round(len_ann * select_percent))
            if select_frames < 1:
                select_frames = 1
            selected_frame = []
            
            start_frame = anns[0][0]
            end_frame = anns[0][1]
            vid_scores_annotated = np.ones((num_frames)) * -10000.
            vid_scores_annotated[start_frame:end_frame] = vid_scores[start_frame:end_frame]
            
            dist_norm_mask = norm.pdf(np.arange(9)-4, 0, 8/3.) / norm.pdf(0, 0, 8/3.)
            
            # Rescore after each frame selection 
            for k in range(select_frames):
                sorted_idx = np.argsort(vid_scores_annotated)     # Sorts in ascending order
                selected_frame.append(sorted_idx[-1])             # Pick frame with highest score 
                vid_scores_annotated[sorted_idx[-1]] = -1000.     # Change score of selected frame 
                # Change score of nearby frames 
                range_low, range_high = max(sorted_idx[-1]-4,0), min(sorted_idx[-1]+5, num_frames)
                range_center = sorted_idx[-1]
                vid_scores_annotated[range_low:range_high] += dist_norm_mask[4-(range_center-range_low):4+(range_high-range_center)]
       
            combined_annots = selected_frame[0:select_frames]
            
            combined_annots.extend(annots)
            new_annotations = []
            for ann in anns:
                sf, ef, label, bboxes = ann[0], ann[1], ann[2], ann[3]
                new_frame_annots = []
                for j in range(len(combined_annots)):
                    if sf <= combined_annots[j] <= ef:
                        new_frame_annots.append(combined_annots[j])
                new_annotations.append((sf, ef, label, bboxes, new_frame_annots))
            new_training_annotations.append((v_name, new_annotations))
            
            
    # save new annotation file
    fp = './training_annots_5percent_from1_AL.pkl'
    with open(fp,'wb') as wid:
        pickle.dump(new_training_annotations, wid, protocol=4)
    print("Saved at ", fp)
    
    exit(0)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
         
      
      
      
      
