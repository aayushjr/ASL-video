import os
import time
import numpy as np
import random
from threading import Thread
from scipy.io import loadmat
from skvideo.io import vread
import pdb
import torch
from torch.utils.data import Dataset
import pickle
import cv2
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
    'Prunes UCF101-24 data'
    def __init__(self, name, clip_shape, batch_size, percent='5', use_random_start_frame=False):
      self._dataset_dir = './data/UCF101'       # Path to UCF101 
      
      if name == 'train':
          self.vid_files = self.get_det_annots_prepared(percent=percent)
          self.shuffle = True
          self.name = 'train'
      else:
          self.vid_files = self.get_det_annots_test_prepared()
          self.shuffle = False
          self.name = 'test'

      self._use_random_start_frame = use_random_start_frame
      self._height = clip_shape[0]
      self._width = clip_shape[1]
      #self._channels = channels
      self._batch_size = batch_size
      self._size = len(self.vid_files)
      self.indexes = np.arange(self._size)
            

    def get_det_annots_prepared(self, percent='5'):
        import pickle
        
        training_annot_file = 'training_annots_multi.pkl'
        #training_annot_file = 'training_annots_multi_'+percent+'perRand.pkl'
         
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        
        training_annotations.extend(training_annotations)
        
        return training_annotations
        
        
    def get_det_annots_test_prepared(self):
        import pickle    
        with open('testing_annots.pkl', 'rb') as ts_rid:
            testing_annotations = pickle.load(ts_rid)
            
        return testing_annotations    
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        depth = 8
        video_rgb = np.zeros((depth, self._height, self._width, 3))
        label_cls = np.zeros((depth, self._height, self._width, 1))     # FG/BG or actor (person only) for this dataset
        mask_cls = np.zeros((depth, self._height, self._width, 1))
        
        v_name, anns = self.vid_files[index]        
        clip, bbox_clip, label, annot_frames = self.load_video(v_name, anns)        
        if clip is None:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
            mask_cls = torch.from_numpy(mask_cls)
            sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]),'mask_cls':mask_cls}
            return sample

        vlen, clip_h, clip_w, _ = clip.shape
        vskip = 2
        
        if len(annot_frames) == 1:
            selected_annot_frame = annot_frames[0]
        else:
            if len(annot_frames) <= 0:
                print('annot index error for', v_name, ', ', len(annot_frames), ', ', annot_frames)
                video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
                video_rgb = torch.from_numpy(video_rgb)            
                label_cls = np.transpose(label_cls, [3, 0, 1, 2])
                label_cls = torch.from_numpy(label_cls)
                mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
                mask_cls = torch.from_numpy(mask_cls)
                sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]),'mask_cls':mask_cls}
                return sample
            annot_idx = np.random.randint(0,len(annot_frames))
            selected_annot_frame = annot_frames[annot_idx]
        start_frame = selected_annot_frame - int((depth * vskip)/2)
        if start_frame < 0:
            vskip = 1
            start_frame = selected_annot_frame - int((depth * vskip)/2)
            if start_frame < 0:
                start_frame = 0
                vskip = 1
        if selected_annot_frame >= vlen:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
            mask_cls = torch.from_numpy(mask_cls)
            sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]),'mask_cls':mask_cls}
            return sample
        if start_frame + (depth * vskip) >= vlen:
            start_frame = vlen - (depth * vskip)
        
        #Random start frame
        if self._use_random_start_frame:
            random_start_frame_btm = selected_annot_frame - (depth * vskip) + 2
            if random_start_frame_btm < 0:
                random_start_frame_btm = 0
            random_start_frame_top = selected_annot_frame - 2
            if random_start_frame_top <= random_start_frame_btm:
                random_start_frame = start_frame
            else:
                random_start_frame = np.random.randint(random_start_frame_btm, random_start_frame_top)
            if random_start_frame + (depth * vskip) >= vlen:
                random_start_frame = vlen - (depth * vskip)
            start_frame = random_start_frame
        
        span = (np.arange(depth)*vskip)
        span += start_frame
        video = clip[span]
        bbox_clip = bbox_clip[span]
        closest_fidx = np.argmin(np.abs(span-selected_annot_frame))
        
        sigma = 4/3. # 2.5
        gaus_vals = norm.pdf(np.arange(-4,5), 0, sigma)
        gaus_vals = gaus_vals / np.max(gaus_vals)
        gaus_mid = 5
        if self.name == 'train':
            #start_pos = np.random.randint(0,16)
            start_pos_h = np.random.randint(0,clip_h - 224) #self._height)
            start_pos_w = np.random.randint(0,clip_w - 296) #self._width)
        else:
            # center crop for validation
            start_pos_h = int((clip_h - 224) / 2)
            start_pos_w = int((clip_w - 296) / 2)
        
        final_gaus_mask = np.zeros((depth))
        
        for j in range(video.shape[0]):
            img = video[j]
            img = img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+296,:]
            img = cv2.resize(img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            img = img / 255.
            video_rgb[j] = img
            
            valid_frame = False
            if vskip == 2:
                if span[j] in annot_frames or span[j]+1 in annot_frames:
                    valid_frame = True
            elif vskip == 1:
                if span[j] in annot_frames:
                    valid_frame = True
                    
            bbox_img = bbox_clip[j]
            bbox_img = bbox_img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+296,:]
            bbox_img = cv2.resize(bbox_img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            label_cls[j, bbox_img > 0, 0] = 1.
            
            if valid_frame:
                gaus_range = [np.max((0,gaus_mid-j-1)), np.min((9, gaus_mid+depth-j-1))]
                frame_range = [np.max((0, j-gaus_vals.shape[0]//2)), np.min((8, j+gaus_vals.shape[0]//2+1))]
                
                mask_vals = gaus_vals[gaus_range[0]:gaus_range[1]]
                
                # Max 
                new_mask_vals = final_gaus_mask[frame_range[0]:frame_range[1]]
                new_mask_vals[mask_vals>new_mask_vals] = (0*new_mask_vals[mask_vals>new_mask_vals]) + mask_vals[mask_vals>new_mask_vals]
                final_gaus_mask[frame_range[0]:frame_range[1]] = np.copy(new_mask_vals)
        
        
        # Max 
        #print(final_gaus_mask)
        final_gaus_mask = final_gaus_mask.reshape((final_gaus_mask.shape[0], 1, 1))
        mask_cls[:, :, :, 0] = final_gaus_mask
        
        mask_cls[mask_cls>1.] = 1.
        
        
        #if np.random.randint(0,2) == 1:
        if index<(self._size/2):        # If first half, horizontal flip. Assumes data duplicated beforehand in load 
            video_rgb[:,:,:,:] = video_rgb[:,:,::-1,:]
            label_cls[:,:,:,:] = label_cls[:,:,::-1,:]
            mask_cls[:,:,:,:] = mask_cls[:,:,::-1,:]
        
        video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
        video_rgb = torch.from_numpy(video_rgb)
        
        label_cls = np.transpose(label_cls, [3, 0, 1, 2])
        label_cls = torch.from_numpy(label_cls)
        
        mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
        mask_cls = torch.from_numpy(mask_cls)
        
        action_tensor = torch.Tensor([label])        
        
        sample = {'data':video_rgb,'segmentation':label_cls,'action':action_tensor, 'mask_cls': mask_cls}
        return sample


    def load_video(self, video_name, annotations):
        video_dir = os.path.join(self._dataset_dir, 'UCF101_Videos/%s.avi' % video_name)
        
        try:
            video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
        except:
            print('Error:', str(video_dir))
            return None, None, None, None
        
        # creates the bounding box annotation at each frame
        n_frames, h, w, ch = video.shape
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label = -1
        #multi frame mode
        annot_idx = 0
        if len(annotations) > 1:
            annot_idx = np.random.randint(0,len(annotations))
        
        if self.name == 'train':
            multi_frame_annot = annotations[annot_idx][4]
        else:
            multi_frame_annot = []
        
        bbox_annot = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        for ann in annotations:
            if not self.name == 'train':
                multi_frame_annot.extend(ann[4])
            start_frame, end_frame, label = ann[0], ann[1], ann[2]
            collect_annots = []
            for f in range(start_frame, min(n_frames, end_frame+1)):
                try:
                    x, y, w, h = ann[3][f-start_frame]
                    bbox[f, y:y+h, x:x+w, :] = 1
                    if f in ann[4]:
                        collect_annots.append([x,y,w,h])
                except:
                    print('ERROR LOADING ANNOTATIONS')
                    print(start_frame, end_frame)
                    print(video_dir)
                    exit()
            
            select_annots = ann[4]
            select_annots.sort()
            if len(collect_annots) == 0:
                continue
            [x, y, w, h] = collect_annots[0]
            if len(collect_annots) == 1:
                bbox_annot[start_frame:end_frame, y:y+h, x:x+w, :] = 1
            else:
                bbox_annot[start_frame:select_annots[0], y:y+h, x:x+w, :] = 1
                for i in range(len(collect_annots)-1):
                    frame_diff = select_annots[i+1] - select_annots[i]
                    if frame_diff > 1:
                        [x, y, w, h] = collect_annots[i]
                        pt1 = np.array([x, y, x+w, y+h])
                        [x, y, w, h] = collect_annots[i+1]
                        pt2 = np.array([x, y, x+w, y+h])
                        points = np.linspace(pt1, pt2, frame_diff).astype(np.int32)
                        for j in range(points.shape[0]):
                            [x1, y1, x2, y2] = points[j]
                            bbox_annot[select_annots[i]+j, y1:y2, x1:x2, :] = 1
                    else:
                        [x, y, w, h] = collect_annots[i]
                        bbox_annot[select_annots[i], y:y+h, x:x+w, :] = 1
                [x, y, w, h] = collect_annots[-1]
                bbox_annot[select_annots[-1]:end_frame, y:y+h, x:x+w, :] = 1
            
        multi_frame_annot = list(set(multi_frame_annot))
        if self.name == 'train':
            return video, bbox_annot, label, multi_frame_annot
        else:
            return video, bbox, label, multi_frame_annot


if __name__ == '__main__':
    import imageio    
    name='train'
    clip_shape=[224,224]
    channels=3
    batch_size = 5
    dataloader = UCF101DataLoader(name, clip_shape, batch_size, '5', False)
    print(len(dataloader))
    #exit()
    index = 0
    while index<2:
        d1 = dataloader.__getitem__(index)
        
        print(frm_idx)
        
        with imageio.get_writer('./results/{:02d}_gt.gif'.format(index), mode='I') as writer:
            for i in range(clip.shape[1]):
                image = (clip[0,i]*255).astype(np.uint8)
                writer.append_data(image) 
        
        print("Done for ", index)
        index += 1         
      
      
      
      
