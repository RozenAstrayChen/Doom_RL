import skimage.color
import skimage.transform
from basic.config import *
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque  # Ordered collection with ends
import tensorflow.contrib.slim as slim
import scipy.signal
# -*- coding: utf-8 -*-
'''
this is the basic object which is process some chores
'''


class Process:

    def __init__(self):
        pass

    def plot_save(self, rewards, name='total'):
        plt.figure(1)
        plt.clf()
        durations_t = torch.FloatTensor(rewards)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        name = './'+ str(name)+'_reward' + '.jpg'
        plt.savefig(name)

    def plot_save_loss(self, loss, a_loss, c_loss, name='rl'):
        plt.figure(2)
        plt.clf()
        loss = torch.FloatTensor(loss)
        a_loss = torch.FloatTensor(a_loss)
        c_loss = torch.FloatTensor(c_loss)
        plt.title('Loss')
        plt.xlabel('Episode*10')
        plt.ylabel('Duration')
        plt.plot(loss.numpy(), color='green', label='loss')
        plt.plot(a_loss.numpy(), color='red', label='a_loss')
        plt.plot(c_loss.numpy(), color='blue', label='c_loss')
        name = './'+ str(name)+'_loss' + '.jpg'
        plt.savefig(name)


    '''
    plt reward immediate
    '''

    def plot_durations(self, rewards):
        plt.figure(1)
        plt.clf()
        durations_t = torch.FloatTensor(rewards)
        plt.title('Training...')
        plt.xlabel('Episode*10')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        """
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        """
        plt.pause(0.001)  # pause a bit so that plots are updated

    '''
    plt loss
    '''
    def plot_loss(self, loss, a_loss, c_loss):
        plt.figure(2)
        plt.clf()
        loss = torch.FloatTensor(loss)
        a_loss = torch.FloatTensor(a_loss)
        c_loss = torch.FloatTensor(c_loss)
        plt.title('Loss')
        plt.xlabel('Episode*10')
        plt.ylabel('Duration')
        plt.plot(loss.numpy(), color='red', label='loss')
        plt.plot(a_loss.numpy(), color='green', label='a_loss')
        plt.plot(c_loss.numpy(), color='blue', label='c_loss')
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated

    '''
    Subsampling image and convert to numpy types
    '''

    def preprocess(self, frame):
        '''
        new_resolution = [resolution[0], resolution[1], 3]
        s = frame[10:-10,30:-30]
        #s = scipy.misc.imresize(s, new_resolution)
        print(s.shape)
        s = np.reshape(s ,[np.prod(s.shape)]) / 255.0
        s = skimage.transform.resize(
            s, resolution)
        s = s.astype(np.float32)

        s = self.frames_reshape(s)
        
        return s
        '''
        cropped_frame = frame[80:,:]
        #cropped_frame = frame[30:-10,30:-30]
        #cropped_frame = frame[15:-5,20:-20]
        
        # Normalize Pixel Values
        normalized_frame = cropped_frame/255.0
        
        # Resize
        preprocessed_frame = skimage.transform.resize(normalized_frame, resolution)
        preprocessed_frame = self.frames_reshape(preprocessed_frame)
        
        
        return preprocessed_frame
    
    def frames_reshape(self, frame):
        return frame.reshape([resolution_dim, resolution[0], resolution[1]])
    '''
  stack_frames
    '''
    def stack_frames(self, stacked_frames, state, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((resolution[0], resolution[1]), dtype=np.int) for i in range(stack_size)], maxlen=4)
            
            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(state)
            stacked_frames.append(state)
            stacked_frames.append(state)
            stacked_frames.append(state)
            
            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=0)
            
        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(state)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=0) 
        return stacked_state, stacked_frames

    '''
    save model
    '''

    def save_model(self, name, num, model):
        current_name = './result/' + str(num) + name
        torch.save(model, current_name)

    '''
    load model
    '''

    def load_model(self, name, num):
        current_name = './result' + str(num) + name 
        print("Loading model from: ", current_name)
        return torch.load(current_name)

    def show_action(self, index):
        if index == 0:
            print('<-- left look')
        elif index == 1:
            print('--> right look')
        else:
            print('^ run')

