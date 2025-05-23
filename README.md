from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from CustomButton import TkinterCustomButton
import numpy as np
import scipy.io as sio
import os, glob, random
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from Fastonn import SelfONNTranspose1d as SelfONNTranspose1dlayer
from Fastonn import SelfONN1d as SelfONN1dlayer
from utils import ECGDataset, ECGDataModule,init_weights,TECGDataset,TECGDataModule
from GAN_Arch_details import Upsample,Downsample,CycleGAN_Unet_Generator,CycleGAN_Discriminator
from numpy import dot
from numpy.linalg import norm

main = Tk()
main.title("Blind ECG Restorstion by Super Neuron Model-GAN's") 
main.geometry("1300x1200")

global filename, model, accuracy

def deleteDirectory():
    filelist = [ f for f in os.listdir('test_outputs/') if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join('test_outputs', f))

def loadModel():
    global model
    text.delete('1.0', END)
    model = CycleGAN_Unet_Generator()
    checkpoint =torch.load("model/model_weights_16NQ3.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    text.insert(END,"Super Neuron  Model Loaded")

def uploadNoisyImage():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "Dataset")
    text.insert(END,"Matlab ECG Noise Image Loaded\n\n")

def GanPredict(name):
    global accuracy
    accuracy = []
    gan_outputs=sio.loadmat("test_outputs/"+name+"_gan_outputs.mat")
    real_sig=sio.loadmat("test_outputs/"+name+"_real_sig.mat")
    gan_outputs=gan_outputs["gan_outputs"]
    real_sig=real_sig["real_sig"]
    ab_beats=sio.loadmat("mats/R"+name+".mat")
    S=ab_beats["S"]
    V=ab_beats["V"]    
    S = pd.DataFrame(data =S)
    V = pd.DataFrame(data =V)    
    win_size=int(np.floor(len(gan_outputs)/4000))    
    gan_outputs=gan_outputs[:win_size*4000]  
    real_sig=real_sig[:win_size*4000]    
    gan_outputs1=gan_outputs.reshape(win_size,4000)   
    real_sig1=real_sig.reshape(win_size,4000)    
    S_arr=np.zeros(((win_size*4000),1))
    V_arr=np.zeros(((win_size*4000),1))
    S_arr[S.values]=1
    V_arr[V.values]=1
    S_arr=S_arr.reshape(win_size,4000)
    V_arr=V_arr.reshape(win_size,4000)
    for i in range(0,4):
        gan_outputs=gan_outputs1[i,:]
        real_sig=real_sig1[i,:]
        V=V_arr[i,:]
        S=S_arr[i,:]
        acc = dot(gan_outputs, real_sig)/(norm(gan_outputs) * norm(real_sig))
        accuracy.append(acc)
        if i == 0:
            time_axis=np.arange(i*4000,(i+1)*4000)/400
            a=plt.figure()
            a.set_size_inches(12, 10)
            ax=plt.subplot(211)
            major_ticksx = np.arange(10*i, 10*(i+1),1 )
            minor_ticksx = np.arange(10*i, 10*(i+1), 0.25)
            major_ticksy = np.arange(-1.5, 1.5,0.3 )
            minor_ticksy = np.arange(-1.5, 1.5, 0.075)            
            ax.set_xticks(major_ticksx)
            ax.set_xticks(minor_ticksx, minor=True)          
            ax.set_yticks(major_ticksy)
            ax.set_yticks(minor_ticksy, minor=True)
        
            plt.plot(time_axis,real_sig,linewidth=0.7,color='k')
            plt.scatter(time_axis[real_sig*S!=0], real_sig[real_sig*S!=0], c='#bcbd22',  s=100,marker=(5, 1), alpha=0.5)
            plt.scatter(time_axis[real_sig*V!=0],real_sig[real_sig*V!=0], c='#2ca02c',s=100, marker=(5, 1), alpha=0.5)     
            ax.grid(which='minor', alpha=0.2,color='r')
            ax.grid(which='major', alpha=0.5,color='r')           
            plt.title("Original ECG Segment", fontsize=15)
            plt.axis([10*i, 10*(i+1),-1.5, 1.5])
            plt.xlabel('Time (seconds)', fontsize=13)
            plt.ylabel('Amplitude', fontsize=13)
            ax2=plt.subplot(212, sharex = ax)
            # Major ticks every 20, minor ticks every 5
            major_ticksx = np.arange(10*i, 10*(i+1),1 )
            minor_ticksx = np.arange(10*i, 10*(i+1), 0.25)          
            major_ticksy = np.arange(-1.5, 1.5,0.3 )
            minor_ticksy = np.arange(-1.5, 1.5, 0.075)         
            ax2.set_xticks(major_ticksx)
            ax2.set_xticks(minor_ticksx, minor=True)         
            ax2.set_yticks(major_ticksy)
            ax2.set_yticks(minor_ticksy, minor=True)
            plt.plot(time_axis,gan_outputs,linewidth=0.7,color='k')
            plt.scatter(time_axis[gan_outputs*S!=0], gan_outputs[gan_outputs*S!=0], c='#bcbd22',  s=100,marker=(5, 1), alpha=0.5)
            plt.scatter(time_axis[gan_outputs*V!=0],gan_outputs[gan_outputs*V!=0], c='#2ca02c',s=100, marker=(5, 1), alpha=0.5)    
            ax2.grid(which='minor', alpha=0.2,color='r')
            ax2.grid(which='major', alpha=0.5,color='r')           
            plt.title("Super Neuron Model-GAN", fontsize=15)
            plt.xlabel('Time (seconds)', fontsize=13)
            plt.ylabel('Amplitude', fontsize=13)
            plt.axis([10*i, 10*(i+1),-1.5, 1.5])         
            plt.tight_layout(pad=1.0)
    plt.show()
    
def predictSignals():
    deleteDirectory()
    name = os.path.basename(filename).split("_")
    print(name[0])
    GanPredict(name[0])
    
def graph():
    global accuracy
    text.delete('1.0', END)
    text.insert(END,"Super Neuron GANs Performance\n")  ############################################################
    text.insert(END,"Accuracy  : "+str(accuracy[0])+"\n")
    text.insert(END,"Precision : "+str(accuracy[1])+"\n")
    text.insert(END,"Recall    : "+str(accuracy[2])+"\n")
    text.insert(END,"FSCORE    : "+str(accuracy[3])+"\n")
    text.update_idletasks()
    height = accuracy
    bars = ('Accuracy','Precision','Recall','FSCORE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Metric Name")
    plt.ylabel("Metric Values")
    plt.title("Super Neuron Model-GANs Performance Graph")
    plt.show()

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Blind ECG Restoration by Super Neuron Model-GANs')##############################################################3
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

modelButton = TkinterCustomButton(text=" Load Super Neuron GANs Model", width=400, corner_radius=5, command=loadModel) #########################
modelButton.place(x=50,y=100)

uploadButton = TkinterCustomButton(text="Upload Noisy ECG Signal", width=300, corner_radius=5, command=uploadNoisyImage)
uploadButton.place(x=470,y=100)

predictButton = TkinterCustomButton(text="Predict Quality Signals", width=300, corner_radius=5, command=predictSignals)
predictButton.place(x=790,y=100)

graphButton = TkinterCustomButton(text="Metric Analysis Graph", width=300, corner_radius=5, command=graph)
graphButton.place(x=50,y=150)

closeButton = TkinterCustomButton(text="Exit", width=300, corner_radius=5, command=close)
closeButton.place(x=470,y=150)


font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()
