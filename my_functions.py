import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import random
   
def train_epoch(model,device,dataloader, loss_fn, optimizer):
    """
    This function train the network for one epoch
    """
    # Train
    loss_list = []
    model.train()
    for x, y,_,_ in dataloader:
        # Move data to device
        x = x.to(device)
        y = y.to(device)
        # Forward
        y_pred = model.forward(x)
        # Compute loss
        loss = loss_fn(y_pred, y)
        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        #Updata weights
        optimizer.step()
        # Save batch loss
        loss_list.append(loss.detach().cpu().numpy())
    return np.mean(loss_list)

def validate_epoch(model,device, dataloader, loss_fn):
    """
    This function validate/test the network performance for one epoch of training
    """
    loss_list = []
    # Discable gradient tracking
    model.eval()
    with torch.no_grad():
        for x, y,_,_ in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model.forward(x)
            loss = loss_fn(y_pred, y)
            loss_list.append(loss.detach().cpu().numpy()) 
    return np.mean(loss_list)
       
def training_cycle(model, device, training_iterator, test_iterator, validation_dataset, loss_fn, optim, num_epochs,writer,
                       verbose= True):
    """
    This function train the network for a desired number of epochs it also test the network 
    reconstruction performance and make plots comparing the input image and the reconstructed one every 5 epochs.
    """
    #initialize the tensorboard writer
    #writer = SummaryWriter()
    #I keep track of losses for plots
    train_loss = []
    val_loss  = []
    i = 0
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        ### Training (use the training function)
        tr_l = train_epoch(model= model,
                           device=device, 
                           dataloader=training_iterator, 
                           loss_fn=loss_fn, 
                           optimizer=optim)
        train_loss.append(tr_l)
        writer.add_scalar("Loss/train", tr_l, epoch)
            
        ### Validation  (use the testing function)
        v_l = validate_epoch(model=model,
                             device=device, 
                             dataloader=test_iterator, 
                             loss_fn=loss_fn)
        val_loss.append(v_l)
        writer.add_scalar("Loss/validation", v_l, epoch)
        end_time = time.time()
        writer.flush()
        if verbose:
            print(f"\nEpoch: {epoch+1}/{num_epochs} -- Epoch Time: {end_time-start_time:.2f} s")
            print("---------------------------------")
            print(f"Train -- Loss: {tr_l:.3f}")
            print(f"Val -- Loss: {v_l:.3f}")
            if (i % 2 == 0): plot_progress(model,validation_dataset,epoch,device,keep_plots = False)
        i +=1 
    return train_loss, val_loss
    

    
def plot_progress(model,dataset,epoch,device,keep_plots = False):
    """
    This function plot the image we send to the autoencoder and the one returned by the
    network.
    """
    elements = [0,1,2,3]

    fig, axs = plt.subplots(3, 4, figsize=(12,6))
    fig.suptitle('Original images and reconstructed image (EPOCH %d)' % (epoch + 1),fontsize=15)
    fig.subplots_adjust(top=0.88)
    axs = axs.ravel()
    for i in range (4):
        img, label = dataset[elements[i]][0:2]
        img = img.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            predicted_map  = model.forward(img)
        # Plot the reconstructed image  
        axs[i].imshow(img[0].numpy().transpose(1, 2, 0))
        #axs[i].set_title(categories[label])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        axs[i+4].imshow(label[0].numpy())
        #axs[i+4].set_title('Reconstructed image')
        axs[i+4].set_xticks([])
        axs[i+4].set_yticks([])

        axs[i+8].imshow(predicted_map[0].numpy().transpose(1, 2, 0))
        axs[i+8].set_title(f"Map sum= {int(predicted_map[0].sum())}")
        #axs[i+8].set_title('Reconstructed image')
        axs[i+8].set_xticks([])
        axs[i+8].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    # Save figures
    if keep_plots:
        os.makedirs('./Img/Training_plots/autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
        fig.savefig('./Img/Training_plots/autoencoder_progress_%d_features/epoch_%d.svg' % (encoded_space_dim, epoch + 1), format='svg')
    plt.show()
    plt.close()





def plot_dataset_images_and_masks(dataset):
    elements = [i for i in range(5,10)]

    fig, axs = plt.subplots(2, 5, figsize=(12,6))
    fig.suptitle("",fontsize=15)
    fig.subplots_adjust(top=0.88)
    axs = axs.ravel()
    for i in range(5):
        idx = elements[i]
        img,mask,_,count = dataset[idx][0:4]
        axs[i].imshow(img.numpy().transpose(1, 2, 0))
        axs[i].set_title(f"Real count = {count}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i+5].imshow(mask.numpy().transpose(1, 2, 0))
        #axs[i+5].set_title(f"Map count= {int(mask.sum()/1.)}")
        axs[i+5].set_xticks([])
        axs[i+5].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    
def dataset_mean_std(dataset):
    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    total_pixels   = 0
    # loop through images
    for idx in range(len(dataset)):
        img = dataset[idx][0]
        psum    += img.sum(axis = [1, 2])
        psum_sq += (img ** 2).sum(axis = [1, 2])
        total_pixels += img.shape[1]*img.shape[2]
        
    # mean and std
    mean = psum / total_pixels
    var  = (psum_sq / total_pixels) - (mean ** 2)
    std  = torch.sqrt(var)
    
    return mean, std


def reconstruct_plot_images_masks(train_features, train_labels, shape,count):
    elements = [i for i in range(5,10)]

    fig, axs = plt.subplots(2, 5, figsize=(12,6))
    fig.suptitle("",fontsize=15)
    fig.subplots_adjust(top=0.88)
    axs = axs.ravel()
    for i in range(5):
        idx = elements[i]
        ### Move the images to the right shapes
        UP = nn.Upsample(size=(shape[0][idx],shape[1][idx]),mode='bilinear')
        img = train_features[idx]#.numpy().transpose(1, 2, 0)
        img = img[None, :]
        img = UP(img)
        img = img[0].numpy().transpose(1, 2, 0)

        label = train_labels[idx]#.numpy().transpose(1, 2, 0)
        label = label[None, :]
        label = UP(label)
        label = label[0].numpy().transpose(1, 2, 0)


        # Plot the reconstructed image
        axs[i].imshow(img)
        axs[i].set_title(f"Real count = {count[idx]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i+5].imshow(label)
        #axs[i+5].set_title(f"Map count= {round(label.sum()/1.)}")
        axs[i+5].set_xticks([])
        axs[i+5].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)  
    
def Non_reconstruct_plot_images_masks(train_features, train_labels, shape,count):
    elements = [i for i in range(5,10)]

    fig, axs = plt.subplots(2, 5, figsize=(12,6))
    fig.suptitle("",fontsize=15)
    fig.subplots_adjust(top=0.88)
    axs = axs.ravel()
    for i in range(5):
        idx = elements[i]
        ### Move the images to the right shapes
        img = train_features[idx].numpy().transpose(1, 2, 0)
        label = train_labels[idx].numpy().transpose(1, 2, 0)

        # Plot the reconstructed image
        axs[i].imshow(img)
        axs[i].set_title(f"Real count = {count[idx]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i+5].imshow(label)
        #axs[i+5].set_title(f"Map count= {round(label.sum()/1.)}")
        axs[i+5].set_xticks([])
        axs[i+5].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)  
    
def plot_dataset_data(dataset):
    elements = [random.randint(0,len(dataset)) for i in range(0,5)]

    fig, axs = plt.subplots(2, 4, figsize=(12,6))
    fig.subplots_adjust(top=0.88)
    axs = axs.ravel()
    for i in range (4):
        img, label = dataset[elements[i]][0:2]
        img = img.unsqueeze(0)
        # Plot the reconstructed image  
        axs[i].imshow(img[0].numpy().transpose(1, 2, 0))
        #axs[i].set_title(categories[label])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        axs[i+4].imshow(label[0].numpy())
        #axs[i+4].set_title('Reconstructed image')
        axs[i+4].set_xticks([])
        axs[i+4].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
    plt.close()