import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import random
from torchvision import transforms
import torchvision
from math import ceil
import torchvision.transforms.functional as TF


import dataset as customdataset
   
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
                       model_path,images_path,verbose= True):
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
        #Every 10 epochs save a prediction
        if (i % 20 == 0): 
            if len(validation_dataset)>4:
                plot_progress(model,validation_dataset,epoch,device,images_path,keep_plots = True)
        #Every 20 epochs save the model
        if (i % 20 == 0):
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optim.state_dict(), }

            torch.save(state, model_path)
        i +=1 
    return train_loss, val_loss
    

    
def plot_progress(model,dataset,epoch,device,images_path,keep_plots = False):
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
        axs[i].imshow(img[0].cpu().numpy().transpose(1, 2, 0))
        #axs[i].set_title(categories[label])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        axs[i+4].imshow(label[0].cpu().numpy())
        #axs[i+4].set_title('Reconstructed image')
        axs[i+4].set_xticks([])
        axs[i+4].set_yticks([])

        axs[i+8].imshow(predicted_map[0][0].cpu().numpy())#.transpose(1, 2, 0))
        axs[i+8].set_title(f"Map sum= {int(predicted_map[0].sum())}")
        #axs[i+8].set_title('Reconstructed image')
        axs[i+8].set_xticks([])
        axs[i+8].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    # Save figures
    if keep_plots:
        fig.savefig(images_path+f'training_{epoch}.svg', format='svg')
    #plt.show()
    plt.close()




"""
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
        axs[i+5].imshow(mask[0].numpy())
        #axs[i+5].set_title(f"Map count= {int(mask.sum()/1.)}")
        axs[i+5].set_xticks([])
        axs[i+5].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
"""    
    
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
        label = label[0][0].numpy()#.transpose(1, 2, 0)


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
        label = train_labels[idx][0].numpy()#.transpose(1, 2, 0)

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

        axs[i+4].imshow(label[0].numpy(),cmap="viridis")
        #axs[i+4].set_title('Reconstructed image')
        axs[i+4].set_xticks([])
        axs[i+4].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
    plt.close()
    
"""    
def transforms_definitions_full_image(img_size):
    ### TRANSFORMS
    train_img_transforms = transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                            torchvision.transforms.RandomAutocontrast(p=0.4),
                                            torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0.1, hue=0.1),
                                            transforms.Resize(img_size),
                                            ])

    train_geometric_transforms = transforms.Compose([
                                               #customdataset.padding(384,640),
                                               #customdataset.Random_Cropping(384,640),
                                               customdataset.MyRotations(),
                                              ])
    test_validation_geometric_transforms = transforms.Compose([
                                               #customdataset.padding(384,640),
                                               #customdataset.Random_Cropping(384,640),
                                              ])
    test_validation_img_transforms = transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                         transforms.Resize(img_size),
                                                         ])

    lbl_transforms = transforms.Compose([
                                         torchvision.transforms.ToTensor(),
                                         transforms.Resize(img_size),
                                         ])    
    return train_img_transforms,train_geometric_transforms,test_validation_geometric_transforms,test_validation_img_transforms,lbl_transforms
"""

def transforms_definitions_cropped_image(pd_1=320,pd_2=320): 
    train_img_transforms = transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                            #torchvision.transforms.RandomAutocontrast(p=0.4),
                                            #torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0.1, hue=0.1),
                                            #transforms.Resize((320,320)),
                                            ])

    train_geometric_transforms = transforms.Compose([
                                               customdataset.padding(pd_1,pd_2),
                                               customdataset.Random_Cropping(pd_1,pd_2),
                                               customdataset.MyRotations(),
                                              ])
    lbl_transforms = transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              #transforms.Resize((320,320)),
                                              ])

    validation_img_transforms = transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                   #transforms.Resize((320,320)),
                                                   ])

    validation_geometric_transforms = transforms.Compose([
                                                          customdataset.padding(pd_1,pd_2),
                                                          customdataset.Random_Cropping(pd_1,pd_2),
                                                         ])

    # validation_lbl_transforms = transforms.Compose([
    #                                                torchvision.transforms.ToTensor(),
    #                                                #transforms.Resize((320,320)),
    #                                                ])

    test_img_transforms = transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                             #transforms.Resize((320,320)),
                                             ])

    test_geometric_transforms = transforms.Compose([
                                                   #customdataset.padding(384,640),
                                                   #customdataset.Random_Cropping(384,640),
                                                   ])

    # test_lbl_transforms = transforms.Compose([
    #                                                torchvision.transforms.ToTensor(),
    #                                                #transforms.Resize((320,320)),
    #                                                ])
    return train_img_transforms,train_geometric_transforms,lbl_transforms,validation_img_transforms,validation_geometric_transforms,test_img_transforms,test_geometric_transforms
"""
def plot_prediction_full_image(inputs, labels, shapes, counts,outputs,re_scale,images_path):
    elements = [5,6,7,8,9]#random.sample([i for i in range(len(test_dataset))], 5)

    fig, axs = plt.subplots(3, 5, figsize=(12,6))
    fig.suptitle("",fontsize=15)
    fig.subplots_adjust(top=0.88)
    axs = axs.ravel()
    for i in range(5):
        idx = elements[i]
        ### Move the images to the right shapes
        UP = nn.Upsample(size=(shapes[0][idx],shapes[1][idx]),mode='bilinear')
        img = inputs[idx]#.numpy().transpose(1, 2, 0)
        img = img[None, :]
        img = UP(img)
        img = img[0].numpy().transpose(1, 2, 0)

        label = labels[idx]#.numpy().transpose(1, 2, 0)
        label = label[None, :]
        label = UP(label)
        label = label[0].numpy().transpose(1, 2, 0)

        map_ = outputs[idx]#.numpy().transpose(1, 2, 0)
        map_ = map_[None, :]
        map_ = UP(map_)
        map_ = map_[0].numpy().transpose(1, 2, 0) 

        # Plot the reconstructed image
        axs[i].imshow(img)
        axs[i].set_title(f"Real count = {counts[idx]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i+5].imshow(label)
        axs[i+5].set_title(f"Map count= {int(label.sum()/re_scale)}")
        axs[i+5].set_xticks([])
        axs[i+5].set_yticks([])
        axs[i+10].imshow(map_)
        axs[i+10].set_title(f"Map count= {int(map_.sum()/re_scale)}")
        axs[i+10].set_xticks([])
        axs[i+10].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)  
    fig.savefig(images_path+f'prediction.svg', format='svg')
    #plt.show()
    plt.close()
"""    
"""
def evaluation_metric_full_image(test_dataset,shapes,labels,outputs,re_scale):
    label_count_list = []
    map_count_list = []
    for idx in [i for i in range(len(test_dataset))]:
        ### Move the images to the right shapes
        UP = nn.Upsample(size=(shapes[0][idx],shapes[1][idx]),mode='bilinear')

        label = labels[idx]#.numpy().transpose(1, 2, 0)
        label = label[None, :]
        label = UP(label)
        label = label[0].numpy().transpose(1, 2, 0)
        label_count = (label.sum()/re_scale)
        label_count_list.append(label_count)

        map_ = outputs[idx]#.numpy().transpose(1, 2, 0)
        map_ = map_[None, :]
        map_ = UP(map_)
        map_ = map_[0].numpy().transpose(1, 2, 0) 
        map_count = (map_.sum()/re_scale)
        map_count_list.append(map_count)
        
    mae = np.sum(np.abs(np.array(label_count_list)-np.array(map_count_list)))/len(map_count_list)
    print("MAE = ", mae)

    rmse = np.sqrt(np.sum(np.abs(np.array(label_count_list)-np.array(map_count_list))**2)/len(map_count_list))
    print("RMSE = ", rmse)

    mape = np.sum(np.abs(np.array(label_count_list)-np.array(map_count_list))/np.array(label_count_list)) /len(map_count_list)
    print("MAPE = ", mape)
    
    return mae, rmse, mape,label_count_list,map_count_list
"""


def image_disector(original_image,window_h,window_w):
# Retrieve original image, shapes and estimate the number of required windows
#original_image = test_dataset[im][0]
    image_h  =  original_image.shape[1]
    image_w  =  original_image.shape[2]
    n_h = ceil(image_h/window_h)
    n_w = ceil(image_w/window_w)

    # Padd image 
    padded_image = customdataset.padding(n_h*window_h,n_w*window_w)([original_image,original_image])[0]
    # Disect image into windows of required size
    image_sections = []
    for i in [(window_h*k) for k in range(n_h)]:
        for j in [(window_w*k) for k in range(n_w)]:
            image_sections.append(TF.crop(padded_image,i,j,window_h,window_w))

    return padded_image.shape,image_sections


def map_reconstruction(original_image_shape,padded_image_shape,image_sections,window_h,window_w):
    image_h  =  original_image_shape[1]
    image_w  =  original_image_shape[2]
    padded_image_h  =  padded_image_shape[1]
    padded_image_w  =  padded_image_shape[2]
    n_h = ceil(image_h/window_h)
    n_w = ceil(image_w/window_w)
    
    #Recontruct image
    stripes = []
    for i in range(n_h):
        stripes.append(torch.cat((image_sections[(i*n_w):((i+1)*n_w)]), 2))
    reconstructed_img = torch.cat(stripes, 1)
    # Erase padding
    x_ = int(abs((image_h-padded_image_h))/2)
    y_ = int(abs((image_w-padded_image_w))/2)
    
    return TF.crop(reconstructed_img,x_,y_,image_h,image_w) 

"""
def plot_prediction_patched_image(test_dataset,final_output,re_scale,images_path):

    elements = [random.randint(0,(len(test_dataset)-1))for i in range(0,5)]#[5,6,7,8,9]

    fig, axs = plt.subplots(3, 5, figsize=(15,12))
    fig.suptitle("",fontsize=15)
    fig.subplots_adjust(top=0.88)
    axs = axs.ravel()
    for i in range(5):
        idx = elements[i]
        ### Move the images to the right shapes
        img = test_dataset[idx][0].numpy().transpose(1, 2, 0)
        label = test_dataset[idx][1][0].numpy()
        map_ = final_output[idx].cpu().numpy()
        #map_[map_ <= 1e1] = 0

        # Plot the reconstructed image
        axs[i].imshow(img)
        #axs[i].set_title(f"Real count = {counts[idx]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i+5].imshow(label)
        axs[i+5].set_title(f"Map count= {int(label.sum()/re_scale)}")
        axs[i+5].set_xticks([])
        axs[i+5].set_yticks([])
        axs[i+10].imshow(map_)
        axs[i+10].set_title(f"Map count= {int(map_.sum()/re_scale)}")
        axs[i+10].set_xticks([])
        axs[i+10].set_yticks([])
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(images_path+f'prediction.svg', format='svg')
    #plt.show()
    plt.close()
"""    
"""
def evaluation_metric_patch_image(test_dataset,final_output,re_scale,model_path_metrics):
    label_count_list = []
    map_count_list = []
    for idx in [i for i in range(len(test_dataset))]:
        ### Move the images to the right shapes
        map_ = final_output[idx].cpu().numpy()

        label = test_dataset[idx][1][0].numpy()
        label_count = int(label.sum()/re_scale)
        label_count_list.append(label_count)

        map_ = final_output[idx].cpu().numpy()
        #map_[map_ <= 1e1] = 0         #threshold small values
        map_count = int(map_.sum()/re_scale)
        map_count_list.append(map_count)

    mae = np.sum(np.abs(np.array(label_count_list)-np.array(map_count_list)))/len(map_count_list)
    print("MAE = ", mae)

    rmse = np.sqrt(np.sum(np.abs(np.array(label_count_list)-np.array(map_count_list))**2)/len(map_count_list))
    print("RMSE = ", rmse)

    mape = np.sum(np.abs(np.array(label_count_list)-np.array(map_count_list))/np.array(label_count_list)) /len(map_count_list)
    print("MAPE = ", mape)
    df = pd.DataFrame(list(zip([mae], [rmse],[mape])),
                columns =["MAE", "RMSE","MAPE"])
    df.to_csv(model_path_metrics)
    df = pd.DataFrame(list(zip(label_count_list, map_count_list)),
                columns =["label_count_list", "map_count_list"])
    df.to_csv(model_path_metrics[:-4]+"_predictions.csv")
    return mae, rmse, mape,label_count_list,map_count_list 
    """