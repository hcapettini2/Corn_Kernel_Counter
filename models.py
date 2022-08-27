### IMPORTS
import torch
import torch.nn as nn


###MODELS
class U_NET(nn.Module):
    """
    This is my simple UNET for understanding how it works
    """
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
                                     nn.Conv2d(in_channels  = 3,      # C1
                                             out_channels = 10,
                                             kernel_size  = 3,
                                             padding = 1,
                                             stride = 2),
                                     nn.BatchNorm2d(num_features=10), #same as out_channels
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels  = 10,      # C2
                                             out_channels = 20,
                                             kernel_size  = 3,
                                             padding = 1,
                                             stride = 2),
                                     nn.BatchNorm2d(num_features=20), #same as out_channels
                                     nn.ReLU(True),
                                     nn.Conv2d(in_channels  = 20,      # C3
                                             out_channels = 30,
                                             kernel_size  = 3,
                                             padding = 1,
                                             stride = 2),
                                     nn.BatchNorm2d(num_features=30), #same as out_channels
                                     nn.ReLU(True),
                                     )

        self.decoder = nn.Sequential(  
                                       nn.Conv2d(in_channels  = 30,      # C1
                                                  out_channels = 20,
                                                  kernel_size  = 3,
                                                  padding = 1,
                                                  stride = 1),
                                        nn.Upsample(scale_factor=2, mode='bilinear'), 
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels  = 20,      # C2
                                                  out_channels = 10,
                                                  kernel_size  = 3,
                                                  padding = 1,
                                                  stride = 1),
                                        nn.Upsample(scale_factor=2, mode='bilinear'), 
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels  = 10,      # C2
                                                  out_channels = 1,
                                                  kernel_size  = 3,
                                                  padding = 1,
                                                  stride = 1),
                                        nn.Upsample(size=(300,300), mode='bilinear'), # I force the image to have the same size as the maps
                                        nn.ReLU(True),
                                       )
        

        
    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y
    
    
    
    
    
class Map_CNN(nn.Module):
    """
    Attemp of UNET with pretrained backbone
    """
    def __init__(self, Pre_Trained_Encoder):
        super().__init__()
        
        self.encoder     = Pre_Trained_Encoder
         ### Encoder
        self.decoder = nn.Sequential(  
                       nn.Conv2d(in_channels  = 512,      # C1
                                  out_channels = 256,
                                  kernel_size  = 3,
                                  padding = 1,
                                  stride = 1),
                        nn.ReLU(True),                    
                        nn.Conv2d(in_channels  = 256,     # C2    
                                  out_channels = 128,
                                  kernel_size  = 3,
                                  padding = 1,
                                  stride = 1),
                        nn.ReLU(True), 
                        nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                        nn.Conv2d(in_channels  = 128,     #C3  
                                  out_channels = 64,
                                  kernel_size  = 3,
                                  padding = 1,
                                  stride = 1),
                        nn.ReLU(True),
                        nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                        nn.Conv2d(in_channels  = 64,      #C4  
                                  out_channels = 1,
                                  kernel_size  = 3,
                                  padding = 1,
                                  stride = 1),
                        nn.ReLU(True),
                        nn.Upsample(size=(300,300), mode='bilinear'), # I force the image to have the same size as the maps
                        nn.Conv2d(in_channels  = 1,       #C5
                                  out_channels = 1,
                                  kernel_size  = 1,
                                  padding = 0,
                                  stride = 1)
            )
        
    def forward(self, x,mode=None):
        y = self.encoder(x)
        y = self.decoder(y)
        return y
    
    
    
    
class Pre_Trained_UNET(nn.Module):
    def __init__(self, Pre_Trained_Encoder,Pre_Trained_Decoder):
        super().__init__()
        
        self.encoder     = Pre_Trained_Encoder
         ### Encoder
        self.decoder     = Pre_Trained_Decoder

        
    def forward(self, x,mode=None):
        y = self.encoder(x)
        y = self.decoder(y)
        return y