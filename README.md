# A Deep Learning Method for On-Ear Corn Kernel Counting
***

<p align="center">
    <img src="https://github.com/hcapettini2/Corn_Kernel_Counter/blob/main/Images/prediction_base.png" alt="Drawing" style="width: 500px"/>
</p>





## Authors

* [**Capettini Hilario**](https://github.com/hcapettini2) (University of Padua)





## Abstract
Crop monitoring and yield estimation provide farmers with important information to take decisions. To do that farmers have to manually count the number of kernels on ears which is a time consuming and prone to error activity given that each ear has between $200-600$ kernels. In this work we propose the usage of deep learning for counting the on-ear kernels by using images that can be taken with a phone. In particular we focus in the usage of a U-Net architecture with a VGG16 backbone for feature extraction. As publicly available data is scarce we first train on a wide dataset containing corn ears in different contexts and then fine tune on a small dataset containing pictures that a farmer could take. We show that after the tuning of the parameters the model achieve state of the art models results. We also observe that the proposed model produce accurate density maps that can be used for kernel localisation.
* The complete report of this project can be found [**here**](https://github.com/eigen-carmona/net-sci-project/blob/master/Report.pdf)

* The presentation of this project can be found [**here**](https://github.com/hcapettini2/Corn_Kernel_Counter/blob/main/Report.pdf)
