# Quartz cement segmentation
### Repo for segmentation of Quartz sandstone images
Quartz cementation is a crucial factor in controlling the petrophysical properties of sandstone reservoirs. However, reliable identification of sandstone cementation requires petrographic analysis of rock thin sections using a combination of optical light microscopy, backscattered electron (BSE), and cathodoluminescence (CL) images. The present repository present the first attempt to automate this process by identifying sandstone cement through convolutional neural networks (CNNs). We used a combination of BSE and CL images acquired from sandstone thin sections sourced from formations in the US, Israel, and the Netherlands. For each image pair we created a labelled mask with 4 classes: (i) quartz grains; (ii) quartz cement; (iii) porosity; and (iv) other phases. We developed a U-Net with a total of 10 layers: 5 layers for the contracting path and 5 layers for the expansive path. The model is trained to predict the segmentation mask for a given input of paired BSE and CL images. A high level of accuracy is achieved for quartz grains (91%), quartz cement (78%), and porosity (95%). By contrast, the other phase class was poorly predicted due to a combination of mineral heterogeneity and low representation.  
