'''Function that takes images and masks as input and returns images and masks in a train and test sets.
   By default it operates a data augmentation on the training set'''

import os
import numpy as np
import cv2
import tifffile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from patchify import patchify
import albumentations as A

def data_prep(images_folder="Data/BSE_CL_images/", masks_folder="Data/Masks/", data_augmentation=True, crop_h=512, crop_w=512, n_channels=2, aug_level=5):
    # Create empty lists for input data
    img_inp = []
    lbl_inp = []

    # Loop over all image and label files in the directories
    for img_name, lbl_name in zip(sorted(os.listdir(images_folder)), sorted(os.listdir(masks_folder))):
        # Load and preprocess image
        img_tmp = tifffile.imread(os.path.join(images_folder, img_name))
        img_tmp_rs = np.moveaxis(img_tmp, 0, -1)[:, :, :n_channels]
        img_tmp_norm = np.empty_like(img_tmp_rs)
        for i in range(n_channels):
            img_tmp_norm[:, :, i] = cv2.normalize(img_tmp_rs[:, :, i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Create image patches
        patches_img = patchify(img_tmp_rs, (crop_h, crop_w, n_channels), step=512)
        patches_img_rs = patches_img.reshape(-1, crop_h, crop_w, n_channels)
        img_inp.extend(patches_img_rs.tolist())

        # Load and preprocess label
        lbl_tmp = tifffile.imread(os.path.join(masks_folder, lbl_name))
        lbl_tmp_encoded = LabelEncoder().fit_transform(lbl_tmp.ravel())
        lbl_tmp_encoded_original_shape = lbl_tmp_encoded.reshape(lbl_tmp.shape)
        # Create label patches
        patches_lbl = patchify(lbl_tmp_encoded_original_shape, (crop_h, crop_w), step=512)
        patches_lbl_rs = patches_lbl.reshape(-1, crop_h, crop_w)
        lbl_inp.extend(patches_lbl_rs.tolist())

    # Convert input data to NumPy arrays
    img_inp_arr = np.array(img_inp)
    lbl_inp_arr = np.array(lbl_inp)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(img_inp_arr, lbl_inp_arr, test_size=0.2, random_state=42)

    # Data Augmentation on train set
    if data_augmentation:
        # Create a list of augmentation transformations to apply
        transforms = A.Compose([A.VerticalFlip(p=0.5), 
                                A.HorizontalFlip(p=0.5),
                                A.Rotate(p=0.7),                        
                                A.RandomBrightnessContrast(p=0.2)
                               ])

        # Create empty lists to store the augmented images
        x_train_aug = []
        y_train_aug = []

        # Generate and save augmented images to x_train_aug and y_train_aug
        for i in range(len(x_train)):
            for j in range(aug_level):
                # Apply the augmentations to the image and mask
                augmented = transforms(image=x_train[i], mask=y_train[i])
                x_aug = augmented['image']
                y_aug = augmented['mask']

                # Append the augmented image and mask to the respective lists
                x_train_aug.append(x_aug)
                y_train_aug.append(y_aug)

        # Convert the lists to NumPy arrays
        x_train_aug = np.array(x_train_aug)
        y_train_aug = np.array(y_train_aug)

    else:
        x_train_aug = x_train
        y_train_aug = y_train
       
    return x_train_aug, x_test, y_train_aug, y_test
      
 
