
images_folder = "Images"
masks_folder = "Label_images"

img_list = sorted(os.listdir(images_folder))
labels_list = sorted(os.listdir(masks_folder))
n_channels = 2
crop_h = 512  # crop height
crop_w = 512  # crop width
img_inp = []
lbl_inp = []
for img_name, lbl_name in zip(img_list, labels_list):
    img_tmp = tifffile.imread(os.path.join(images_path, img_name))
    img_tmp_rs = np.moveaxis(img_tmp, 0, -1)  # Reshape the original image to have height X width X n_channels
    img_tmp_rs = img_tmp_rs[:, :, :2]  # prendiamo solo le prime due immagini (BSE e CL)
    
    # normalize_images
    img_tmp_norm = img_tmp_rs.copy()
    img_tmp_norm[:, :, 0] = cv2.normalize(img_tmp_rs[:, :, 0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)
    img_tmp_norm[:, :, 1] = cv2.normalize(img_tmp_rs[:, :, 1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)
   
    # Divide each image in patches of size 512x512
    patches_img = patchify(img_tmp_rs, (crop_h, crop_w, n_channels), step=512)
    patches_img_rs = np.reshape(patches_img, (-1, crop_h, crop_w, n_channels))
    tmp_img_list = list(patches_img_rs)
    img_inp.extend(tmp_img_list)
    
    # Labels
    lbl_tmp = h5py.File(os.path.join(labels_path, lbl_name))
    key = list(lbl_tmp.keys())
    # print(key)
    lbl_tmp = lbl_tmp[key[0]][()]  # Ci prendiamo i dati
    # print(f'label_shape is {len(lbl_tmp.shape)}')
    if len(lbl_tmp.shape) > 2:
        lbl_tmp = lbl_tmp.reshape((lbl_tmp.shape[1], lbl_tmp.shape[2]))
    lbl_tmp[lbl_tmp >= 4] = 4.
    labelencoder = LabelEncoder()
    h, w = lbl_tmp.shape
    lbl_tmp_reshaped = lbl_tmp.reshape(-1, )
    lbl_tmp_reshaped_encoded = labelencoder.fit_transform(lbl_tmp_reshaped)
    lbl_tmp_encoded_original_shape = lbl_tmp_reshaped_encoded.reshape(h, w)
    
    patches_lbl = patchify(lbl_tmp_encoded_original_shape, (crop_h, crop_w), step=512)
    patches_lbl_rs = np.reshape(patches_lbl, (-1, crop_h, crop_w))
    tmp_lbl_list = list(patches_lbl_rs)
    lbl_inp.extend(tmp_lbl_list)
    
    # Image Augmentation Section
    aug = A.Compose([A.RandomCrop(width=512, height=512),
          A.VerticalFlip(p=0.5), 
          A.HorizontalFlip(p=0.5),
          A.Rotate(p=0.7)])

    for i in range(5):
        augmented = aug(image=img_tmp_rs, mask=lbl_tmp_encoded_original_shape)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        img_inp.append(image_aug)
        lbl_inp.append(mask_aug)
        
img_inp_arr = np.array(img_inp)
lbl_inp_arr = np.array(lbl_inp)

