'''This file upload all the needed functions and performs a unet segmentation on the sandstone images dataset'''
import Functions/data_prep.py as data_prep
import Functions/multi_unet_model.py as multi_unet_model
import Functions/training_model.py as training_model
import Functions/plot_confusion_matrix.py as plot_confusion_matrix

# Define segmentation parameters
n_classes = 4
crop_h = 512
crop_w = 512
n_channels = 2

# We call the data_prep function with the default arguments. It will automatically create train and test sets
x_train_aug, x_test, y_train_aug, y_test = data_prep()

# We run the function train_model that automatically run the unet_model on the training dataset
history, unet_model = train_model(x_train_aug, y_train_aug, x_test, y_test, n_classes, crop_h, crop_w, n_channels, batch_size=8, epochs=10, early_stopping_patience=5, verbose=1) 

# Check the results on the test set
y_pred=unet_model.predict(x_test)

# The model outputs 4 different images with each pixel ranging from 0 to 1
# The final segmentation is obtained after selecting, for each pixel, the max values among the different images
y_pred_argmax=np.argmax(y_pred, axis=3)

# Let's calculate the confusion matrix
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
y_unique = np.unique(y_test)
mcm = multilabel_confusion_matrix(y_test.reshape(-1,1), y_pred_argmax.reshape(-1,1), labels = y_unique)
cm = confusion_matrix(y_test.reshape(-1,1), y_pred_argmax.reshape(-1,1), labels = y_unique)

figure = plot_confusion_matrix(cm, classes=['Quartz', 'Quartz Cement', 'Porosity', 'Other'], normalize=True, font_size=12)
