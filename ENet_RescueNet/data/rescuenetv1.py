'''
Masks are generated by my own scritps. Mask of all classes in uint8 format means multi hot encoded. 
This script tries to segment all types of classes.
Classes are: {'Background':0, 'Debris':1, 'Water':2, 'Building_No_Damage':3, 'Building_Minor_Damage':4, 'Building_Major_Damage':5, 
            'Building_Total_Destruction':6, 'Vehicle':7, 'Road':8, 'Tree':9, 'Pool':10, 'Sand':11}

'''

import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils
# @ sh:add
from PIL import Image, ImageOps, ImageFilter


class RescueNet(data.Dataset):
    """

    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = "train/train-org-img/"
    train_lbl_folder = "train/train-label-img/"

    # Validation dataset root folders
    val_folder = "val/val-org-img/"
    val_lbl_folder = "val/val-label-img/"

    # Test dataset root folders
    test_folder = "test/test-org-img/"
    test_lbl_folder = "test/test-label-img/"

    # Filters to find the images
    org_img_extension = '.jpg'
    #lbl_name_filter = '.png'

    lbl_img_extension = '.png'
    lbl_name_filter = 'lab'

    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # The values above are remapped to the following
    new_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            #('background', (255, 255, 255)),
            ('debris', (112, 9, 255)),
            ('water', (61, 230, 250)),
            ('building-no-damage', (180, 120, 120)),
            ('building-medium-damage', (235, 255, 7)),
            ('building-major-damage', (255, 184, 6)),
            ('building-total-destruction', (255, 0, 0)),
            ('vehicle', (255, 0, 245)),
            ('road', (140, 140, 140)),
            ('tree', (4, 250, 7)),
            #('pool', (255, 71, 0)),
            ('pool', (255, 235, 0)),
            ('sand', (160, 150, 20))
    ])

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.org_img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.org_img_extension)

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.org_img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.lbl_img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
        
    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """

        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        # Remap class labels
        label = utils.remap(label, self.full_classes, self.new_classes)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.mode == 'test':
            return img, label, os.path.basename(self.test_labels[index])
        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
