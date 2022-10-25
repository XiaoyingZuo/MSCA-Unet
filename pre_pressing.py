import os
from sklearn.utils import resample
from torch.utils.data import Dataset
import cv2
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import tensorflow
monai


data_path = 'G:/Data set summary/pancreas'
ct_list_file = 'G:/Medical_image_segmentation_project/Pancreatic_Tumor_Segmentation_main/dataset/ct.txt'
seg_list_file = 'G:/Medical_image_segmentation_project/Pancreatic_Tumor_Segmentation_main/dataset/seg.txt'

ct_train_file = 'G:/Medical_image_segmentation_project/Pancreatic_Tumor_Segmentation_main/dataset/train/ct.txt'
seg_train_file = 'G:/Medical_image_segmentation_project/Pancreatic_Tumor_Segmentation_main/dataset/train/seg.txt'

# ct_train_file = 'dataset/val/ct.txt'
# seg_train_file = 'dataset/val/seg.txt'
ct_save_path = 'G:/Medical_image_segmentation_project/Pancreatic_Tumor_Segmentation_main/dataset/CT'
seg_save_path = 'G:/Medical_image_segmentation_project/Pancreatic_Tumor_Segmentation_main/dataset/SEG'


def load_itk_image(filename):
    """
    Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def save_itk_image(filename, img_array):
    img = sitk.GetImageFromArray(img_array)
    sitk.WriteImage(img, filename)

#
# def center_crop(img):
#     d, h, w = img.shape
#     crop_size = [d / 6, h / 6, w / 6]
#     crop_img = img[:, int(crop_size[1]):int(h - crop_size[1]), int(crop_size[2]):int(w - crop_size[2])]
#     # print(crop_img.shape)
#     return crop_img
def center_crop(img):

    d, h, w = img.shape
    crop_size = [d/2, h/2, w/2]
    crop_img = img[:, int(crop_size[1]-128):int(crop_size[1]+128), int(crop_size[2]-128):int(crop_size[2]+128)]
    print(crop_img.shape)
    return crop_img


def expand_slice(img):
    slice_list = []
    for slice in img:
        # slice = cv2.resize(slice, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        slice_list.append(slice)
        slice_list.append(slice)
    expand_img = np.array(slice_list)
    return expand_img


def resample(image, spacing, new_spacing=[1.0, 1.0, 1.0], order=1):
    new_shape = np.round(image.shape * spacing / new_spacing)

    # the actual spacing to resample.
    resample_spacing = spacing * image.shape / new_shape

    resize_factor = new_shape / image.shape

    image_new = ndimage.interpolation.zoom(image, resize_factor,
                                           mode='nearest', order=order)

    return (image_new, resample_spacing)


def HU2uint8(image, HU_min=-100.0, HU_max=200.0, HU_nan=-2000.0):
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


def crop_16x(img):
    d, h, w = img.shape
    return img[:int(d / 16) * 16, :int(h / 16) * 16, :int(w / 16) * 16]


def generate_filename_list():
    img_a_list = []
    img_d_list = []
    img_v_list = []

    seg_a_list = []
    seg_d_list = []
    seg_v_list = []

    for folder in os.listdir(data_path):
        a_path = os.path.join(data_path, folder, '{} A'.format(folder))
        d_path = os.path.join(data_path, folder, '{} D'.format(folder))
        v_path = os.path.join(data_path, folder, '{} V'.format(folder))

        for filename in os.listdir(a_path):
            if os.path.splitext(filename)[1] == '.gz':
                seg_a_list.append(os.path.normpath(os.path.join(a_path, filename)).replace('\\', '/'))

            if os.path.splitext(filename)[1] == '.nii':
                img_a_list.append(os.path.normpath(os.path.join(a_path, filename)).replace('\\', '/'))

        for filename in os.listdir(d_path):
            if os.path.splitext(filename)[1] == '.gz':
                seg_d_list.append(os.path.normpath(os.path.join(d_path, filename)).replace('\\', '/'))

            if os.path.splitext(filename)[1] == '.nii':
                img_d_list.append(os.path.normpath(os.path.join(d_path, filename)).replace('\\', '/'))

        for filename in os.listdir(v_path):
            if os.path.splitext(filename)[1] == '.gz':
                seg_v_list.append(os.path.normpath(os.path.join(v_path, filename)).replace('\\', '/'))

            if os.path.splitext(filename)[1] == '.nii':
                img_v_list.append(os.path.normpath(os.path.join(v_path, filename)).replace('\\', '/'))

    with open(ct_list_file, "w") as f:
        for line in img_a_list:
            f.write(line + '\n')
        for line in img_d_list:
            f.write(line + '\n')
        for line in img_v_list:
            f.write(line + '\n')

    with open(seg_list_file, "w") as f:
        for line in seg_a_list:
            f.write(line + '\n')
        for line in seg_d_list:
            f.write(line + '\n')
        for line in seg_v_list:
            f.write(line + '\n')


def preprocess():
    ct_list = []
    with open(ct_list_file, 'r') as f:
        for line in f:
            ct_list.append(line.strip('\n'))

    seg_list = []
    with open(seg_list_file, 'r') as f:
        for line in f:
            seg_list.append(line.strip('\n'))

    # keep_slice_list = []
    # start_slice, end_slice = 0, 0

    seg_file_list = []
    ct_file_list = []
    for img_idx in range(len(seg_list)):
        seg_img, _, seg_spacing = load_itk_image(filename=seg_list[img_idx])
        seg_save_name = seg_list[img_idx].split('/')[-1]

        seg_img, _ = resample(seg_img, seg_spacing)

        seg_img = center_crop(seg_img)
        seg_img = crop_16x(seg_img)

        ct_img, _, ct_spacing = load_itk_image(filename=ct_list[img_idx])
        ct_save_name = ct_list[img_idx].split('/')[-1]

        ct_img = HU2uint8(image=ct_img)
        ct_img, _ = resample(ct_img, ct_spacing)

        ct_img = center_crop(ct_img)
        ct_img = crop_16x(ct_img)

        if ct_img.shape != seg_img.shape:
            print('{} shape not equal, ignored.'.format(ct_save_name))
            continue

        seg_file_list.append(os.path.join(seg_save_path, seg_save_name).replace('\\', '/'))
        ct_file_list.append(os.path.normpath(os.path.join(ct_save_path, ct_save_name)).replace('\\', '/'))

        save_itk_image(os.path.normpath(os.path.join(seg_save_path, seg_save_name)).replace('\\', '/'), seg_img)
        save_itk_image(os.path.normpath(os.path.join(ct_save_path, ct_save_name)).replace('\\', '/'), ct_img)

        if ct_img.shape != seg_img.shape:
            print('{} shape not equal, ignored.'.format(ct_save_name))

        print('[{}] {} {} shape: {} '.format(img_idx, seg_save_name, ct_save_name, ct_img.shape))

    with open(ct_list_file, "w") as f:
        for line in ct_file_list:
            f.write(line + '\n')

    with open(seg_list_file, "w") as f:
        for line in seg_file_list:
            f.write(line + '\n')


def crop_img_slice():
    ct_list = []
    with open(ct_train_file, 'r') as f:
        for line in f:
            ct_list.append(line.strip('\n'))

    seg_list = []
    with open(seg_train_file, 'r') as f:
        for line in f:
            seg_list.append(line.strip('\n'))

    for img_idx in range(len(seg_list)):
        keep_slice_list = []
        seg_img, _, seg_spacing = load_itk_image(filename=seg_list[img_idx])
        seg_save_name = seg_list[img_idx].split('/')[-1]

        for slice_idx in range(len(seg_img)):
            if seg_img[slice_idx].sum() != 0:
                keep_slice_list.append(slice_idx)

        if len(keep_slice_list) == 0:
            print(seg_save_name + ' have no mask.')

        print('[{}]{} have {} slices'.format(img_idx, seg_list[img_idx], len(keep_slice_list)))

        first_slice = keep_slice_list[0]
        last_slice = keep_slice_list[-1]

        # seg_img = expand_slice(seg_img)
        seg_img = seg_img[first_slice:last_slice + 1, ...]

        save_itk_image(os.path.join(seg_save_path, seg_save_name).replace('\\', '/'), seg_img)

        ct_img, _, ct_spacing = load_itk_image(filename=ct_list[img_idx])
        ct_save_name = ct_list[img_idx].split('/')[-1]

        # ct_img = expand_slice(ct_img)
        ct_img = ct_img[first_slice:last_slice + 1, ...]

        if ct_img.shape != seg_img.shape:
            print('{} shape not equal, ignored.'.format(ct_save_name))
            continue

        save_itk_image(os.path.join(ct_save_path, ct_save_name).replace('\\', '/'), ct_img)


if __name__ == '__main__':
    # generate_filename_list()
    preprocess()
    # crop_img_slice()

