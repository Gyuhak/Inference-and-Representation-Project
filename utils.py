# utils.py

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys


def load_data(base_dir):
    isd_img = os.path.isdir('data/sintel/img')
    isd_occ = os.path.isdir('data/sintel/occ')
    isf_img = os.path.isfile('data/sintel/img/temple_3_albedo_5_frame_0050.png')
    if not isd_img or not isd_occ or not isf_img:
        reform_data(base_dir)

    print('load_data')
    # imgs = read_images('data/sintel/img', img_type='double')
    # occs = read_images('data/sintel/occ', img_type='single')
    imgs, occs = read_images('data/sintel/')

    import random
    temp = list(zip(imgs, occs))
    random.shuffle(temp)
    imgs, occs = zip(*temp)

    return np.asarray(imgs), np.asarray(occs)


def read_images(target_dir):
    imgs = []
    occs = []
    occ_files = os.listdir(os.path.join(target_dir, 'occ'))
    occ_files = sorted(occ_files)
    for occ_file in occ_files:
        if occ_file.endswith('.png'):
            name_format = occ_file.split('_')
            occ_path = os.path.join(target_dir, 'occ', occ_file)
            # alley_1_0_frame_0001
            img0_name = '%s_%s_%s_%s_%s_%04d.png' % (name_format[0],
                                                     name_format[1],
                                                     name_format[2],
                                                     name_format[3],
                                                     name_format[4],
                                                     int(name_format[5].split('.')[0]))
            img1_name = '%s_%s_%s_%s_%s_%04d.png' % (name_format[0],
                                                     name_format[1],
                                                     name_format[2],
                                                     name_format[3],
                                                     name_format[4],
                                                     int(name_format[5].split('.')[0]) + 1)
            img0_path = os.path.join(target_dir, 'img', img0_name)
            img1_path = os.path.join(target_dir, 'img', img1_name)
            tocc_img = misc.imread(occ_path, flatten=True)
            # tocc_img[tocc_img > 0] = 1
            occs.append(np.asarray(tocc_img).flatten())
            imgs.append(np.dstack((misc.imread(img0_path, flatten=True),
                                   misc.imread(img1_path, flatten=True))).flatten())
    return imgs, occs


def resize_images(target_dir, rst_dir, itype='clean'):
    print('resize_images: %s' % target_dir)

    if not os.path.isdir(rst_dir):
        os.makedirs(rst_dir)

    sub_dirs = os.listdir(target_dir)

    for sub_dir in sub_dirs:
        if not sub_dir.startswith('.'):
            print(sub_dir)
            img_files = os.listdir(os.path.join(target_dir, sub_dir))
            for img_file in img_files:
                if img_file.endswith('.png'):
                    file_path = os.path.join(target_dir, sub_dir, img_file)
                    img = misc.imread(file_path)
                    if img is None:
                        print('cannot read image files')
                        return

                    # original image size: 436 × 1024
                    # img = misc.imresize(img, (256, 600))
                    misc.imsave(os.path.join(rst_dir, '%s_%s_0_%s' % (sub_dir, itype, img_file)),
                                img[:256, 100:356])
                    misc.imsave(os.path.join(rst_dir, '%s_%s_1_%s' % (sub_dir, itype, img_file)),
                                img[:256, 400:656])
                    misc.imsave(os.path.join(rst_dir, '%s_%s_2_%s' % (sub_dir, itype, img_file)),
                                img[:256, 700:956])
                    misc.imsave(os.path.join(rst_dir, '%s_%s_3_%s' % (sub_dir, itype, img_file)),
                                img[180:, 100:356])
                    misc.imsave(os.path.join(rst_dir, '%s_%s_4_%s' % (sub_dir, itype, img_file)),
                                img[180:, 400:656])
                    misc.imsave(os.path.join(rst_dir, '%s_%s_5_%s' % (sub_dir, itype, img_file)),
                                img[180:, 700:956])


def reform_data(base_dir):
    img_dir_albedo = os.path.join(base_dir, 'MPI-Sintel-training_images/training/albedo/')
    img_dir_clean = os.path.join(base_dir, 'MPI-Sintel-training_images/training/clean/')
    img_dir_final = os.path.join(base_dir, 'MPI-Sintel-training_images/training/final/')
    occ_dir = os.path.join(base_dir, 'MPI-Sintel-training_extras/training/occlusions/')

    resize_images(img_dir_albedo, 'data/sintel/img', itype='albedo')
    resize_images(occ_dir, 'data/sintel/occ', itype='albedo')
    resize_images(img_dir_clean, 'data/sintel/img', itype='clean')
    resize_images(occ_dir, 'data/sintel/occ', itype='clean')
    resize_images(img_dir_final, 'data/sintel/img', itype='final')
    resize_images(occ_dir, 'data/sintel/occ', itype='final')


def draw_rectangle(sp, offset, width, height):
    sp.add_patch(
        patches.Rectangle(
            (float(offset[0]), float(offset[1])), width - 1, height - 1,
            linewidth=1.5, edgecolor='red', fill=False
        )
    )


def imshow_in_subplot_with_labels(r, c, idx, img, xlabel, ylabel):
    sp = plt.subplot(r, c, idx)
    sp.set_xlabel(xlabel)
    sp.set_ylabel(ylabel)
    imshow_gray(img)
    return sp


def imshow_in_subplot_with_title(r, c, idx, img, title):
    sp = plt.subplot(r, c, idx)
    sp.set_title(title)
    imshow_gray(img)
    return sp


def imshow_in_subplot(r, c, idx, img):
    sp = plt.subplot(r, c, idx)
    imshow_gray(img)
    return sp


def imshow(img):
    plt.imshow(img, interpolation='nearest')


def imshow_gray(img):
    plt.imshow(img, cmap='gray', interpolation='nearest')


def plot_in_subplot_with_title(r, c, idx, plot, title):
    sp = plt.subplot(r, c, idx)
    sp.set_title(title)
    plot_with_margin(plot)
    return sp


def plot_in_subplot(r, c, idx, plot):
    sp = plt.subplot(r, c, idx)
    plot_with_margin(plot)
    return sp


def plot_with_margin(plot):
    plt.plot(plot)
    margin = len(plot) * 0.1
    plt.xlim(-margin, len(plot) + margin)


def init_figure(fig_idx):
    fig_idx += 1
    this_fig = plt.figure(fig_idx)
    this_fig.set_tight_layout(True)
    return this_fig


def init_figure_with_idx(fig_idx, figsize=(10, 10)):
    this_fig = plt.figure(fig_idx, figsize=figsize)
    this_fig.set_tight_layout(True)
    return this_fig


def show_plots():
    plt.show()


def save_fig_in_dir(fig, dirname='', filename=None):
    if filename is None:
        return

    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    fig.savefig(os.path.join(dirname, filename))


def open_in_preview(file_names):  # for mac
    params = ['open', '-a', 'Preview']
    import subprocess
    subprocess.call(params.extend(file_names))


def is_cv2():
    return check_opencv_version("2.")


def is_cv3():
    return check_opencv_version("3.")


def check_opencv_version(major, lib=None):
    if lib is None:
        import cv2 as lib
    return lib.__version__.startswith(major)


def is_python2():
    return python_version() == 2


def is_python3():
    return python_version() == 3


def python_version():
    return sys.version_info.major


# End of script
