import input_parameters
import template_matching

import time
import tqdm

import sys

from yeaz.unet.segment import segment
from yeaz.disk import Reader as nd
import argparse
import skimage
from yeaz.unet import neural_network as nn

import torch
import glob


# arguments from argparse
# '--image_path', type=str, help="Specify the path to a single image or to a folder of images", required=True)
# '--mask_path', type=str, help="Specify where to save predicted masks", required=True)
# '--image_type', type=str, help="Specify the imaging type, possible types are 'bf', 'pc', and 'fission'. Supersedes path_to_weights.")
# '--path_to_weights', default=None, type=str, help="Specify weights path.")
# '--fov', default=[0], nargs='+', type=int, help="Specify field of view index (can specify more than one with space between them).")
# '--range_of_frames', nargs=2, default=[0, 0], type=int, help="Specify start and end in range of frames. (e.g. 0 10)")
# '--threshold', default=None, type=float, help="Specify threshold value.")
# '--min_seed_dist', default=5, type=int, help="Specify minimum distance between seeds.")
# '--device', default='cpu', type=str, help="Specify device to run on (cpu or cuda).")


def LaunchPrediction(im, mic_type, pretrained_weights=None, device='cpu'):
    """It launches the neural neutwork on the current image and creates
    an hdf file with the prediction for the time T and corresponding FOV.
    """
    im = skimage.exposure.equalize_adapthist(im)
    im = im * 1.0;
    pred = nn.prediction(im, mic_type, pretrained_weights, device=device)
    return pred


def ThresholdPred(thvalue, pred):
    """Thresholds prediction with value"""
    if thvalue == None:
        thresholdedmask = nn.threshold(pred)
    else:
        thresholdedmask = nn.threshold(pred, thvalue)
    return thresholdedmask


def LaunchInstanceSegmentation(reader, image_type, fov_indices=[0], time_value1=0, time_value2=0, thr_val=None,
                               min_seed_dist=5, path_to_weights=None, device='cpu'):
    if (device == 'cuda') and torch.cuda.is_available():
        f_device = 'cuda'
    else:
        f_device = 'cpu'

        # cannot have both path_to_weights and image_type supplied
    if (image_type is not None) and (path_to_weights is not None):
        print("image_type and path_to_weights cannot be both supplied.")
        return

    # check if correct imaging value
    if (image_type not in ['bf', 'pc']) and (path_to_weights is None):
        print("Wrong imaging type value ('{}')!".format(image_type),
              "imaging type must be either 'bf' or 'pc'")
        return

    # check range_of_frames constraint
    if time_value1 > time_value2:
        print("Error", 'Invalid Time Constraints')
        return

    # displays that the neural network is running
    print('Running the neural network on {} ...'.format(f_device))

    for fov_ind in tqdm.tqdm(fov_indices, desc='FOV', position=0):

        # iterates over the time indices in the range
        for t in tqdm.tqdm(range(time_value1, time_value2 + 1), desc='Time', position=1, leave=False):
            # print('--------- Segmenting field of view:',fov_ind,'Time point:',t)

            # calls the neural network for time t and selected fov
            im = reader.LoadOneImage(t, fov_ind)

            try:
                pred = LaunchPrediction(im, image_type, pretrained_weights=path_to_weights, device=f_device)
            except ValueError:
                print('Error! ',
                      'The neural network weight files could not '
                      'be found. \nMake sure to download them from '
                      'the link in the readme and put them into '
                      'the folder unet, or specify a path to a custom weights file with -w argument.')
                return

            thresh = ThresholdPred(thr_val, pred)
            seg = segment(thresh, pred, min_seed_dist)
            reader.SaveMask(t, fov_ind, seg)
            print('--------- Finished segmenting.')

            # apply tracker if wanted and if not at first time
            temp_mask = reader.CellCorrespondence(t, fov_ind)
            reader.SaveMask(t, fov_ind, temp_mask)


def main():
    ###################################
    # step 1 : do the template matching
    ###################################
    # print('\n')
    # print("Template matching")
    # args = template_matching.arguments()
    # template_matching.main()
    # exit()

    ###################################
    # step 2 : do the segmentation [code from launch_NN_commandline]
    ###################################

    path_to_templates = "/Users/nmarounina/Desktop/Projects/Yeast_segmentation/my_tests/Data_Set_1_for_YeaZv2/separated_traps"
    # "/Users/nmarounina/Desktop/Projects/Yeast_segmentation/my_tests/Data_Set_1_for_YeaZv2/"
    # "/Users/nmarounina/Desktop/Projects/Yeast_segmentation/segmentation_tracking_pipeline/output_data/dset_for_yeazV2/split_data"

    # here we segment the reference traps :
    list_folders = glob.glob(path_to_templates + '/*')
    for folder in list_folders:
        print('\n')
        print(folder)
        list_path_traps = glob.glob(folder + '/split_data/*')

        for path_to_trap in list_path_traps:
            print(">>",path_to_trap)
            image_path = glob.glob(path_to_trap + '/*.tif')[0]
            mask_path = image_path.split(".tif")[0]
            reader = nd.Reader("", mask_path + '.h5', image_path)

            path_to_weights = '/Users/nmarounina/Desktop/Projects/Yeast_segmentation/segmentation_tracking_pipeline/YeaZ-GUI/yeaz/unet/weights/weights_budding_BF_multilab_0_1'
            try:
                LaunchInstanceSegmentation(reader, image_type=None, fov_indices=[0], time_value1=0, time_value2=369,
                                           thr_val=0.9,
                                           min_seed_dist=5, path_to_weights=path_to_weights, device='cpu')
            except:
                pass

    # here is the code to segment several tifs in one folder:
    # list_images = glob.glob(path_to_templates + '/*.tif')
    #
    #
    # for image_path in list_images:
    #
    #     print('\n')
    #     print(image_path)
    #     mask_path = image_path.split(".tif")[0]
    #     reader = nd.Reader("", mask_path + '.h5', image_path)

    # this is to deal with the split_images situation, generic case
    # list_path_traps = glob.glob(path_to_templates + '/*')
    #
    # for path_to_trap in list_path_traps:
    #
    #     print('\n')
    #     print(path_to_trap.split('/')[-1])
    #     image_path = glob.glob(path_to_trap + '/*.tif')[0]
    #     mask_path = image_path.split(".tif")[0]
    #     reader = nd.Reader("", mask_path + '.h5', image_path)

    # path_to_weights = '/Users/nmarounina/Desktop/Projects/Yeast_segmentation/segmentation_tracking_pipeline/YeaZ-GUI/yeaz/unet/weights/weights_budding_BF_multilab_0_1'
    # try:
    #     LaunchInstanceSegmentation(reader, image_type=None, fov_indices=[0], time_value1=0, time_value2=369, thr_val=0.9,
    #                        min_seed_dist=5, path_to_weights=path_to_weights, device='cpu')
    # except:
    #     pass


if __name__ == '__main__':
    main()

# def main(args):
#     if '.h5' in args.mask_path:
#         args.mask_path = args.mask_path.replace('.h5', '')
#
#     reader = nd.Reader("", args.mask_path + '.h5', args.image_path)
#
#     LaunchInstanceSegmentation(reader, args.image_type, args.fov,
#                                args.range_of_frames[0], args.range_of_frames[1],
#                                args.threshold, args.min_seed_dist,
#                                args.path_to_weights, device=args.device)
