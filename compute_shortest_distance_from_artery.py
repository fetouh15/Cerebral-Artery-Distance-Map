# -*- coding: utf-8 -*-
"""
DESCRIPTION:
Computes the minimal distance between each voxel in the mask and the artery of choice.
Saves the distance map (nifti image) of the artery.

"""

import argparse
import ctypes
import os.path

from scipy.spatial import distance as dist

import nibabel as nib
import numpy as np

import multiprocessing
import time


def parse_arguments():
    """
    Simple CommandLine argument parsing function making use of the argparse module

    :return: parsed arguments object args
    """
    parser = argparse.ArgumentParser(
        description=" Calculation of ROI vascular distances"
    )

    parser.add_argument(
        "-v",
        "--vessels",
        help="Full Brain Vessel Segmentation [ .nii / .nii.gz ]",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tof",
        help="TOF_resampled [ .nii / .nii.gz ]",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mask",
        help="ROI mask[ .nii / .nii.gz ]",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-a",
        "--artery",
        help="Enter label number for target artery. If artery is present in left and right hemispheres, enter left hemisphere label and use -rha to enter right hemisphere label.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-rha",
        "--right_hemisphere_artery",
        help="Enter right hemisphere label number of the target artery.",
        type=int,
    )
    parser.add_argument(
        "-od",
        "--output_directory",
        help="Absolute path to output directory",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    return args


def euclidean_distance(index, artery_locations):
    """
    Calculates Euclidean distance between a point and all artery locations
    """
    start_local = time.time()
    target_voxel = np.zeros((1, 3))
    target_voxel[0] = np.array(list(index))

    distances = dist.cdist(target_voxel, artery_locations, metric="euclidean")

    arterial_distances = np.frombuffer(np_artery.get_obj(), dtype=np.float64).reshape(
        np_shape
    )

    arterial_distances[index] = np.min(distances)

    elapsed_time = round((time.time() - start) / 60, 2)
    projected_time = int(
        ((time.time() - start_local) * iterations) / (60 * multiprocessing.cpu_count())
    )

    print(
        "\r Elapsed Time: {}".format(elapsed_time)
        + " mins / Projected Remaining Time: {}".format(projected_time),
        end=" mins",
    )


def pool_initializer(artery, shape, masked_data_filter_size):
    """
    Initializes global variables
    """
    global np_artery
    np_artery = artery
    global np_shape
    np_shape = shape
    global iterations
    iterations = masked_data_filter_size


def check_parameters(vessels, mask, output_directory):
    """
    Checks if full brain vessel segmentation file size and voxel size is equal to the mask's. Checks if output directory exist.
    """
    assert vessels.header.get_zooms() == mask.header.get_zooms(), (
            "Check input files. mask and vessels have different voxel size: "
            + str(mask.header.get_zooms())
            + " "
            + str(vessels.header.get_zooms())
    )
    assert vessels.shape == mask.shape, (
            "Check input files. mask and vessels size dont match "
            + str(mask.shape)
            + " "
            + str(vessels.shape)
    )
    assert os.path.exists(output_directory), "Output directory doesnt exist"


def main():
    global start
    start = time.time()

    # Argument Parsing
    args = parse_arguments()
    vessels = args.vessels
    mask = args.mask
    tof = args.tof
    output_directory = args.output_directory
    artery = args.artery
    right_hemisphere_artery = args.right_hemisphere_artery
    # Load Images
    vessels_img = nib.load(vessels)
    mask_img = nib.load(mask)
    if tof is not None:
        tof_img = nib.load(tof)
    # Validate Parameters
    check_parameters(vessels_img, mask_img, output_directory)

    # Extract data to numpy arrays
    mask_data = np.asanyarray(mask_img.dataobj).astype(int)
    vessels_data = np.asanyarray(vessels_img.dataobj).astype(int)
    if tof is not None:
        tof_data = np.asanyarray(tof_img.dataobj).astype(int)

    # Set other arteries to zero
    if right_hemisphere_artery is not None:
        vessels_data[vessels_data == right_hemisphere_artery] = artery
    vessels_data[vessels_data != artery] = 0
    vessels_data[vessels_data == artery] = 1

    # Listing aca,mca and pca artery locations individually
    artery_locations_a = np.nonzero(vessels_data)
    artery_locations = list(zip(*artery_locations_a))

    # Listing the nonzero voxels of the mask
    mask_data[mask_data > 0] = 1
    if tof is not None:
        mask_data = np.where(tof_data == 0, 0, mask_data)
    mask_data_filter_tuple = np.nonzero(mask_data)
    mask_data_filter = list(zip(*mask_data_filter_tuple))

    # Creating output arrays and shared memory arrays

    shared_arr = multiprocessing.Array(
        ctypes.c_float, mask_data.shape[0] * mask_data.shape[1] * mask_data.shape[2] * 2
    )
    arterial_distances = np.frombuffer(shared_arr.get_obj(), dtype=np.float64).reshape(
        mask_data.shape
    )

    # Create pool using all cpu cores
    pool = multiprocessing.Pool(
        processes=multiprocessing.cpu_count(),
        initializer=pool_initializer,
        initargs=(
            shared_arr,
            mask_data.shape,
            len(mask_data_filter),
        ),
    )

    # Find Euclidean distance for all nonzero voxels of the mask
    result = pool.starmap_async(
        euclidean_distance,
        [(index, artery_locations) for index in mask_data_filter],
    )
    result.wait()
    pool.close()
    pool.join()

    # Saving the output file

    pix_dim = vessels_img.header.get_zooms()[0]
    arterial_distances *= pix_dim

    nib.save(
        nib.Nifti1Image(
            arterial_distances, affine=mask_img.affine, header=mask_img.header
        ),
        output_directory + "/distance_map_" + str(artery) + ".nii.gz",
    )


if __name__ == "__main__":
    main()
