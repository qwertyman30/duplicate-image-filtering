import argparse
import cv2
from tqdm import tqdm
import shutil
import os
from image_utils import preprocess_image_change_detection, compare_frames_change_detection


def filter_duplicates(data_folder: str,
                      common_resolution: tuple = (1152, 864),
                      gaussian_blus_radius_list: list = [3, 3],
                      similarity_threshold: int = 10000,
                      min_contour_area: int = 500) -> dict:
    """
    Filter duplicate images from a folder

    Parameters:
    input_folder (str): path to the input folder
    common_resolution (tuple): resolution to resize to. Default: (1152, 864)
    similarity_threshold (int): similarity threshold. Default: 10000
    min_contour_area (int): minimum contour area. Default: 500

    Returns:
    dict: dictionary of filtered images
    """
    image_dict = {}

    files = os.listdir(data_folder)
    files = [file for file in files if file.endswith(".png")]

    for filename in tqdm(files):
        img_path = os.path.join(data_folder, filename)
        img = cv2.imread(img_path)
        # Resizing images to a common resolution since the images are of different sizes
        img = cv2.resize(img, common_resolution)
        # Preprocessing images
        preprocessed_img = preprocess_image_change_detection(
            img, gaussian_blur_radius_list=gaussian_blus_radius_list)

        similar_found = False
        for _, value in image_dict.items():
            # Comparing images and getting the similarity score
            score, _, _ = compare_frames_change_detection(
                value, preprocessed_img, min_contour_area)
            # If the similarity score is less than the threshold, the images are similar
            if score < similarity_threshold:
                similar_found = True
                break

        if not similar_found:
            image_dict[filename] = preprocessed_img

    return image_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d',
                           '--data_folder',
                           help='Path to the data folder',
                           required=True)
    argparser.add_argument('-r',
                           '--resolution',
                           help='Path to the input folder',
                           default=(1280, 720),
                           type=tuple,
                           required=False)
    argparser.add_argument('-s',
                           '--similarity_threshold',
                           help='Similarity threshold',
                           default=25000,
                           type=int,
                           required=False)
    argparser.add_argument('-g',
                           '--gaussian_blus_radius',
                           help='Gaussian blur radius',
                           default=5,
                           type=int,
                           required=False)
    argparser.add_argument('-m',
                           '--min_contour_area',
                           help='Minimum contour area',
                           default=100,
                           type=int,
                           required=False)
    argparser.add_argument(
        '-f',
        '--filter_directory',
        help='Directory where the filtered images would be saved to',
        default="filtered",
        required=False)

    args = argparser.parse_args()

    gaussian_blur_radius_list = [args.gaussian_blus_radius] * 2
    # Filter the duplicate images from the dataset
    filtered_images = filter_duplicates(
        args.data_folder,
        common_resolution=args.resolution,
        gaussian_blus_radius_list=gaussian_blur_radius_list,
        similarity_threshold=args.similarity_threshold,
        min_contour_area=args.min_contour_area)

    print("Found {} unique images".format(len(filtered_images)))

    # Create the filter directory if it doesn't exist
    if not os.path.exists(args.filter_directory):
        os.makedirs(args.filter_directory)

    # Copy the filtered images to the filter directory
    for key, val in filtered_images.items():
        src_path = os.path.join(args.data_folder, key)
        dst_path = os.path.join(args.filter_directory, key)
        shutil.copy(src_path, dst_path)