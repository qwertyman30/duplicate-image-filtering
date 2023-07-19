import argparse
import cv2
import glob
from tqdm import tqdm
from image_utils import preprocess_image_change_detection, compare_frames_change_detection


def resize_image(img: cv2.Mat, resolution: tuple) -> cv2.Mat:
    """
    Resize image to a given resolution
  
    Parameters:
    img (cv2.Mat): image to resize
    resolution (tuple): resolution to resize to
  
    Returns:
    cv2.Mat: resized image
    """
    return cv2.resize(img, resolution)


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

    filenames = glob.glob(data_folder + "/*.png")

    for filename in tqdm(filenames):
        # Resizing images to a common resolution since the images are of different sizes
        img = resize_image(cv2.imread(filename), common_resolution)
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
                           required=False)
    argparser.add_argument('-s',
                           '--similarity_threshold',
                           help='Similarity threshold',
                           default=10000,
                           required=False)
    argparser.add_argument('-g',
                           '--gaussian_blus_radius_list',
                           help='Gaussian blur radius list',
                           default=[5, 5],
                           required=False)
    argparser.add_argument('-m',
                           '--min_contour_area',
                           help='Minimum contour area',
                           default=500,
                           required=False)
    args = argparser.parse_args()

    filtered_images = filter_duplicates(
        args.data_folder,
        common_resolution=args.resolution,
        gaussian_blus_radius_list=args.gaussian_blus_radius_list,
        similarity_threshold=args.similarity_threshold,
        min_contour_area=args.min_contour_area)
