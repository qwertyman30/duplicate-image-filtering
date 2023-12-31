This repo contains code that would filter the given dataset to remove duplicate images

<b><u>Steps to run the script</b></u><br>
<b>USING LOCAL PYTHON</b>
1. pip install -r requirements.txt
2. python main.py -d <PATH_TO_DATA_FOLDER> <ANY_ADDITIONAL_ARGS><br>
eg. python main.py -d data/

<b>USING DOCKER</b>
1. docker build -t <NAME_OF_IMAGE> .<br>
eg. `docker build -t filter .`
2. docker run --rm -v <PATH_TO_DATA_FOLDER>:/app/data/ -v <PATH_TO_FILTERED_IMAGES_FOLDER>:/app/filtered --name <NAME_OF_CONTAINER> <NAME_OF_IMAGE> -d /app/data <ANY_ADDITIONAL_ARGS><br>
eg. `docker run --rm -v ./data:/app/data/ -v ./filtered:/app/filtered/ --name filter-container filter -d /app/data`

<b>ADDITIONAL ARGUMENTS THAT CAN BE SPECIFIED DURING RUN TIME</b><br>
1. -r or --resolutions: common resolution to resize each image to. The dataset contains images of varying resolutions. The compare function needs the images to be of the same resolution. DEFAULT: (1152, 852)
2. -s or --similarity_threshold: similarity threshold. If the score is below this threshold, the images are considered duplicate. DEFAULT: 10000
3. -g or --gaussian_blur_radius_list: Gaussian blur radius list. Applies gaussian blur during preprocessing if specified. DEFAULT: [5, 5]
4. -m or --min_contour_area: Minimum contour area. This filters out small contour changes between images. DEFAULT: 500
5. -f or --filter_directory: Folder where the filtered images without duplicates get stored. DEFAULT: "filtered"