import argparse
import os
import os.path
import glob
from shutil import rmtree, move, copy
import random

ffmpeg_path = "C:/ffmpeg/bin/"
video_folder_path = "D:/KIEN/Dataset/UCF101/UCF-101/"
dataset_folder_path = "D:/KIEN/Dataset/UCF101/UCF101_Dataset/"

# Create Parser
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default=ffmpeg_path, help='path to ffmpeg.exe')
parser.add_argument("--video_folder", type=str, default=video_folder_path,
                    help='path to the folder containing videos')
parser.add_argument("--dataset_folder", type=str, default=dataset_folder_path,
                    help='path to the output folder')
parser.add_argument("--img_width", type=int, default=320, help='output image width')
parser.add_argument("--img_height", type=int, default=240, help='ouput image height')
args = parser.parse_args()

def extract_frames(videos, inputDir, outputDir):
    for video in videos:
        parts = video.split('/')        # parts = ['ApplyEyeMakeup', 'v_ApplyEyeMakeup_g08_c01.avi']
        classname = parts[0]
        filename = parts[1]
        filename_no_ext = filename.split('.')[0]

        # Check if this class exists
        if not os.path.exists(os.path.join(outputDir, classname)):
            os.mkdir("{}/{}".format(outputDir, classname))
        #
        # Check if this filename folder exists
        if not os.path.exists(os.path.join(outputDir, classname, filename_no_ext)):
            print("Create folder for %s/%s/%s" % (outputDir, classname, filename_no_ext))
            os.mkdir("{}/{}/{}".format(outputDir, classname, filename_no_ext))

        retn = os.system('{} -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%03d.jpg'.format(
            os.path.join(args.ffmpeg_dir, "ffmpeg.exe"),
            os.path.join(inputDir, video),
            args.img_width, args.img_height,
            os.path.join(outputDir, classname, filename_no_ext)))
        if retn:
            print("Error converting file:{}. Exiting.".format(video))

def create_clips_5(root,destination):
    # Distributes the images extracted by 'extract_frames()' to destination folder, which contain 5 frames each
    folderCounter = 9538

    classnames = glob.glob(os.path.join(root, '*'))
    print(classnames)

    for classname in classnames:
        filenames = glob.glob(os.path.join(classname, '*'))

        for filename in filenames:
            images = sorted(os.listdir(os.path.join(root, classname, filename)))
            # print(len(images))

            if not os.path.exists("{}/{}".format(destination, folderCounter)):
                os.makedirs("{}/{}".format(destination, folderCounter))

            # Choose three frames from folder images,
            # [-5][-3][-1]: choose from last, iterate every two frames; [0][2][4] choose from first
            copy("{}/{}".format(filename, images[-9]), "{}/{}/{}".format(destination, folderCounter, images[-9]))
            copy("{}/{}".format(filename, images[-7]), "{}/{}/{}".format(destination, folderCounter, images[-7]))
            copy("{}/{}".format(filename, images[-5]), "{}/{}/{}".format(destination, folderCounter, images[-5]))
            copy("{}/{}".format(filename, images[-3]), "{}/{}/{}".format(destination, folderCounter, images[-3]))
            copy("{}/{}".format(filename, images[-1]), "{}/{}/{}".format(destination, folderCounter, images[-1]))

            folderCounter += 1
            """
            for imageCounter, image in enumerate(images):

                # Iterate every two images
                if (imageCounter % 2 == 0):
                    # Create new folder for every 5 images
                    if (imageCounter % 5 == 0):
                        #if (imageCounter + 8 >= len(images)):
                        if (imageCounter > 99) or (imageCounter + 8 >= len(images)):
                            break
                        folderCounter += 1

                        if not os.path.exists("{}/{}".format(destination, folderCounter)):
                            os.makedirs("{}/{}".format(destination, folderCounter))
                    copy("{}/{}".format(filename, image), "{}/{}/{}".format(destination, folderCounter, image))
            """
            # rmtree(os.path.join(root, file))

def create_clips_3(root,destination):
    # Distributes the images extracted by 'extract_frames()' to destination folder, which contain 3 frames each
    folderCounter = 1

    classnames = glob.glob(os.path.join(root, '*'))

    for classname in classnames:
        filenames = glob.glob(os.path.join(classname, '*'))

        for filename in filenames:
            images = sorted(os.listdir(os.path.join(root, classname, filename)))
            # print(len(images))

            if not os.path.exists("{}/{}".format(destination, folderCounter)):
                os.makedirs("{}/{}".format(destination, folderCounter))

            # Choose three frames from folder images,
            # [-5][-3][-1]: choose from last, iterate every two frames; [0][2][4] choose from first
            copy("{}/{}".format(filename, images[-5]), "{}/{}/{}".format(destination, folderCounter, images[-5]))
            copy("{}/{}".format(filename, images[-3]), "{}/{}/{}".format(destination, folderCounter, images[-3]))
            copy("{}/{}".format(filename, images[-1]), "{}/{}/{}".format(destination, folderCounter, images[-1]))

            folderCounter += 1
            """
            for imageCounter, image in reversed(list(enumerate(images))):
                # Iterate every two images
                if (imageCounter % 2 == 0):
                    # Create new folder for every 3 images
                    if (imageCounter % 3 == 0):
                        if (imageCounter > 4) or (imageCounter + 6 >= len(images)):
                            break
                        folderCounter += 1

                        if not os.path.exists("{}/{}".format(destination, folderCounter)):
                            os.mkdir("{}/{}".format(destination, folderCounter))
                    copy("{}/{}".format(filename, image), "{}/{}/{}".format(destination, folderCounter, image))
            """
            # rmtree(os.path.join(root, file))

def main(version):
    # Create dataset folder if it doesnt exist already
    if not os.path.isdir(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    train_extractPath = os.path.join(args.dataset_folder, "train_extracted")
    trainPath = os.path.join(args.dataset_folder, "train6")

    test_extractPath = os.path.join(args.dataset_folder, "test_extracted")
    testPath = os.path.join(args.dataset_folder, "test6")

    # validationPath = os.path.join(args.dataset_folder, "val5")

    # Create dataset folder
    if not os.path.exists(train_extractPath):
        os.mkdir(train_extractPath)
    if not os.path.exists(trainPath):
        os.mkdir(trainPath)

    if not os.path.exists(test_extractPath):
        os.mkdir(test_extractPath)
    if not os.path.exists(testPath):
        os.mkdir(testPath)

    # os.mkdir(validationPath)
    """
    # Select half clips at random from test set for validation set.
    val_size = len([name for name in os.listdir(testPath)]) / 2
    int_val_size = int(val_size)
    testClips = os.listdir(testPath)
    indices = random.sample(range(len(testClips)), int_val_size)
    for index in indices:
        if index != 0: 
            move("{}/{}".format(testPath, index), "{}/{}".format(validationPath, index))
    """
    if version == '1':
        # Extract all train frames
        f = open("ucfTrainTestlist/trainlist01.txt", "r")
        videos = [row.split()[0] for row in list(f)]
        extract_frames(videos, args.video_folder, train_extractPath)

        # Extract all test frames
        f = open("ucfTrainTestlist/testlist01.txt", "r")
        videos = [row.strip() for row in list(f)]
        extract_frames(videos, args.video_folder, test_extractPath)
    if version == '2':
        # move frame from _extracted folder to destination folder, contains 3 or 5 frames each
        create_clips_5(train_extractPath, trainPath)
        # create_clips_5(test_extractPath, testPath)

if __name__ == '__main__':
    main(version='2')

