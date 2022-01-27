import os
import time

import cv2
from tqdm import tqdm


def process_mime(root_path: str,
                 target_path: str,
                 target_img_shape: tuple = (64, 64)):
    """
        Converts MIME data in the form
        root/
        |
        |--- mime_bottle/
        |       |
        |       |--- 4329Aug02/
        |       |       |
        |       |       |--- hd_kinect_rgb.mp4
        |       |       |
        ...     ...     ...

        to the form required to use
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

        NOTE: Currently only the rgb videos of the hd_kinect (top down) are used.

    :param root_path: Path to unprocessed data
    :param target_path: Target path for processed data
    :param target_img_shape: Target image dimensions.
        Original image is up-/down-sampled accordingly.
    :return: None
    """

    assert os.path.isdir(root_path)
    os.makedirs(target_path, exist_ok=True)

    with open(target_path + '/annotations.txt', 'w') as annotations_file:
        pass

    for task_id, mime_task in enumerate(os.listdir(root_path)):

        print(f"\n##### Converting {mime_task} data:")
        time.sleep(1)

        task_joint_path = os.path.join(root_path, mime_task)
        task_target_path = os.path.join(target_path, mime_task)
        os.makedirs(task_target_path, exist_ok=True)

        if not os.path.isdir(task_joint_path):
            continue

        pbar = tqdm(os.listdir(task_joint_path))
        for sample_id, task_sample in enumerate(pbar):

            sample_joint_path = os.path.join(task_joint_path, task_sample)
            sample_target_path = os.path.join(task_target_path, '{:04d}/'.format(sample_id + 1))
            os.makedirs(sample_target_path, exist_ok=True)

            if not os.path.isdir(sample_joint_path):
                continue
            else:
                data_joint_path = os.path.join(sample_joint_path, 'hd_kinect_rgb.mp4')

            vidcap = cv2.VideoCapture(data_joint_path)
            success, image = vidcap.read()
            frame_count = 1
            while success:
                # Normalize image from (0, 255) to (0, 1)
                # image = image.astype(float)/255.0

                # Down-sample image to target image size
                image = cv2.resize(image, target_img_shape, interpolation=cv2.INTER_AREA)

                cv2.imwrite(sample_target_path + f'img_{frame_count:05}.jpg', image)
                success, image = vidcap.read()
                frame_count += 1

            with open(target_path + '/annotations.txt', 'a') as annotations_file:
                annotations_file.write(f"{mime_task}/{'{:04d}/'.format(sample_id + 1)} {1} {frame_count - 1} {task_id}\n")

        print()


if __name__ == "__main__":
    process_mime(
        root_path='/media/yannik/samsung_ssd/data/mime_unprocessed/',
        target_path='/media/yannik/samsung_ssd/data/mime_processed_256pix/',
        target_img_shape=(256, 256),
    )
