import os
import time

import cv2
from tqdm import tqdm
from natsort import natsort_keygen
natsort_key = natsort_keygen()


def process_simitate(root_path: str,
                     target_path: str,
                     target_img_shape: tuple = (128, 128),
                     sample_freq: int = 1,
                     skip_existing: bool = True):

    """ Converts the SIMITATE data in the form

    root/
        basic_motions/
            circle/
                andrea/
                    circle_2018-09-17-21-28-38/
                        ...
                        _kinect2_qhd_image_color_rect_compressed/
                            frame1537212531309996486.jpg
                            frame1537212531339020622.jpg
                            ...
                        ...
                    circle_2018-09-17-21-28-51/
                    ...
                ivanna/
                ...
            heart/
            ...
        ...

    into the form required for https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch


    :param root_path: Path to SIMITATE data
    :param target_path: Target path for processed data
    :param target_img_shape: Target shape for processed images
                             Images are interpolated, if required.
    :param sample_freq: Sample frequency for target image series. Set to 1 to use all frames.
    :param skip_existing: Set true to not re-write existing folders at the target path
    :return:
    """

    assert os.path.isdir(root_path)
    os.makedirs(target_path, exist_ok=True)

    with open(os.path.join(target_path, 'annotations.txt'), 'w') as annotations_file:
        pass

    for motion in os.listdir(root_path):
        joint_path1 = os.path.join(root_path, motion)
        if not os.path.isdir(joint_path1):
            continue
        for movement in os.listdir(joint_path1):
            joint_path2 = os.path.join(joint_path1, movement)
            for subject in os.listdir(joint_path2):
                joint_path3 = os.path.join(joint_path2, subject)
                for exp_id, exp in enumerate(os.listdir(joint_path3)):
                    joint_path4 = os.path.join(joint_path3, exp)
                    joint_path5 = os.path.join(joint_path4, '_kinect2_qhd_image_color_rect_compressed')
                    print(joint_path5)
                    assert os.path.isdir(joint_path5)

                    target_frame_count = 0

                    time.sleep(0.02)
                    print(f"\n{motion}: {movement}: {subject}: {exp_id}")
                    time.sleep(0.02)

                    vid_target_path = f'{target_path}/{motion}_{movement}/{subject}_{exp_id}/'

                    if not skip_existing:
                        os.makedirs(vid_target_path, exist_ok=True)
                    else:
                        try:
                            os.makedirs(vid_target_path, exist_ok=False)
                        except OSError:
                            continue

                    pbar = tqdm(sorted(os.listdir(joint_path5), key=natsort_key))
                    for frame_id, img in enumerate(pbar):

                        if not frame_id % sample_freq:

                            try:
                                img_source_path = os.path.join(joint_path5, img)
                                img_target_path = os.path.join(vid_target_path, f'img_{target_frame_count:05}.jpg')

                                image = cv2.imread(img_source_path)
                                resized_image = cv2.resize(image, dsize=target_img_shape, interpolation=cv2.INTER_AREA)
                                cv2.imwrite(img_target_path, resized_image)
                            except cv2.error:
                                continue

                            target_frame_count += 1

                    with open(os.path.join(target_path, 'annotations.txt'), 'a') as annotations_file:
                        annotations_file.write(f'{motion}_{movement}/{subject}_{exp_id} 0 {target_frame_count - 1} 0\n')


if __name__ == "__main__":
    process_simitate(root_path='/media/yannik/samsung_ssd/data/simitate/',
                     target_path='/media/yannik/samsung_ssd/data/simitate_processed_64pix/',
                     target_img_shape=(64, 64),
                     sample_freq=5)
