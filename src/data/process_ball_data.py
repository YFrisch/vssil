import os
import time

import cv2
from tqdm import tqdm
from natsort import natsort_keygen
natsort_key = natsort_keygen()


def process_ball(root_path: str,
                 target_path: str,
                 target_img_shape: tuple = (128, 128),
                 sample_freq: int = 1,
                 skip_existing: bool = True):
    """ TODO

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

    for mode in ['demo', 'train', 'valid']:
        mode_path = os.path.join(root_path, mode)
        if not os.path.isdir(mode_path):
            continue

        for exp in os.listdir(mode_path):

            exp_path = os.path.join(mode_path, exp)

            if not os.path.isdir(exp_path):
                continue

            target_frame_count = 0

            time.sleep(0.02)
            print(f"\n{mode_path}: {exp}")
            time.sleep(0.02)

            vid_target_path = f'{target_path}/{mode}/{exp}'

            if not skip_existing:
                os.makedirs(vid_target_path, exist_ok=True)
            else:
                try:
                    os.makedirs(vid_target_path, exist_ok=False)
                except OSError:
                    continue

            pbar = tqdm(sorted(os.listdir(exp_path), key=natsort_key))
            for frame_id, img in enumerate(pbar):

                if (frame_id + 1) % sample_freq == 0:

                    try:
                        img_source_path = os.path.join(exp_path, img)
                        img_target_path = os.path.join(vid_target_path, f'img_{target_frame_count:05}.jpg')

                        image = cv2.imread(img_source_path)
                        resized_image = cv2.resize(image, dsize=target_img_shape, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(img_target_path, resized_image)
                    except cv2.error:
                        continue

                    target_frame_count += 1

            with open(os.path.join(target_path, 'annotations.txt'), 'a') as annotations_file:
                annotations_file.write(f'{mode}/{exp} 0 {target_frame_count - 1} 0\n')


if __name__ == "__main__":
    process_ball(root_path='/media/yannik/samsung_ssd/data/data_Ball/data_Ball/',
                 target_path='/media/yannik/samsung_ssd/data/ball_processed_64pix',
                 target_img_shape=(64, 64),
                 sample_freq=1,
                 skip_existing=False)
