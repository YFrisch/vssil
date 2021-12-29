import os
import time

import cv2
from tqdm import tqdm
from natsort import natsort_keygen

natsort_key = natsort_keygen()


def process_dmcs(root_path: str,
                 target_path: str,
                 target_img_shape: tuple = (128, 128),
                 sample_freq: int = 1,
                 skip_existing: bool = True):
    """ Converts .mp4 recordings from DeepMind Control Suite in the form

        root/
            env_task/
                1000.mp4
                2000.mp4
                ...

        into the form required for https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch


    :param root_path: Path to DMCS data
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

    for env_task_id, env_task in enumerate(os.listdir(root_path)):

        print(f"##### Converting {env_task} data:")

        env_task_path = os.path.join(root_path, env_task)

        if (not os.path.isdir(env_task_path)) or env_task.startswith('__'):
            print("Skipped.")
            continue

        for exp_id, exp in enumerate(os.listdir(env_task_path)):

            exp_path = os.path.join(env_task_path, exp)

            pbar = tqdm(os.listdir(exp_path))
            for sample_id, sample in enumerate(pbar):

                sample_path = os.path.join(exp_path, sample)
                if os.path.isdir(sample_path):
                    continue
                sample_target_path = f"{target_path}/{env_task}_{exp_id}/{'{:04d}/'.format(sample_id + 1)}/"
                os.makedirs(sample_target_path, exist_ok=True)

                vidcap = cv2.VideoCapture(sample_path)
                success, image = vidcap.read()
                frame_count = 1

                while success:
                    image = cv2.resize(image, target_img_shape, interpolation=cv2.INTER_AREA)

                    cv2.imwrite(sample_target_path + f'img_{frame_count:05}.jpg', image)
                    success, image = vidcap.read()
                    frame_count += 1

                with open(target_path + '/annotations.txt', 'a') as annotations_file:
                    annotations_file.write(
                        f"{env_task}_{exp_id}/{'{:04d}/'.format(sample_id + 1)} {1} {frame_count - 1} {env_task_id}\n")

            print()


if __name__ == "__main__":
    target_img_shape = (128, 128)
    process_dmcs(
        root_path='/media/yannik/samsung_ssd/data/deepmind_control_suite_unprocessed',
        target_path=f'/media/yannik/samsung_ssd/data/walker_processed_{target_img_shape[0]}pix',
        target_img_shape=target_img_shape,
        sample_freq=1,
        skip_existing=True
    )
