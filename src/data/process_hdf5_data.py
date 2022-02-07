import os
import time
import shutil

import h5py
import cv2
from tqdm import tqdm


def process_hdf5_data(file_path: str, target_path: str):

    assert os.path.isfile(file_path)
    if os.path.exists(target_path) and os.path.isdir(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path + "/sim", exist_ok=False)

    with open(target_path + '/annotations.txt', 'w') as annotations_file:
        pass

    f = h5py.File(file_path, 'r')

    n_samples = len(f['data'].keys())

    print(f"##### Converting {n_samples} roboturk samples.")
    print()
    time.sleep(1)

    pbar = tqdm(f['data'].keys())
    for sample_id, sample_key in enumerate(pbar):

        sample_target_path = target_path + f"/sim/{'{:04d}/'.format(sample_id)}"

        os.makedirs(sample_target_path, exist_ok=False)

        # img = f['data']['demo_0']['obs']['image']
        img = f['data'][sample_key]['obs']['agentview_image']

        for frame_count, frame in enumerate(img):

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_target_path = sample_target_path + f'img_{frame_count + 1:05}.jpg'

            cv2.imwrite(filename=frame_target_path, img=frame)

        with open(target_path + '/annotations.txt', 'a') as annotations_file:
            annotations_file.write(f"{target_path}/{'{:04d}/'.format(sample_id + 1)} {1} {frame_count} {sample_id}\n")


if __name__ == "__main__":
    process_hdf5_data(
        file_path="/media/yannik/samsung_ssd/data/hdf5_data/image.hdf5",
        target_path="/media/yannik/samsung_ssd/data/roboturk_processed_84pix/"
    )



