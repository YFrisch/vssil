import os

import cv2
import matplotlib.pyplot as plt


def process_vssil_data(root_path: str,
                       target_path: str,
                       target_img_shape: tuple = (256, 256)):

    """ Brings VSSIL data into the form required for
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch


    :param root_path:
    :param target_path:
    :param target_img_shape:
    :return:
    """

    assert os.path.isdir(root_path)
    os.makedirs(target_path, exist_ok=True)

    with open(os.path.join(target_path, 'annotations.txt'), 'w') as annotations_file:
        pass

    for exp in os.listdir(root_path):
        os.makedirs(os.path.join(target_path, exp), exist_ok=True)
        exp_path = os.path.join(root_path, exp)
        assert os.path.isdir(exp_path)
        for video_file in os.listdir(exp_path):
            video_path = os.path.join(exp_path, video_file)
            assert os.path.isfile(video_path)
            if not video_file.endswith('.mp4'):
                continue

            video_file = video_file[:-4]

            img_target_path = os.path.join(
                os.path.join(target_path, exp),
                video_file
            )

            os.makedirs(img_target_path, exist_ok=True)

            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            if not success:
                print(f'Could not read {video_path}!')
            frame_count = 0
            target_frame_count = 0
            sample_freq = 1
            print()
            while success:
                print(f'{video_path}: {frame_count}\r', end="")
                if frame_count % sample_freq == 0:
                    # Crop image // Img is in (H, W, C) = (1080, 1920, C)
                    if exp.startswith("subj"):
                        #image = cv2.cvtColor(image[160:1040, 280:1500, :], cv2.COLOR_BGR2RGB)
                        image = image[160:1040, 280:1500, :]
                    elif exp.startswith("tiago"):
                        #image = cv2.cvtColor(image[130:980, 380:1650, :], cv2.COLOR_BGR2RGB)
                        image = image[130:980, 380:1650, :]
                    else:
                        pass
                    # Down-sample image
                    image = cv2.resize(image, dsize=target_img_shape, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(img_target_path, f'img_{target_frame_count:05}.jpg'), image)
                    target_frame_count += 1
                success, image = vidcap.read()
                frame_count += 1
            with open(os.path.join(target_path, 'annotations.txt'), 'a') as annotations_file:
                annotations_file.write(f'{exp}/{video_file} 0 {target_frame_count - 1} 0\n')


if __name__ == "__main__":

    # target_shape = (384, 216)
    target_shape = (256, 256)

    process_vssil_data(
        root_path="/home/yannik/Videos/vssil/final",
        # target_path=f"/home/yannik/Videos/vssil/vssil_processed_{target_shape[0]}x{target_shape[1]}pix",
        target_path=f"/home/yannik/Videos/vssil/vssil_processed_{target_shape[0]}pix",
        target_img_shape=target_shape
    )