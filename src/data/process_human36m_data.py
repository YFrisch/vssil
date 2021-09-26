import os

import cv2


def process_human_36m(root_path: str,
                      target_path: str,
                      target_img_shape: tuple = (64, 64)):
    """ Brings the Human3.6M data from https://vision.imar.ro/human3.6m into the required form of
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

        # NOTE: The Human3.6M data is expected to be in the form
                root_path/
                    test/
                        S2/
                            Videos/
                                X.mp4
                                ...
                            ...
                        ...
                    train/
                        ...

    :param root_path: Path to the root directory of the Human3.6M data
    :param target_path: Path to save processed data to
    :param target_img_shape: Target image dimensions.
        Original image is up-/down-sampled accordingly.
    :return: None
    """
    assert os.path.isdir(root_path)
    os.makedirs(target_path, exist_ok=True)

    with open(os.path.join(target_path, 'annotations.txt'), 'w') as annotations_file:
        pass

    for mode in ['train', 'test']:
        mode_path = os.path.join(root_path, f'{mode}/')
        assert os.path.isdir(mode_path)
        for subject in os.listdir(mode_path):
            os.makedirs(os.path.join(target_path, subject), exist_ok=True)
            subject_path = os.path.join(mode_path, subject)
            subject_path = os.path.join(subject_path, 'Videos/')
            assert os.path.isdir(subject_path)
            for video_file in os.listdir(subject_path):
                video_path = os.path.join(subject_path, video_file)
                assert os.path.isfile(video_path)
                if not video_file.endswith('.mp4'):
                    continue

                video_file = video_file[:-4]
                video_file = video_file.replace(' ', '_')
                video_file = video_file.replace('.', '_')

                img_target_path = os.path.join(
                    os.path.join(target_path, subject),
                    video_file
                )

                os.makedirs(img_target_path, exist_ok=True)
                vidcap = cv2.VideoCapture(video_path)
                success, image = vidcap.read()
                if not success:
                    print(f'Could not read {video_path}!')
                frame_count = 0
                target_frame_count = 0
                sample_freq = 10
                print()
                while success:
                    print(f'{video_path}: {frame_count}\r', end="")
                    if frame_count % sample_freq == 0:
                        # Down-sample image
                        image = cv2.resize(image, dsize=target_img_shape, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(img_target_path, f'img_{target_frame_count:05}.jpg'), image)
                        target_frame_count += 1
                    success, image = vidcap.read()
                    frame_count += 1
                with open(os.path.join(target_path, 'annotations.txt'), 'a') as annotations_file:
                    annotations_file.write(f'{subject}/{video_file} 0 {target_frame_count - 1} 0\n')


if __name__ == "__main__":
    process_human_36m(
        root_path='/media/yannik/samsung_ssd/data/human_3.6m/',
        target_path='/media/yannik/samsung_ssd/data/human_36m_processed_256pix/',
        target_img_shape=(256, 256)
    )

