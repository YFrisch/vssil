import os

import cv2


def process_movi(root_path: str,
                 target_path: str,
                 target_img_shape: tuple = (64, 64)):
    """ Brings the MoVi data from https://dataverse.scholarsportal.info/dataverse/MoVi
        into the required form of https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

        # NOTE: The MoVi data is expected to be in the form

            root_path/
                test/
                    F_CP1_Subject_1.mp4
                    F_CP1_Subject_2.mp4
                    ...
                train/
                    F_CP1_Subject_10.mp4
                    F_CP1_Subject_11.mp4
                    ...

    :param root_path: Path to the root directory of the MoVi data
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