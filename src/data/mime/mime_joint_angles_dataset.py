import os
import ast

import torch

from .mime_base import MimeBase


class MimeJointAngles(MimeBase):

    """ Wrapper for the joint angle data of the MIME dataset.

        TODO: Handle the gripper data!
    """

    def __init__(self,
                 base_path: str = os.path.join(os.getcwd(), '/datasets'),
                 tasks: str = 'stir',
                 start_ind: int = 0,
                 stop_ind: int = -1,
                 timesteps_per_sample: int = -1,
                 overlap: int = 20
                 ):

        print(f"##### Loading MIME dataset of joint angles for task '{tasks}'.")

        self.joint_header = ['left_w0', 'left_w1', 'left_w2', 'right_s0', 'right_s1', 'right_w0',
                             'right_w1', 'head_pan', 'right_w2', 'head_nod', 'torso_t0', 'left_e0',
                             'left_e1', 'left_s0', 'left_s1', 'right_e0', 'right_e1']

        super(MimeJointAngles, self).__init__(sample_file_name="joint_angles.txt",
                                              base_path=base_path,
                                              tasks=tasks,
                                              name="joint_angles",
                                              start_ind=start_ind,
                                              stop_ind=stop_ind,
                                              timesteps_per_sample=timesteps_per_sample,
                                              overlap=overlap)

    def read_sample(self, path: str) -> (torch.Tensor, int):
        """ Reads the joint-angle data and end effector state for the given path.

            TODO: Handle gripper data!
        """
        joint_tensor = None
        try:
            with open(path) as joint_angles_file:

                for lined_id, line in enumerate(joint_angles_file.readlines()):
                    joint_dict = ast.literal_eval(line)
                    _joint_tensor = torch.tensor(list(joint_dict.values())).unsqueeze(0)
                    if joint_tensor is None:
                        joint_tensor = _joint_tensor
                    elif _joint_tensor.shape[1] == joint_tensor.shape[1]:
                        joint_tensor = torch.cat([joint_tensor, _joint_tensor], dim=0)
                    else:
                        # TODO: This skips some entries like {"l_gripper_l_finger_joint": 0.019992780607938767}
                        pass

        except RuntimeError as e:
            print(f"\n\n\nCould not read joint angles at {path}:")
            print(e)
            print("\n\n\n")

        return joint_tensor.unsqueeze(1).unsqueeze(1), joint_tensor.shape[0]
