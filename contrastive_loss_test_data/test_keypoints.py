import torch


def get_perfect_keypoints(T: int):
    """ Use manual seed to use the same key-points each time... """
    kpt_coordinates = torch.zeros((T, 7, 3))

    # T=0
    kpt_coordinates[0, 0, :] = torch.tensor([0.7, -0.15, 1.0])  # Head
    kpt_coordinates[0, 1, :] = torch.tensor([0.35, -0.1, 1.0])  # Upper Body
    kpt_coordinates[0, 2, :] = torch.tensor([0.0, 0.0, 1.0])  # Lower Body
    kpt_coordinates[0, 3, :] = torch.tensor([0.1, 0.4, 1.0])  # Left knee
    kpt_coordinates[0, 4, :] = torch.tensor([-0.4, -0.15, 1.0])  # Right knee
    kpt_coordinates[0, 5, :] = torch.tensor([-0.65, -0.4, 1.0])  # Right foot
    kpt_coordinates[0, 6, :] = torch.tensor([-0.25, 0.5, 1.0])  # Left foot

    # T=1
    kpt_coordinates[1, 0, :] = torch.tensor([0.7, -0.15, 1.0])
    kpt_coordinates[1, 1, :] = torch.tensor([0.35, -0.1, 1.0])
    kpt_coordinates[1, 2, :] = torch.tensor([0.0, 0.0, 1.0])
    kpt_coordinates[1, 3, :] = torch.tensor([0.05, 0.4, 1.0])
    kpt_coordinates[1, 4, :] = torch.tensor([-0.4, -0.15, 1.0])
    kpt_coordinates[1, 5, :] = torch.tensor([-0.65, -0.4, 1.0])
    kpt_coordinates[1, 6, :] = torch.tensor([-0.25, 0.5, 1.0])

    # T=2
    kpt_coordinates[2, 0, :] = torch.tensor([0.75, -0.05, 1.0])
    kpt_coordinates[2, 1, :] = torch.tensor([0.35, -0.05, 1.0])
    kpt_coordinates[2, 2, :] = torch.tensor([0.0, 0.0, 1.0])
    kpt_coordinates[2, 3, :] = torch.tensor([0.0, 0.45, 1.0])
    kpt_coordinates[2, 4, :] = torch.tensor([-0.4, -0.15, 1.0])
    kpt_coordinates[2, 5, :] = torch.tensor([-0.6, -0.4, 1.0])
    kpt_coordinates[2, 6, :] = torch.tensor([-0.35, 0.4, 1.0])

    # T=3
    kpt_coordinates[3, 0, :] = torch.tensor([0.75, 0.0, 1.0])
    kpt_coordinates[3, 1, :] = torch.tensor([0.4, 0.0, 1.0])
    kpt_coordinates[3, 2, :] = torch.tensor([0.05, 0.05, 1.0])
    kpt_coordinates[3, 3, :] = torch.tensor([-0.1, 0.4, 1.0])
    kpt_coordinates[3, 4, :] = torch.tensor([-0.35, -0.15, 1.0])
    kpt_coordinates[3, 5, :] = torch.tensor([-0.6, -0.45, 1.0])
    kpt_coordinates[3, 6, :] = torch.tensor([-0.45, 0.3, 1.0])

    return kpt_coordinates


def get_bad_keypoints(T: int):
    """ Use manual seed to use the same key-points each time... """
    kpt_coordinates = torch.zeros((T, 7, 3))

    # T=0
    kpt_coordinates[0, 0, :] = torch.tensor([0.7, -0.15, 1.0])  # Head
    kpt_coordinates[0, 1, :] = torch.tensor([0.35, -0.1, 1.0])  # Upper Body
    kpt_coordinates[0, 2, :] = torch.tensor([0.0, 0.0, 1.0])  # Lower Body
    kpt_coordinates[0, 3, :] = torch.tensor([0.1, 0.4, 1.0])  # Left knee
    kpt_coordinates[0, 4, :] = torch.tensor([-0.4, -0.15, 1.0])  # Right knee
    kpt_coordinates[0, 5, :] = torch.tensor([-0.65, -0.4, 1.0])  # Right foot
    kpt_coordinates[0, 6, :] = torch.tensor([-0.25, 0.5, 1.0])  # Left foot

    # T=1
    kpt_coordinates[1, 0, :] = torch.tensor([0.7, -0.15, 1.0])
    kpt_coordinates[1, 1, :] = torch.tensor([0.35, -0.1, 1.0])
    kpt_coordinates[1, 2, :] = torch.tensor([0.0, 0.0, 1.0])
    kpt_coordinates[1, 3, :] = torch.tensor([0.05, 0.4, 1.0])
    kpt_coordinates[1, 4, :] = torch.tensor([-0.4, -0.15, 1.0])
    kpt_coordinates[1, 5, :] = torch.tensor([-0.65, -0.4, 1.0])
    kpt_coordinates[1, 6, :] = torch.tensor([-0.25, 0.5, 1.0])

    # T=2
    kpt_coordinates[2, 0, :] = torch.tensor([-0.7, -0.15, 1.0])
    kpt_coordinates[2, 1, :] = torch.tensor([-0.35, -0.1, 1.0])
    kpt_coordinates[2, 2, :] = torch.tensor([0.0, 0.0, 1.0])
    kpt_coordinates[2, 3, :] = torch.tensor([0.0, 0.45, 1.0])
    kpt_coordinates[2, 4, :] = torch.tensor([-0.4, -0.15, 1.0])
    kpt_coordinates[2, 5, :] = torch.tensor([-0.6, -0.4, 1.0])
    kpt_coordinates[2, 6, :] = torch.tensor([-0.35, 0.4, 1.0])

    # T=3
    kpt_coordinates[3, 0, :] = torch.tensor([0.7, -0.15, 1.0])
    kpt_coordinates[3, 1, :] = torch.tensor([-0.35, -0.1, 1.0])
    kpt_coordinates[3, 2, :] = torch.tensor([0.05, 0.05, 1.0])
    kpt_coordinates[3, 3, :] = torch.tensor([-0.1, 0.4, 1.0])
    kpt_coordinates[3, 4, :] = torch.tensor([-0.35, -0.15, 1.0])
    kpt_coordinates[3, 5, :] = torch.tensor([-0.6, -0.45, 1.0])
    kpt_coordinates[3, 6, :] = torch.tensor([-0.45, 0.3, 1.0])

    return kpt_coordinates


def get_random_keypoints(T: int):
    kpt_coordinates = torch.rand((T, 7, 3))
    kpt_coordinates[..., :2] *= 2.0
    kpt_coordinates[..., :2] -= 1.0
    kpt_coordinates[..., 2] = 1.0
    return kpt_coordinates
