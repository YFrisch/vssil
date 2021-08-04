import io
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from torchvision import transforms


def gen_eval_imgs(sample_img: torch.Tensor,
                  target_img: torch.Tensor,
                  reconstruction: torch.Tensor,
                  key_points: torch.Tensor):
    fig, ax = plt.subplots(1, 3)
    viridis = cm.get_cmap('viridis', key_points.shape[1])
    ax[0].imshow(sample_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
    for kp in range(key_points.shape[1]):
        x = int(((key_points[0, kp, 0] + 1.0) / 2.0) * sample_img.shape[2])
        y = int(((key_points[0, kp, 1] + 1.0) / 2.0) * sample_img.shape[3])
        ax[0].scatter(x, y, marker='o', cmap=viridis)
    ax[0].set_title('sample')
    ax[1].imshow(target_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax[1].set_title('target')
    ax[2].imshow(reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy())
    ax[2].set_title('reconstruction')
    memory_buffer = io.BytesIO()
    plt.savefig(memory_buffer, format='png')
    memory_buffer.seek(0)
    pil_img = Image.open(memory_buffer)
    pil_to_tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)
    plt.close()
    return pil_to_tensor
