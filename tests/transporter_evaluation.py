import os

import torch
import yaml
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from src.agents.transporter_agent import TransporterAgent
from src.data.video_dataset import VideoFrameDataset, ImglistToTensor
from src.utils.argparse import parse_arguments
from src.utils.viz import gen_eval_imgs

if __name__ == "__main__":

    args = parse_arguments()

    with open('/home/yannik/vssil/results/transporter/2021_8_3_22_46/config.yml', 'r') as stream:
        transporter_conf = yaml.safe_load(stream)
        transporter_conf['device'] = 'cpu'
        print(transporter_conf['log_dir'])

    preprocess = transforms.Compose([
            ImglistToTensor(),
        ])

    data_set = VideoFrameDataset(
            root_path=args.data,
            annotationfile_path=os.path.join(args.data, 'annotations.txt'),
            num_segments=1,
            frames_per_segment=100,
            imagefile_template='img_{:05d}.jpg',
            transform=preprocess,
            random_shift=True,
            test_mode=False
        )

    eval_data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True
    )

    transporter_agent = TransporterAgent(dataset=data_set, config=transporter_conf)
    transporter_agent.load_checkpoint(
        chckpt_path='/home/yannik/vssil/results/transporter/2021_8_3_22_46/checkpoints/chckpt_f0_e20.PTH'
    )

    for i, p in enumerate(transporter_agent.model.parameters()):
        print(f'i: {i}\t mean: {p.data.mean()}\t req. grad: {p.requires_grad}')
    exit()

    with torch.no_grad():

        for i, (sample, label) in enumerate(eval_data_loader):
            sample_img = sample[:, :-3, ...].squeeze(0)
            target_img = sample[:, 3:, ...].squeeze(0)
            # sample_img, target_img = transporter_agent.preprocess(x=sample, label=label, config=transporter_conf)

            key_point_maps = transporter_agent.model.keypointer(sample_img)
            key_points = transporter_agent.model._keypoint_loc_mean(key_point_maps)
            reconstruction = transporter_agent.model(sample_img, target_img).clip(0.0, 1.0)

            img = None
            plt.figure()
            for frame_counter in range(0, sample_img.shape[0]):
                img_tensor = gen_eval_imgs(sample_img[frame_counter, ...].unsqueeze(0),
                                           target_img[frame_counter, ...].unsqueeze(0),
                                           reconstruction[frame_counter, ...].unsqueeze(0),
                                           key_points[frame_counter, ...].unsqueeze(0))
                if img is None:
                    img = plt.imshow(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
                else:
                    img.set_data(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.pause(.01)
                plt.draw()

            exit()




