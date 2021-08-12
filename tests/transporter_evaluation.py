import os

import torch
import yaml
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io.video import write_video

from src.agents.transporter_agent import TransporterAgent
from src.data.video_dataset import VideoFrameDataset, ImglistToTensor
from src.utils.argparse import parse_arguments
from src.utils.visualization import gen_eval_imgs

if __name__ == "__main__":

    args = parse_arguments()

    with open('/home/yannik/vssil/results/transporter/2021_8_6_16_0/config.yml', 'r') as stream:
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
        chckpt_path='/home/yannik/vssil/results/transporter/2021_8_6_16_0/checkpoints/chckpt_f0_e95.PTH'
    )

    with torch.no_grad():

        for i, (sample, label) in enumerate(eval_data_loader):
            sample_img = sample[:, :-3, ...].squeeze(0) - 0.5
            target_img = sample[:, 3:, ...].squeeze(0) - 0.5

            source_feature_maps = transporter_agent.model.encoder(sample_img)
            source_keypoints, source_gaussian_maps = transporter_agent.model.keypointer(sample_img)

            target_feature_maps = transporter_agent.model.encoder(target_img)
            target_keypoints, target_gaussian_maps = transporter_agent.model.keypointer(target_img)

            transported_features = transporter_agent.model.transport(
                source_gaussian_maps, target_gaussian_maps, source_feature_maps, target_feature_maps)

            reconstruction = transporter_agent.model.decoder(transported_features).clip(-0.5, 0.5)

            """
            img = None
            plt.figure()
            for frame_counter in range(0, sample_img.shape[0]):
                img_tensor = gen_eval_imgs(sample_img[frame_counter, ...].unsqueeze(0) + 0.5,
                                           target_img[frame_counter, ...].unsqueeze(0) + 0.5,
                                           reconstruction[frame_counter, ...].unsqueeze(0) + 0.5,
                                           target_keypoints[frame_counter, ...].unsqueeze(0))
                if img is None:
                    img = plt.imshow(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
                else:
                    img.set_data(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
                plt.pause(.1)
                plt.draw()

            """

            img_series_tensor = None
            for frame_counter in range(0, sample_img.shape[0]):
                print(f"Frame: {frame_counter}|{data_set.frames_per_segment}")
                img_tensor = gen_eval_imgs(sample_img[frame_counter, ...].unsqueeze(0) + 0.5,
                                           target_img[frame_counter, ...].unsqueeze(0) + 0.5,
                                           reconstruction[frame_counter, ...].unsqueeze(0) + 0.5,
                                           target_keypoints[frame_counter, ...].unsqueeze(0)).unsqueeze(0)

                if img_series_tensor is None:
                    img_series_tensor = img_tensor
                else:
                    img_series_tensor = torch.cat([img_series_tensor, img_tensor])

            img_series_tensor = torch.tensor(data=img_series_tensor*254, dtype=torch.uint8)
            #print(img_series_tensor.mean())
            print(img_series_tensor.shape)
            write_video(filename='eval_example.mp4',
                        video_array=img_series_tensor.squeeze().permute(0, 2, 3, 1)[..., :3],
                        fps=4,
                        video_codec='libx264')

            exit()



