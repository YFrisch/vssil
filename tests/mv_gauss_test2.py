import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms


from src.models.vqvae import VQ_VAE_KPT
from src.losses.temporal_separation_loss import temporal_separation_loss
from src.losses.pixelwise_contrastive_loss_2 import pixelwise_contrastive_loss
from src.data.video_dataset import VideoFrameDataset, ImglistToTensor


preprocess = transforms.Compose([
        # NOTE: The first transform already converts the range to (0, 1)
        ImglistToTensor(),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.3),
        ])
    ])

data_set = VideoFrameDataset(
    root_path='/media/yannik/samsung_ssd/data/simitate_processed_128pix',
    annotationfile_path='/media/yannik/samsung_ssd/data/simitate_processed_128pix/annotations.txt',
    num_segments=1,
    frames_per_segment=8,
    imagefile_template='img_{:05d}.jpg',
    transform=preprocess,
    random_shift=True,
    test_mode=True
)

print(len(data_set))
len_train = int(0.8 * len(data_set))
len_val = len(data_set) - len_train

train_data_set, val_data_set = random_split(data_set, [len_train, len_val])

batch_size = 8

train_data_loader = DataLoader(
    dataset=train_data_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

dev = 'cuda:0'

kpt_net = VQ_VAE_KPT(
    batch_size=batch_size,
    time_steps=8,
    num_embeddings=512,
    heatmap_width=32,
    embedding_dim=64,
    device=dev
)

cfg = {
    'training': {
        'separation_loss_scale': 2.0,
        'separation_loss_sigma': 0.02,
        'pixelwise_contrastive_time_window': 5,
        'pixelwise_contrastive_alpha': 5.0,
        'pixelwise_contrastive_scale': 10.0
    },
    'model': {
        'n_feature_maps': 64
    }
}

iter_data = iter(train_data_loader)

optim = torch.optim.Adam(kpt_net.parameters(), lr=0.001, amsgrad=False)

for i in range(50000):

    try:
        (data, _) = next(iter_data)
    except StopIteration:
        iter_data = iter(train_data_loader)
        (data, _) = next(iter_data)

    img_inp = data.to(dev)

    optim.zero_grad()

    vq_loss, data_recon, gmap_recon, perplexity, gmaps, fmaps, kpts = kpt_net(img_inp, verbose=False)
    recon_error = F.mse_loss(data_recon, img_inp) / torch.var(img_inp)
    gmap_recon_error = F.mse_loss(gmap_recon, img_inp) / torch.var(img_inp)
    sep_error = temporal_separation_loss(cfg, coords=kpts) * cfg['training']['separation_loss_scale']
    pc_error = pixelwise_contrastive_loss(
        keypoint_coordinates=kpts,
        image_sequence=img_inp,
        feature_map_seq=gmaps,
        time_window=cfg['training']['pixelwise_contrastive_time_window'],
        alpha=cfg['training']['pixelwise_contrastive_alpha'],
        verbose=False
    ) * cfg['training']['pixelwise_contrastive_scale']
    L = recon_error + vq_loss + gmap_recon_error + sep_error + pc_error
    L.backward()

    optim.step()

    if i % 100 == 0:

        print(f'Iter {i}\t Rec. loss {recon_error}\t VQ loss {vq_loss}\t '
              f'G Rec loss {gmap_recon_error}\t Sep loss {sep_error}\t '
              f'PC loss {pc_error.item()}')

    if i % 5000 == 0:

        fig, ax = plt.subplots(8, 8, figsize=(25, 15))
        ax = ax.flat
        for c in range(0, gmaps.shape[1]):
            im = gmaps[0, 0, c, :, :]
            ax[c].imshow(im.detach().cpu().numpy(), cmap='gray')
        plt.savefig(f'test_plots/gmaps_i{i}.png')

        fig, ax = plt.subplots(8, 8, figsize=(25, 15))
        ax = ax.flat
        for c in range(0, fmaps.shape[1]):
            im = fmaps[0, 0, c, :, :]
            ax[c].imshow(im.detach().cpu().numpy(), cmap='gray')
        plt.savefig(f'test_plots/fmaps_i{i}.png')

        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        ax[0].imshow(img_inp[0, 0, ...].cpu().numpy().transpose(1, 2, 0))
        ax[1].imshow(data_recon[0, 0, ...].detach().cpu().numpy().transpose(1, 2, 0))
        ax[2].imshow(gmap_recon[0, 0,  ...].detach().cpu().numpy().transpose(1, 2, 0))
        plt.savefig(f'test_plots/rec_i{i}.png')




