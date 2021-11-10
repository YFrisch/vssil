import numpy as np
import torch
import torch.nn.functional as F
import umap.umap_ as umap
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


from src.data.video_dataset import VideoFrameDataset, ImglistToTensor
from src.models.vqvae import VQ_VAE


if __name__ == "__main__":
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
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        random_shift=True,
        test_mode=True
    )

    print(len(data_set))
    len_train = int(0.8 * len(data_set))
    len_val = len(data_set) - len_train

    train_data_set, val_data_set = random_split(data_set, [len_train, len_val])

    train_data_loader = DataLoader(
        dataset=train_data_set,
        batch_size=32,
        shuffle=True
    )

    val_data_loader = DataLoader(
        dataset=val_data_set,
        batch_size=32,
        shuffle=True
    )

    model = VQ_VAE().to('cuda:0')

    optim = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(20000):
        (data, _) = next(iter(train_data_loader))
        data = data.squeeze(1).to('cuda:0')
        optim.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / torch.var(data)
        loss = recon_error + vq_loss
        loss.backward()

        optim.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    # train_res_recon_error_smooth = train_res_recon_error
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
    # train_res_perplexity_smooth = train_res_perplexity

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')

    plt.savefig('losses.png')

    model.eval()
    torch.save(model.state_dict(), 'vqvae_model.pth')

    (valid_originals, _) = next(iter(val_data_loader))
    valid_originals = valid_originals.squeeze(1).to('cuda:0')

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    (train_originals, _) = next(iter(train_data_loader))
    train_originals = train_originals.squeeze(1).to('cuda:0')
    vq_output_eval = model._pre_vq_conv(model._encoder(train_originals))
    _, train_quantize, _, _ = model._vq_vae(vq_output_eval)
    train_reconstructions = model._decoder(train_quantize)
    # _, train_reconstructions, _, _ = model._vq_vae(train_originals)

    def show(img):
        plt.figure()
        npimg = img.numpy()
        fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    show(make_grid(valid_reconstructions.cpu().data) + 0.5, )
    plt.savefig('valid_rec.png')
    show(make_grid(valid_originals.cpu() + 0.5))
    plt.savefig('valid_orig.png')

    plt.figure()

    proj = umap.UMAP(n_neighbors=3,
                     min_dist=0.1,
                     metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())

    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)

    plt.savefig('latent.png')
