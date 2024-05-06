from utils import *
import imageio.v2 as imageio
import glob


def plot_vae_training_plot(train_losses, test_losses, title, fname):
    recon_loss_train, quantize_loss_train, = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    recon_loss_test, quantize_loss_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, recon_loss_train, label='recon_loss_train')
    plt.plot(x_train, quantize_loss_train, label='quantize_loss_train')
    plt.plot(x_test, recon_loss_test, label='recon_loss_test')
    plt.plot(x_test, quantize_loss_test, label='quantize_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)


def create_gif(dir_, anim_file):
    anim_file = f'{anim_file}-cvae.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(f'{dir_}/*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def save_results(fn):
    train_losses, test_losses, reconstructions, = fn()
    reconstructions, =  reconstructions.astype('float32')

    plot_vae_training_plot(train_losses, test_losses, f'Q2(Train Plot',
                           f'results/vae_train_plot.png')
    show_samples(reconstructions, title=f'Reconstructions',
                 fname=f'results/reconstructions.png')

    create_gif("test_images", "test")
    create_gif("train_images", "train")


def q3_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(
        train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses, f'Q3 Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/q3_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses, f'Q3 Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/q3_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q3 Dataset {dset_id} Samples',
                 fname=f'results/q3_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q3 Dataset {dset_id} Reconstructions',
                 fname=f'results/q3_dset{dset_id}_reconstructions.png')


def q4_a_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir(3)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(
        train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses, f'Q4(a) Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/q4_a_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,
                       f'Q4(a) Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/q4_a_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q4(a) Dataset {dset_id} Samples',
                 fname=f'results/q4_a_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q4(a) Dataset {dset_id} Reconstructions',
                 fname=f'results/q4_a_dset{dset_id}_reconstructions.png')


def q4_b_save_results(fn):
    part = 'b'
    data_dir = get_data_dir(3)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))

    train_losses, test_losses, samples, reconstructions = fn(train_data, test_data)
    samples, reconstructions = samples.astype('float32') * 255, reconstructions.astype('float32') * 255
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q4({part}) Train Plot',
                           f'results/q4_{part}_train_plot.png')
    show_samples(samples, title=f'Q4({part}) Samples',
                 fname=f'results/q4_{part}_samples.png')
    show_samples(reconstructions, title=f'Q4({part}) Reconstructions',
                 fname=f'results/q4_{part}_reconstructions.png')
