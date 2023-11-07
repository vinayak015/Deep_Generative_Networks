from utils import *
import imageio.v2 as imageio
import glob


def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

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
    train_losses, test_losses, samples, reconstructions, interpolations = fn()
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype(
        'float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q2(Train Plot',
                           f'results/vae_train_plot.png')
    show_samples(samples, title=f'Samples',
                 fname=f'results/samples.png')
    show_samples(reconstructions, title=f'Reconstructions',
                 fname=f'results/reconstructions.png')
    show_samples(interpolations, title=f'Interpolations',
                 fname=f'results/interpolations.png')
    create_gif("test_images", "test")
    create_gif("train_images", "train")
