from torch.nn import functional as F
import numpy as np


def least_squares_generator_loss(j):
    return 0.5 * ((j - 1) ** 2).mean()


def hinge_generator_loss(j):
    return (-j).mean()


def least_squares_disc_loss(r_j, f_j):
    return 0.5 * (((r_j - 1) ** 2).mean() + (f_j ** 2).mean())


def hinge_discriminator_loss(r_j, f_j):
    return (F.relu(1 - r_j) + F.relu(1 + f_j)).mean()


def mel_gan_disc_loss(
        real_judgements,
        fake_judgements,
        gan_loss=least_squares_disc_loss):
    return sum(gan_loss(r, f) for r, f in zip(real_judgements, fake_judgements))


def mel_gan_feature_loss(real_features, fake_features):
    loss = 0
    # features are lists of lists

    # scale by the number of discriminators
    nd = (1 / len(real_features))

    for r_group, f_group in zip(real_features, fake_features):

        # also scale by the number of layers in this discriminator
        nl = (1 / len(r_group))

        for r_f, f_f in zip(r_group, f_group):


            # TODO: How should this be scaled?  There seems to be significant
            # disagreement between:
            #
            # the official implementation:
            # https://github.com/descriptinc/melgan-neurips/blob/master/scripts/train.py#L162
            # scaling by the number of discriminators and the number of layers
            # in each
            #
            # the paper suggesting that each is scaled by the "number
            # of units" in each layer
            #
            # the other popular implementation
            # https://github.com/seungwonpark/melgan/blob/master/utils/train.py#L79
            # scaling using lambda=10, which seems just plain wrong/counter to
            # what the paper says.

            # scale by the number of elements in this layer
            # n = (1 / float(np.prod(r_f.shape[1:])))

            l_loss = (nl * nd) * F.l1_loss(r_f, f_f)
            loss += l_loss

    return loss


def mel_gan_gen_loss(
        real_features,
        fake_features,
        real_judgements,
        fake_judgements,
        gan_loss=least_squares_generator_loss,
        feature_loss_weight=10):

    j_loss = sum(gan_loss(f) for r, f in zip(real_judgements, fake_judgements))

    f_loss = mel_gan_feature_loss(real_features, fake_features)
    return j_loss + (feature_loss_weight * f_loss)
