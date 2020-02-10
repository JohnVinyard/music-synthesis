from ..util.modules import zero_grad, freeze, unfreeze
import torch


class GeneratorTrainer(object):
    def __init__(
            self,
            generator,
            g_optim,
            discriminator,
            d_optim,
            loss,
            feature_loss=True,
            feature_loss_scale=10):

        super().__init__()
        self.feature_loss_scale = feature_loss_scale
        self.feature_loss = feature_loss
        self.loss = loss
        self.d_optim = d_optim
        self.discriminator = discriminator
        self.g_optim = g_optim
        self.generator = generator

    def train(self, samples, features):
        batch_size = samples.shape[0]

        zero_grad(self.g_optim, self.d_optim)
        freeze(self.discriminator)
        unfreeze(self.generator)

        fake = self.generator(features)
        f_features, f_score = self.discriminator(fake)
        r_features, r_score = self.discriminator(samples)

        loss = self.loss(f_score)
        if self.feature_loss:
            feature_loss = 0
            for f_f, r_f in zip(f_features, r_features):
                feature_loss += torch.abs(f_f - r_f).sum() / \
                                f_f.view(batch_size, -1).contiguous().shape[-1]
            loss = loss + (self.feature_loss_scale * feature_loss)

        loss.backward()
        self.g_optim.step()
        return {'g_loss': loss.item(), 'fake': fake.data.cpu().numpy()}


class DiscriminatorTrainer(object):
    def __init__(self, generator, g_optim, discriminator, d_optim, loss):
        super().__init__()
        self.loss = loss
        self.d_optim = d_optim
        self.discriminator = discriminator
        self.g_optim = g_optim
        self.generator = generator

    def train(self, samples, features):
        zero_grad(self.g_optim, self.d_optim)
        freeze(self.generator)
        unfreeze(self.discriminator)

        fake = self.generator(features)
        f_features, f_score = self.discriminator(fake)
        r_features, r_score = self.discriminator(samples)

        loss = self.loss(r_score, f_score)
        loss.backward()
        self.d_optim.step()
        return {'d_loss': loss.item()}
