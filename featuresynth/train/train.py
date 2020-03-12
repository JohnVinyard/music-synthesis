from ..util.modules import zero_grad
from ..loss import hinge_generator_loss, hinge_discriminator_loss
from datetime import datetime
import torch
import zounds


class GeneratorTrainer(object):
    def __init__(
            self,
            generator,
            g_optim,
            discriminator,
            d_optim,
            loss,
            sub_loss=hinge_generator_loss):

        super().__init__()
        self.sub_loss = sub_loss
        self.loss = loss
        self.d_optim = d_optim
        self.discriminator = discriminator
        self.g_optim = g_optim
        self.generator = generator

    def train(self, samples, features):
        zero_grad(self.g_optim, self.d_optim)

        fake = self.generator(features)
        f_features, f_score = self.discriminator(fake, features)
        r_features, r_score = self.discriminator(samples, features)

        loss = self.loss(
            r_features, f_features, r_score, f_score, gan_loss=self.sub_loss)

        loss.backward()
        self.g_optim.step()
        try:
            fake = fake.data.cpu().numpy()
        except AttributeError:
            fake = {k:v.data.cpu().numpy() for k, v in fake.items()}
        return {'g_loss': loss.item(), 'fake': fake}


class DiscriminatorTrainer(object):
    def __init__(
            self,
            generator,
            g_optim,
            discriminator,
            d_optim,
            loss,
            sub_loss=hinge_discriminator_loss):

        super().__init__()
        self.sub_loss = sub_loss
        self.loss = loss
        self.d_optim = d_optim
        self.discriminator = discriminator
        self.g_optim = g_optim
        self.generator = generator

    def train(self, samples, features):
        zero_grad(self.g_optim, self.d_optim)

        fake = self.generator(features)
        _, f_score = self.discriminator(fake, features)
        _, r_score = self.discriminator(samples, features)

        loss = self.loss(r_score, f_score, gan_loss=self.sub_loss)
        loss.backward()

        self.d_optim.step()
        return {'d_loss': loss.item()}


def training_loop(batch_stream, experiment, device, loggers):
    start_time = datetime.utcnow()

    for i, batch in enumerate(batch_stream):
        preprocessed = experiment.preprocess_batch(batch)

        tensors = []
        for x in preprocessed:
            try:
                tensors.append(torch.from_numpy(x).to(device).float())
            except TypeError:
                x = {k:torch.from_numpy(v).to(device).float() for k, v in x.items()}
                tensors.append(x    )
        # tensors = [torch.from_numpy(x).to(device).float() for x in preprocessed]

        step = next(experiment.training_steps)
        step_result = step(*tensors)

        elapsed_time = datetime.utcnow() - start_time

        log_results = {}

        for logger in loggers:
            log_result = logger(
                experiment, preprocessed, step_result, i, elapsed_time)
            if log_result is None:
                continue
            else:
                log_results.update(log_result)

        yield i, elapsed_time, log_results
