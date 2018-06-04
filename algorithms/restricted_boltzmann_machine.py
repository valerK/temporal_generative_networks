import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

import numpy as np

class RestrictedBoltzmannMachine(nn.Module):

    def __init__(self, visible_layer_size, hidden_layer_size, learning_rate=0.01, gibbs_steps=50):

        super(RestrictedBoltzmannMachine, self).__init__()

        self.weights = torch.zeros([visible_layer_size, hidden_layer_size])
        self.visible_bias = torch.zeros([visible_layer_size, 1])
        self.hidden_bias = torch.zeros([hidden_layer_size, 1])

        self.learning_rate = learning_rate
        self.steps = gibbs_steps

    def sample_hidden(self, visible):
        probs = F.sigmoid(torch.t(torch.mm(torch.t(visible), self.weights)) + self.hidden_bias)

        return probs, Bernoulli(probs).sample()

    def sample_visible(self, hidden):
        probs = F.sigmoid(torch.mm(self.weights, hidden) + self.visible_bias)

        return probs, Bernoulli(probs).sample()

    def calculate_energy(self, visible, hidden):

        return - torch.mm(torch.t(self.visible_bias), visible) - torch.mm(torch.t(self.hidden_bias), hidden) \
               - torch.mm(torch.mm(torch.t(visible), self.weights), hidden)

    def forward(self, samples):

        for sample_idx in range(len(samples)):

            burn_in = 0
            min_value = np.inf

            visible = torch.from_numpy(samples[sample_idx].data.numpy().reshape([-1, 1]))

            for step in range(self.steps):

                _, hidden = self.sample_hidden(visible=visible)
                _, visible = self.sample_visible(hidden=hidden)

                energy = self.calculate_energy(visible=visible, hidden=hidden).data.numpy().flatten()

                burn_in += 1

                if energy < min_value and burn_in > 30:
                    min_value = energy
                    samples[sample_idx] = visible

        return samples

    def optimize(self, samples, predictions):

        for sample_idx in range(len(samples)):

            hidden_probs_t0, _ = self.sample_hidden(visible=samples[sample_idx])
            hidden_probs_tk, _ = self.sample_hidden(visible=predictions[sample_idx])

            self.weights += self.learning_rate * (
                        torch.mm(samples[sample_idx], torch.t(hidden_probs_t0)) -
                        torch.mm(predictions[sample_idx], torch.t(hidden_probs_tk)))
            self.visible_bias += self.learning_rate * (samples[sample_idx] - predictions[sample_idx])
            self.hidden_bias += self.learning_rate * (hidden_probs_t0 - hidden_probs_tk)


if __name__ == '__main__':

    from observations import mnist
    from matplotlib import pyplot as plt

    rbm = RestrictedBoltzmannMachine(visible_layer_size=784,
                                     hidden_layer_size=500,
                                     learning_rate=0.01,
                                     gibbs_steps=100)

    (x_train, y_train), (x_test, y_test) = mnist('../data/mnist')

    EPOCHS = 100
    BATCH_SIZE = 10

    for i in range(EPOCHS):

        for idx in range(0, x_train.shape[0], BATCH_SIZE):

            print('Started to train batch no.' + str(idx / BATCH_SIZE) + ' in epoch no.' + str(i))

            batch = x_train[idx:idx+BATCH_SIZE].copy() / 255.0
            batch[batch >= 0.2] = 1.0
            batch[batch != 1.0] = 0.0

            preds = rbm(torch.from_numpy(batch.reshape([BATCH_SIZE, 784, 1])).float())
            rbm.optimize(samples=torch.from_numpy(batch.reshape([BATCH_SIZE, 784, 1])).float(),
                         predictions=preds)

            if idx % 10000 == 0:
                for pic_idx in range(batch.shape[0]):
                    plt.imshow(batch[pic_idx].reshape(28, 28))
                    plt.show()
                    plt.imshow(preds.data.numpy()[pic_idx].reshape(28, 28))
                    plt.show()
