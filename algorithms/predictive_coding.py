import torch
from torch import nn
from torch.nn import functional as F

class Predictor(nn.Module):

    def __init__(self, layer_sizes, learning_rate=0.01):

        super(Predictor, self).__init__()

        self.mean_target = torch.zeros([1, layer_sizes[0]])
        self.covariance_target = torch.diag(torch.ones(layer_sizes[0]))

        self.covariance_observation = torch.diag(torch.ones(layer_sizes[-1]))

        self.layers = []

        for layer_idx in range(len(layer_sizes) - 1):
            self.layers.append(
                nn.Linear(in_features=layer_sizes[layer_idx], out_features=layer_sizes[layer_idx + 1])
            )

        self.learning_rate = learning_rate

    def set_trainability(self, flag):

        self.covariance_observation.requires_grad_(flag)
        self.mean_target.requires_grad_(flag)
        self.covariance_target.requires_grad_(flag)

        for layer in self.layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(flag)

    def calculate_log_normal(self, vector, mean, covariances):

        return 0.5 * ((- torch.log(torch.det(covariances))) -
                      torch.mm(torch.mm((vector - mean), torch.inverse(covariances)), torch.t(vector - mean)))

    def calculate_function(self, value):

        for layer in self.layers:
            value = F.sigmoid(layer(value))

        return value

    def calculate_error(self, vector, mean, covariances):

        return torch.mm(torch.inverse(covariances), torch.t(vector - mean))

    def train_parameters(self, observation, target):

        self.set_trainability(True)

        for step in range(0, 10):

            likelihood = self.calculate_log_normal(vector=observation,
                                                   mean=self.calculate_function(target),
                                                   covariances=self.covariance_observation)

            prior = self.calculate_log_normal(vector=target,
                                              mean=self.mean_target,
                                              covariances=self.covariance_target)

            posterior = prior + likelihood
            posterior.backward()

            self.mean_target.data += self.mean_target.grad.data * self.learning_rate
            self.mean_target.grad.data.zero_()

            self.covariance_observation.data += self.learning_rate * torch.mm(
                observation - self.calculate_function(target),
                torch.t(observation - self.calculate_function(target)))
            self.covariance_observation.grad.data.zero_()

            self.covariance_target.data += self.learning_rate * torch.mm(
                target - self.mean_target,
                torch.t(target - self.mean_target))
            self.covariance_target.grad.data.zero_()

            for layer in self.layers:
                for parameter in layer.parameters():
                    parameter.data += parameter.grad.data * self.learning_rate
                    parameter.grad.data.zero_()

    def predict_target(self, observation):

        self.set_trainability(False)

        optimizable_target = torch.zeros([1, 10], requires_grad=True)

        '''
        flag_stop = 0
        current_error = np.inf
        '''

        for step in range(0, 100):

            likelihood = self.calculate_log_normal(vector=observation,
                                                   mean=self.calculate_function(optimizable_target),
                                                   covariances=self.covariance_observation)

            prior = self.calculate_log_normal(vector=optimizable_target,
                                              mean=self.mean_target,
                                              covariances=self.covariance_target)

            posterior = prior + likelihood

            posterior.backward()

            optimizable_target.data += optimizable_target.grad.data * self.learning_rate
            optimizable_target.grad.data.zero_()
            '''
            error_value = torch.mean(torch.pow(
                self.calculate_error(optimizable_target, self.mean_target, self.covariance_target),
                exponent=2)).data.numpy().flatten()[0]

            if error_value < current_error:
                current_error = error_value
                flag_stop = 0
            else:
                flag_stop += 1

            if flag_stop == 5:
                break
            '''

        return optimizable_target

    def predict_observation(self, target):

        self.set_trainability(False)

        optimizable_observation = torch.zeros([1, 784], requires_grad=True)

        for step in range(0, 1000):

            likelihood = self.calculate_log_normal(vector=optimizable_observation,
                                                   mean=self.calculate_function(target),
                                                   covariances=self.covariance_observation)

            prior = self.calculate_log_normal(vector=target,
                                              mean=self.mean_target,
                                              covariances=self.covariance_target)

            posterior = prior + likelihood

            posterior.backward()

            optimizable_observation.data += optimizable_observation.grad.data * self.learning_rate
            optimizable_observation.grad.data.zero_()

        return optimizable_observation


if __name__ == '__main__':

    import numpy as np
    from observations import mnist
    from matplotlib import pyplot as plt
    from sklearn.metrics.classification import classification_report, accuracy_score

    predictor = Predictor(layer_sizes=[10, 100, 300, 784])

    (x_train, y_train), (x_test, y_test) = mnist('../data/mnist')

    EPOCHS = 100

    for i in range(EPOCHS):

        for idx in range(0, x_train.shape[0]):

            print('Started to train picture no.' + str(idx) + ' in epoch no.' + str(i))

            predictor.train_parameters(observation=torch.from_numpy(x_train[idx].reshape(1, 784) / 255.0).float(),
                                       target=torch.from_numpy(np.eye(10)[y_train[idx]].reshape(1, 10)).float())

            if idx % 10000 == 0:

                pred_list = []

                for index in range(500):
                    print('Predicting sample no.', index)
                    pred_list.append(
                        float(np.argmax(predictor.predict_target(
                            observation=torch.from_numpy(x_test[index].reshape(1, 784) / 255.0).float()).data.numpy(),
                                        axis=1))
                    )

                print(classification_report(y_test[0:500], pred_list))
                print('Accuracy:', accuracy_score(y_test[0:500], pred_list))

                predictions = predictor.predict_observation(
                    target=torch.from_numpy(np.eye(10)[y_train[2]].reshape(1, 10)).float()).data

                print('Real observation value no.1:', y_train[2])
                plt.imshow(predictions.reshape([28, 28]))
                plt.show()

                predictions = predictor.predict_observation(
                    target=torch.from_numpy(np.eye(10)[y_train[3]].reshape(1, 10)).float()).data

                print('Real observation value no.2:', y_train[3])
                plt.imshow(predictions.reshape([28, 28]))
                plt.show()

                predictions = predictor.predict_observation(
                    target=torch.from_numpy(np.eye(10)[y_train[5]].reshape(1, 10)).float()).data

                print('Real observation value no.2:', y_train[5])
                plt.imshow(predictions.reshape([28, 28]))
                plt.show()
