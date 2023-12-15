# from teenygrad.tensor import Tensor
# from teenygrad.nn.optim import SGD
from mygrad.tensor import Tensor
from mygrad.nn.optim import SGD
import numpy as np
import math

def generate_data(size, noise_perc):
    # Generate dosages between 0 and 1
    dosages = np.linspace(0, 1, size)

    # Generate corresponding efficacy values based on the described relationship
    efficacy = 1 - 4 * (dosages - 0.5) ** 2

    # Introduce random noise for a small portion of the data
    num_points_with_noise = int(noise_perc * size)
    indices_with_noise = np.random.choice(len(dosages), num_points_with_noise, replace=False)
    efficacy[indices_with_noise] += np.random.uniform(-0.15, 0.15, num_points_with_noise)

    # Ensure efficacy values are between 0 and 1
    efficacy = np.clip(efficacy, 0, 1)

    # Ensure efficacy values are between 0 and 1
    efficacy = np.clip(efficacy, 0, 1)

    return dosages, efficacy


class Linear:
    def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        self.bias = Tensor.zeros(out_features)

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)


class TinyNet:
    def __init__(self):
        self.l1 = Linear(1, 2, bias=True)
        self.l2 = Linear(2, 1, bias=True)

    def __call__(self, x):
        x = self.l1(x)
        x = x.softplus()
        x = self.l2(x)
        return x

    def __repr__(self):
        return f"l1.weights: {self.l1.weight} \n l1.bias = {self.l1.bias} \n l2.weight: {self.l2.weight} \n l2.bias = {self.l2.bias}"


net = TinyNet()


def test():
    net.l1.weight = Tensor([[ 0.30456709],[-0.63777754]])
    net.l1.bias = Tensor([-0.57553545, -0.35180987])
    net.l2.weight = Tensor([[-0.49758171, -0.54923827]])
    net.l2.bias = Tensor([1.23538092])

    opt = SGD([net.l1.weight, net.l1.bias, net.l2.weight, net.l2.bias], lr=3e-4)
    print(f"{net.l1.weight=}")
    print(f"{net.l1.bias=}")


    inp = Tensor([0.49987])

    # res = net(inp)
    forward = net.l1(inp)
    print(f"net.l1(inp) = {forward}")
    forward = forward.softplus()
    print(f"softplus() = {forward}")
    forward = net.l2(forward)
    print(f"net.l2() = {forward}")

    label = Tensor([0.9987])
    print(f"{label=}")
    loss = (label - forward).square().sum()
    print(f"{loss=}")

    print()
    print("loss.backward()")
    backward = loss.backward().realize()
    print(f"{backward=}")




# test()

## train:

def train():
    train_data_size = 30000
    random_noise_perc = 0.1


    opt = SGD([net.l1.weight, net.l1.bias, net.l2.weight, net.l2.bias], lr=3e-4)
    data, label = generate_data(train_data_size, random_noise_perc)
    x_train, y_train, x_test, y_test = data[:25000], label[:25000], data[25000:], label[25000:]

    steps=300
    with Tensor.train():
        for step in range(steps):
            samp = np.random.randint(0, x_train.shape[0], size=(64))
            batch = Tensor([x for x in x_train[samp]], requires_grad=False).reshape((64,1))

            labels = Tensor(y_train[samp], requires_grad=False).reshape((64,1))

            out = net(batch)

            loss = out.binary_crossentropy(labels)
            # loss = (labels - out).square().sum()
            # print(f"{loss=}")

            opt.zero_grad()

            loss.backward()

            opt.step()


            exit(1)
            # print(out == labels)

            if step % 10 == 0:
                # print(f"{out[-10:-4]=}")
                # print(f"{labels[-10:-4]=}")
                # print()
                # print(f"{(out + labels)=}")
                print(f"Step {step+1} | Loss: {loss.numpy()} ")



    print(net)

train()
