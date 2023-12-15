
# from teenygrad.tensor import Tensor
# from tinygrad.helpers import fetch
# from tinygrad.helpers import dtypes
# from teenygrad.nn.optim import SGD

from mygrad.tensor import Tensor
from mygrad.helpers import fetch, dtypes, Timing
from mygrad.nn.optim import SGD

import gzip
import numpy as np

np.set_printoptions(suppress=True)

def fetch_mnist(tensors=False):
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"   # http://yann.lecun.com/exdb/mnist/ lacks https
  X_train = parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:]
  X_test = parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
  Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:]
  if tensors: return Tensor(X_train).reshape(-1, 1, 28, 28), Tensor(Y_train), Tensor(X_test).reshape(-1, 1, 28, 28), Tensor(Y_test)
  else: return X_train, Y_train, X_test, Y_test
# a = Tensor([1.5, 2])
# b = Tensor([2.3, 1])

class Linear:
    def __init__(self, in_features, out_features, bias=False, initialization: str='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        print(f"{self.weight=}")
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)



class TinyNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x


net = TinyNet()

opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)

print("##### FETCH MNIST")
X_train, Y_train, X_test, Y_test = fetch_mnist()

steps = 250
with Tensor.train():
    for step in range(steps):
        samp = np.random.randint(0, X_train.shape[0], size=(64))
        # print(f"{samp=}")

        batch = Tensor(X_train[samp], requires_grad=False)
        # print(f"{batch=}")

        labels = Tensor(Y_train[samp])
        # print(f"{labels=}")

        out = net(batch)

        loss = out.sparse_categorical_crossentropy(labels)

        print(f"{loss=}")
        opt.zero_grad()
        print(f"{opt=}")

        loss.backward()

        opt.step()
        exit(1)


        # print(f"{pred=}")
        # print(f"{labels=}")

        pred = out.argmax(axis=-1)
        acc = (pred == labels).mean()

        if step % 1 == 0:
            print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")



with Timing("Time: "):
  avg_acc = 0
  for step in range(steps):
    # random sample a batch
    samp = np.random.randint(0, X_test.shape[0], size=(64))
    batch = Tensor(X_test[samp], requires_grad=False)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc / steps}")
