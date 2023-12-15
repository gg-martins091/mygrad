
from teenygrad.tensor  import Tensor
from matplotlib import pyplot as plt
import numpy as np


initial_weights = Tensor([0.3], requires_grad=True) # intercept
initial_slope = Tensor([0], requires_grad=True)
batch = Tensor([0.5, 2.3, 2.9]) # x (weight on the video)
labels = Tensor([1.4, 1.9, 3.2])

def equation(x, w, b):
    y = w + b * x # y = height on the video
    return y

def lossfunc_forward(out):
    return (labels-out).square().sum() # sum of square residuals


lr = 0.01
weights_used = []
losses = []

for i in range(200):
    out = equation(batch, initial_weights, initial_slope)
    loss = lossfunc_forward(out)
    initial_weights.grad = None
    weights_used.append(initial_weights.numpy())


    loss.backward()
    losses.append(loss.numpy())

    weight_step_size = initial_weights.grad * lr
    bias_step_size = initial_slope.grad * lr

    if weight_step_size < 0.0001 or bias_step_size < 0.0001:
        # raise ValueError('step_size < 0.0001')
        pass

    initial_weights.assign(initial_weights.detach() - weight_step_size)
    initial_slope.assign(initial_slope.detach() - bias_step_size)
    print(f"{initial_weights=}")
    print(f"{initial_slope=}")
    print(f"{loss=}")
    print()
    print()


    print(f"{initial_weights=}")
    print(f"{initial_slope=}")
    # y = equation(batch, initial_weights, initial_slope)
    # print(f"{y.numpy()=}")

# plt.plot(batch.numpy(), labels.numpy(), 'ro', batch.numpy(), y.numpy(), 'g--')
# plt.show()


# print(f"{losses_gradients}")
# plt.plot(weights_used, losses_gradients, 'ro')
# plt.show()

# 2 - now we create an equation for the curve of ssr:

# ssr = (labels     -               redicted             ).square().sum()
# ssr = (labels     -             equation(...)           ).square().sum()
# ssr = (labels     -  (weight + initial_slope * batch)  ).square().sum()


# so for each step, we can plug in the weight and the slope and take the ssr 
# sr1 = (1.4     -  (weight + initial_slope * 0.5)  ).square()
# sr2 = (1.9     -  (weight + initial_slope * 2.3)  ).square()
# sr3 = (3.2     -  (weight + initial_slope * 2.9)  ).square()
# ssr = sum (sr1, sr2, sr3)


# now we can just take the derivative of this function
# and determine the slope at any value for weight (intercept)


# print(f"{ssr(batch,initial_weights)=}")




