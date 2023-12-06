# mygrad
## from teenygrad to tinygrad

mygrad is my study journey through the deep learning framework tinygrad.
Following the advices from george himself, first we'll figure teenygrad out, than get into the other 4k+ lines of tinygrad.

## How?
I'm literally copying the code from teenygrad while figuring out (almost) line by line what does what, where, when, etc.

## Why?
I'm a regular developer that didn't know anything about deep learning, ml and neural networks. So one day I decided to at least get an overview of what is a model, what does it mean to train it, and how to run one that's already trained. (What does it mean "the model is trained"? I asked myself over and over again). I don't really understand the entire math behind everything but I got a good initial grasp from resources online.


## Running the code
I'm really not a python developer, so I don't even know if this stuff is even up to date/right.

Create a virtual env with python3.8, source it and install dependecies
```sh
python3 -m venv .env;
source .env/bin/activate;
pip install -r requirements.txt
```

```sh
python -c "from mygrad.tensor import Tensor
a = Tensor(2)
print(a)"
```


## What is that tinygrad teenygrad thing you talk about?
[tinygrad](https://github.com/tinygrad/tinygrad)
[teenygrad](https://github.com/tinygrad/teenygrad)


