from sacred import Experiment
ex = Experiment('my_experiment',interactive=True)
@ex.config
def my_config1():
    a = 10
    b = 'test'

@ex.capture
def print_a_and_b(a, b):
    print("a =", a)
    print("b =", b)

@ex.main
def my_main():
    print_a_and_b(1,'ll')


import visdom
import numpy as np
vis = visdom.Visdom()
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))