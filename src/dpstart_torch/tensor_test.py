import torch
import numpy as np
from torch.autograd import Variable


def test_ctr():
    a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
    assert a.type() == 'torch.FloatTensor'
    assert a.size() == torch.Size([3, 2])

    a[0, 1] = 100
    assert torch.equal(a, torch.Tensor([[2, 100], [4, 8], [7, 9]]))


def test_long():
    b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
    assert b.type() == 'torch.LongTensor'


def test_zeros():
    c = torch.zeros((3, 2))
    assert c.type() == 'torch.FloatTensor'
    assert torch.equal(c, torch.Tensor([[0., 0.],
                                        [0., 0.],
                                        [0., 0.]]))


def test_variable():
    x = Variable(torch.Tensor([1]), requires_grad=True)
    w = Variable(torch.Tensor([2]), requires_grad=True)
    b = Variable(torch.Tensor([3]), requires_grad=True)

    y = w * x + b

    # 计算梯度
    y.backward()  # same as y.backward(torch.FloatTensor([1]))
    assert torch.equal(b.grad, torch.Tensor([1]))
    assert torch.equal(w.grad, torch.Tensor([1]))
    assert torch.equal(x.grad, torch.Tensor([2]))


def test_gpu():
    print(torch.cuda.is_available())
