import pytest
import tensorflow.compat.v1 as tf1

@pytest.fixture(scope='module', autouse=True)
def disable_eager_execution():

    tf1.disable_v2_behavior()


def test_constant():
    message = tf1.constant("Welcome to the exciting world of Deep Neural Networks!")
    with tf1.Session() as sess:
        print(sess.run(message))


def test_graph():
    v_1 = tf1.constant([1, 2, 3, 4])
    v_2 = tf1.constant([2, 1, 5, 3])
    v_add = tf1.add(v_1, v_2)
    with tf1.Session() as sess:
        print(sess.run(v_add))
