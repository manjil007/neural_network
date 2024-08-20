import unittest

import numpy as np

from models._base_network import _baseNetwork


from optimizers._base_optimizer import _BaseOptimizer
from utils import load_mnist_trainval, load_mnist_test, generate_batched_data
from utils import train, evaluate


class TestTraining(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def test_regularization(self):
        optimizer = _BaseOptimizer()
        model = _baseNetwork()

        w_grad = model.gradients['W1'].copy()
        optimizer.apply_regularization(model)
        w_grad_reg = model.gradients['W1']
        reg_diff = w_grad_reg - w_grad
        expected_diff = model.weights['W1'] * optimizer.reg

        diff = np.mean(np.abs(reg_diff - expected_diff))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_one_layer_train(self):
        optimizer = _BaseOptimizer()
        model = _baseNetwork()
        train_data, train_label, _, _ = load_mnist_trainval()
        test_data, test_label = load_mnist_test()

        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=128, shuffle=True)
        _, train_acc = train(1, batched_train_data, batched_train_label, model, optimizer, debug=False)

        batched_test_data, batched_test_label = generate_batched_data(test_data, test_label, batch_size=128)
        _, test_acc = evaluate(batched_test_data, batched_test_label, model, debug=False)
        self.assertGreater(train_acc, 0.3)
        self.assertGreater(test_acc, 0.3)


