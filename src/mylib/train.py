#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Module description
'''

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class SyntheticBernuliDataset(object):
    r'''Base class for synthetic dataset.'''

    def __init__(self, n: int = 10, m: int = 100, seed: int = 42):
        """
        Args:
            n (int): feature number. Defaults to 10.
            m (int): object number. Defaults to 100.
            seed (int): random state seed. Defaults to 42.
        """
        rs = np.random.RandomState(seed)

        # Генерим вектор параметров из нормального распределения
        self.w = rs.randn(n)
        # Генерим вектора признаков из нормального распределения
        self.X = rs.randn(m, n)

        # Гипотеза порождения данных - целевая переменная из схемы Бернули
        self.y = rs.binomial(1, expit(self.X@self.w))


class Trainer(object):
    r'''Base class for all trainers.'''

    def __init__(self, model, X: np.ndarray, Y:  np.ndarray, seed: int = 42):
        """
        Args:
            model: The class with fit and predict methods.
            X (np.ndarray):  The array of shape [num_elemennts, num_feature]
            Y (np.ndarray): [num_elements, num_answers]
            seed (int): random state seed. Defaults to 42.
        """

        self.model = model
        self.seed = seed
        (
            self.X_train,
            self.X_val,
            self.Y_train,
            self.Y_val
        ) = train_test_split(X, Y, random_state=self.seed)

    def train(self):
        """
        Train model
        """
        self.model.fit(self.X_train, self.Y_train)

    def eval(self, output_dict: bool = False) -> str | dict:
        """
        Evaluate model for initial validadtion dataset.

        Args:
            output_dict (bool): If True, return output as dict.

        Returns:
            (str|dict): classification report
        """
        return classification_report(
            self.Y_val,
            self.model.predict(
                self.X_val), output_dict=output_dict)

    def test(self, X: np.ndarray, Y: np.ndarray, output_dict: bool = False) -> str | dict:
        """
        Evaluate model for given dataset.


        Args:
            X (np.ndarray): The array of shape [num_elements, num_feature]
            Y (np.ndarray): The array of shape [num_elements, num_answers]
            output_dict (bool, optional): If True, return output as dict. Defaults to False.

        Returns:
            (str|dict): classification report
        """
        return classification_report(
            Y, self.model.predict(X), output_dict=output_dict)


def cv_parameters(X: np.ndarray, Y: np.ndarray, seed: int = 42, minimal: float = 0.1,
                  maximum: float = 25, count: int = 100) -> tuple[np.ndarray, list[float], list]:
    """
    Function for the experiment with different regularisation parameters ("Cs")
        and return accuracy and params for LogisticRegression for each parameter.    

    Args:
        X (np.ndarray):  The array of shape [num_elements, num_feature]
        Y (np.ndarray):  The array of shape [num_elements, num_answers]
        seed (int, optional): Seed for random state. Defaults to 42.
        minimal (float, optional): Minimum value for the Cs linspace. Defaults to 0.1.
        maximum (float, optional): Maximum value for the Cs linspace. Defaults to 25.
        count (int, optional): Number of the Cs points. Defaults to 100.

    Returns:
        (np.ndarray): Cs
        (list[float]): list of accuracies
        (list): list of params
    """
    Cs = np.linspace(minimal, maximum, count)
    parameters = []
    accuracy = []
    for C in Cs:
        trainer = Trainer(
            LogisticRegression(penalty='l1', solver='saga', C=1/C),
            X, Y,
        )

        trainer.train()

        accuracy.append(trainer.eval(output_dict=True)['accuracy'])

        parameters.extend(trainer.model.coef_)

    return Cs, accuracy, parameters
