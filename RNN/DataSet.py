from Experiment import Experiment
import shelve
import numpy as np
from StructData import StructData
import os


class DataSet:
    TRAIN = "train/"
    TEST = "test/"
    DATASET_PATH = "datasets/UCI/"
    INPUT_SIGNAL_TYPES = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
                          "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
                          "total_acc_x_", "total_acc_y_", "total_acc_z_"]

    LABELS = ["WALKING", "WALKING_UPSTAIRS",
              "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

    def __init__(self, filename, status):
        self.status = status
        self.filename = filename
        if (status == 'l'):
            self.my_shelve = shelve.open(filename)

    def PreparingData(self):
        if(self.status == 'l'):
            return self.my_shelve
        elif (self.status == 's'):
            X_train_signals_paths = [
                self.DATASET_PATH + self.TRAIN + "Inertial Signals/" + signal + "train.txt"
                for signal in self.INPUT_SIGNAL_TYPES]
            X_test_signals_paths = [
                self.DATASET_PATH + self.TEST + "Inertial Signals/" + signal + "test.txt"
                for signal in self.INPUT_SIGNAL_TYPES
            ]

            X_train = self.load_X(X_train_signals_paths)
            X_test = self.load_X(X_test_signals_paths)

            y_train_path = self.DATASET_PATH + self.TRAIN + "y_train.txt"
            y_test_path = self.DATASET_PATH + self.TEST + "y_test.txt"

            y_train = self.load_y(y_train_path)
            y_test = self.load_y(y_test_path)

            my_shelve = shelve.open(self.filename, 'n')
            my_shelve['X_train'] = X_train
            my_shelve['X_test'] = X_test
            my_shelve['y_train'] = y_train
            my_shelve['y_test'] = y_test
            self.my_shelve = my_shelve
            return my_shelve

    def load_X(self, X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_y(self, y_path):
        file = open(y_path, 'r')
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()

        return y_ - 1  # based index0
