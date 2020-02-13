import utilities as ut
from sklearn.model_selection import train_test_split
import shelve


class DataSet:

    def __init__(self, pathDS, status='l'):
        self.__pathDS = pathDS
        self._status = status
        self.__filename = 'Shelvefiledata'

    def PreparingData(self):
        if self._status == "l":
            my_shelve = shelve.open(self.__filename)
            return self.__filename
        elif self._status == "s":
            X_train, labels_train, list_ch_train = ut.read_data(
                data_path=self.__pathDS, split="train")  # train

            X_test, labels_test, list_ch_test = ut.read_data(
                data_path=self.__pathDS, split="test")  # test

            features_train = ut.read_Features(
                data_path=self.__pathDS, split="train")  # features train
            features_test = ut.read_Features(
                data_path=self.__pathDS, split="test")  # features train

            assert list_ch_train == list_ch_test, "Mistmatch in channels!"

            # Normalize?
            X_train, X_test = ut.standardize(X_train, X_test)
            # X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train,
            #                                                 stratify=labels_train, random_state=123)

            # One-hot encoding:
            y_tr = ut.one_hot(labels_train)
            # y_vld = ut.one_hot(lab_vld)
            y_test = ut.one_hot(labels_test)

            my_shelve = shelve.open(self.__filename, 'n')
            my_shelve['data_train'] = X_train
            # my_shelve['data_vld'] = X_vld
            my_shelve['data_test'] = X_test
            my_shelve['labels_train'] = y_tr
            my_shelve['labels_test'] = y_test
            # my_shelve['labels_vld'] = y_vld
            my_shelve['labels_test'] = y_test
            my_shelve['features_train'] = features_train
            my_shelve['features_test'] = features_test
            return self.__filename
