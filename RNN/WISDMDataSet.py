from StructData import StructData
from Experiment import Experiment
import Utility as ut
import shelve
import numpy as np
import pandas as pd
import os


class DataSet:
    _ActivitiesNum = 6

    def __init__(self, status='l'):
        self._pathDS = "datasets/wisdm_raw_data/wisdm_raw_datashelv.out"
        self._status = status
        filename_experiments = 'ShelveExperiments/WiDMShelveFiles/ShelveExperiments.out'
        if(status == 'l'):
            try:
                fileExperiments = shelve.open(filename_experiments)
                self._experiments = fileExperiments['_experiments']
            except:
                print("cannot open shelve experiments WISDOME")
        elif (status == 's'):
            self.ExtractWISDOMEDS(filename_experiments)

    def ExtractWISDOMEDS(self, ExperimentsShelveFile):
        # my_shelve = shelve.open(ExperimentsShelveFile, 'n')
        self._dictActivity = {"Jogging": "0", "Walking": "1",
                              "Upstairs": "2", "Downstairs": "3",
                              "Sitting": "4", "Standing": "5"}

        FileDataSet = shelve.open(self._pathDS)
        user_id = FileDataSet['usr_id']
        arr_activities = FileDataSet['arr_activities']
        timestamp = FileDataSet['timestamp']
        arr_x = FileDataSet['arr_x']
        arr_y = FileDataSet['arr_y']
        arr_z = FileDataSet['arr_z']
        FileDataSet.close()

        user_id = user_id.T[np.newaxis][0]
        arr_activities = arr_activities.T[np.newaxis][0]
        arr_x = arr_x.T[np.newaxis][0]
        arr_y = arr_y.T[np.newaxis][0]
        arr_z = arr_z.T[np.newaxis][0]
        df = pd.DataFrame(
            {'usr_id': user_id, 'arr_activities': arr_activities,
             'arr_x': arr_x, 'arr_y': arr_y, 'arr_z': arr_z})
        gusrs = df.groupby(['usr_id', 'arr_activities']).apply(
            lambda x: x.values.tolist())
        self._experiments = []
        for i in range(gusrs.shape[0]):
            ActivityByUser = gusrs.values[i]
            # extract Data for saveing in experiment
            UserID = ActivityByUser[0][4]
            ActivityID = self._dictActivity[ActivityByUser[0][0]]
            collectX = [ActivityByUser[i][1]
                        for i in range(len(ActivityByUser))]
            collectY = [ActivityByUser[i][2]
                        for i in range(len(ActivityByUser))]
            collectZ = [ActivityByUser[i][3]
                        for i in range(len(ActivityByUser))]

            experiment = Experiment(
                nusr=UserID, ActivityID=ActivityID, X=collectX, Y=collectY, Z=collectZ)
            self._experiments.append(experiment)
        my_shelve = shelve.open(ExperimentsShelveFile)
        my_shelve['_experiments'] = self._experiments
        print("All experiments UserId, Activity, All_X, All_Y, All_Z is saved in shelveFile with Name" +
              str(ExperimentsShelveFile))

    def SplitTrainTest(self, test_user_ids, _experiments):
        _trainingExperiments = []
        _testExperiments = []
        _test_user_ids = test_user_ids
        for expr in _experiments:
            # check if userId is in test or not
            if not int(expr._nusr) in test_user_ids:
                _trainingExperiments.append(expr)
            else:
                _testExperiments.append(expr)
        return _trainingExperiments, _testExperiments

    def SplitExprIntoSegmentSize(self, exper, segment_size):
        paramX = []
        paramY = []
        paramZ = []
        features = []
        answers_raw = []
        k = 0
        while k + segment_size <= len(exper._X) - 1:
            x_add = exper._X[k: k + segment_size: 1]
            y_add = exper._Y[k: k + segment_size: 1]
            z_add = exper._Z[k: k + segment_size: 1]

            paramX = np.hstack((paramX, x_add))
            paramY = np.hstack((paramY, y_add))
            paramZ = np.hstack((paramZ, z_add))

            ff = ut.Extract_Features(x_add, y_add, z_add)
            features.append(ff)

            ans = np.zeros((1, self._ActivitiesNum))[0]
            ans[int(exper._ActivityID)] = 1
            answers_raw = np.hstack((answers_raw, ans))
            k = k + segment_size//2
        return StructData(x=paramX, y=paramY, z=paramZ, features=features,
                          answers=answers_raw)

    def SplitAllExpermintsAsSegmentSize(self, segment_size,
                                        _trainingExperiments, _testExperiments,
                                        filename):
        # SplitTrainigTest
        # SplitTrainTest(test_user_ids)
        # SplitExpintoSegment
        # training data
        x = []
        #gyro_x = []
        y = []
        #gyro_y = []
        z = []
        #gyro_z = []
        features = []
        answers_raw = []

        # testing data

        test_x = []
        #test_gyro_x = []
        test_y = []
        #test_gyro_y = []
        test_z = []
        #test_gyro_z = []
        test_features = []
        test_answers_raw = []
        i = 0
        for expr in _trainingExperiments:
            trainingexprdata = self.SplitExprIntoSegmentSize(
                expr, segment_size)
            x = np.hstack((x, trainingexprdata.paramX))
            y = np.hstack((y, trainingexprdata.paramY))
            z = np.hstack((z, trainingexprdata.paramZ))
            features += trainingexprdata.paramFeatures

            # gyro_x=np.hstack((gyro_x, trainingexprdata.paramGX))
            # gyro_y=np.hstack((gyro_y, trainingexprdata.paramGY))
            # gyro_z=np.hstack((gyro_z, trainingexprdata.paramGZ))
            answers_raw = np.hstack(
                (answers_raw, trainingexprdata.ParamAnswers_raw))
        trainingData = StructData(
            x=x, y=y, z=z, features=features, answers=answers_raw)
        for expr in _testExperiments:
            testingexprdata = self.SplitExprIntoSegmentSize(expr, segment_size)
            test_x = np.hstack((test_x, testingexprdata.paramX))
            test_y = np.hstack((test_y, testingexprdata.paramY))
            test_z = np.hstack((test_z, testingexprdata.paramZ))
            test_features += testingexprdata.paramFeatures

            test_answers_raw = np.hstack(
                (test_answers_raw, testingexprdata.ParamAnswers_raw))

        testingData = StructData(x=test_x, y=test_y, z=test_z,
                                 features=test_features,
                                 answers=test_answers_raw)
        # answer_vector=[]
        # for i in range(len(trainingData.ParamAnswers_raw)):
        #     vect=np.zeros((1, 6))[0]
        #     vect[np.int(trainingData.ParamAnswers_raw[i])-1]=1
        #     answer_vector=np.hstack((answer_vector, vect))
        # test_answ_vector=[]
        # for i in range(len(testingData.ParamAnswers_raw)):
        #     vect=np.zeros((1, 6))[0]
        #     vect[np.int((testingData.ParamAnswers_raw[i]))-1]=1
        #     test_answ_vector=np.hstack((test_answ_vector, vect))
        all_data = np.vstack((trainingData.paramX, trainingData.paramY,
                              trainingData.paramZ))
        all_data = all_data.T[np.newaxis][0]
        all_test_data = np.vstack((testingData.paramX, testingData.paramY,
                                   testingData.paramZ))
        all_test_data = all_test_data.T[np.newaxis][0]

        # Normlize Features
        trainingData.paramFeatures = trainingData.paramFeatures - \
            np.mean(trainingData.paramFeatures, axis=0)
        trainingData.paramFeatures = trainingData.paramFeatures / \
            np.std(trainingData.paramFeatures, axis=0)
        testingData.paramFeatures = testingData.paramFeatures - \
            np.mean(testingData.paramFeatures, axis=0)
        testingData.paramFeatures = testingData.paramFeatures / \
            np.std(testingData.paramFeatures, axis=0)

        my_shelve = shelve.open(filename, 'n')
        my_shelve['data_train'] = all_data
        my_shelve['features'] = trainingData.paramFeatures
        my_shelve['data_test'] = all_test_data
        my_shelve['features_test'] = testingData.paramFeatures
        my_shelve['labels_train'] = answers_raw
        my_shelve['labels_test'] = test_answers_raw
        my_shelve['segment_size'] = segment_size

    def PreparingData(self, segment_size=200, test_user_ids=range(27, 36)):
        shelvePreparedData = "ShelvePreparedWISDMDataSegSiz/" + \
            str(segment_size)
        if not os.path.exists(shelvePreparedData+".bak"):
            _trainingExperiments, _testExperiments = self.SplitTrainTest(
                test_user_ids, self._experiments)
            self.SplitAllExpermintsAsSegmentSize(
                segment_size, _trainingExperiments, _testExperiments,
                shelvePreparedData)
        
        return shelve.open(shelvePreparedData)
