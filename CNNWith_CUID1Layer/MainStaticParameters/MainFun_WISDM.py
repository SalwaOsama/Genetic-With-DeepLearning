import WISDMDataSet as wisdm
from CNN import CNN
from AlternativeCNN import CNN
import CNN_WH_Tes_Tra_Feat
import CNN_WH_Tes_Tra_withoutFeat


def main():
    segment_size = 200
    n_filters = 196
    n_channels = 3
    epochs = 100000
    batch_size = 200
    learning_rate = 5e-4
    dropout_rate = 0.15
    eval_iter = 1000
    filters_size = 12   
    n_classes = 6
    n_hidden=1024


    ds = wisdm.DataSet('l')
    shelveDataFile = ds.PreparingData(segment_size=segment_size)
    cnn = CNN_WH_Tes_Tra_withoutFeat.CNN(shelveDataFile, segment_size=segment_size, n_filters=n_filters,
            n_channels=n_channels, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            dropout_rate=dropout_rate, eval_iter=eval_iter, filters_size=filters_size, n_classes=n_classes,n_hidden=n_hidden)
    # # WISDM dataset with CNN_WH_Tes_Tra_WithoutFeat.py
    filetrainingChekpoint="TrainingCNNCheckPoint/WISDM/checkpoints_WH_Tes_Tra_WithoutFeat-cnn" 

    cnn.Training(filetrainingChekpoint)
    cnn.Testing(filetrainingChekpoint)
