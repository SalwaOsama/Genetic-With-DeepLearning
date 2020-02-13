from DataSet import DataSet
from RNN_UCI import RNN as rnnuci
from RNN_WISM import RNN as rnnwism
import WISDMDataSet as wisdm

def main():
    DS = 'wisdm'
    if(DS == 'uci'):
        MainUCI()
    else:
        MainWISDM()


def MainWISDM():
    segment_size = 200
    # n_filters = 196
    n_channels = 3
    # epochs = 100000
    # batch_size = 200
    # learning_rate = 5e-4
    # dropout_rate = 0.15
    # eval_iter = 1000
    # filters_size = 12
    # n_classes = 6
    # n_hidden=1024

    n_hidden = 32
    n_classes = 6
    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    training_iters = 300
    batch_size = 1500
    display_iter = 30000

    ds = wisdm.DataSet('l')
    shelveDataFile = ds.PreparingData()
    rnn = rnnwism(shelveDataFile, n_hidden=n_hidden, n_classes=n_classes,
                  learning_rate=learning_rate, lambda_loss_amount=lambda_loss_amount,
                  training_iters=training_iters, batch_size=batch_size,
                  display_iter=display_iter, segment_size=segment_size, n_channels=n_channels)
    rnn.TrainingRNNandAccuracy()


def MainUCI():
    n_hidden = 32
    n_classes = 6
    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    training_iters = 300
    batch_size = 1500
    display_iter = 30000
    ds = DataSet("datasets/ShelveUCIData", 'l')
    shelveDataFile = ds.PreparingData()
    rnn = rnnuci(shelveDataFile, n_hidden=n_hidden, n_classes=n_classes,
                 learning_rate=learning_rate, lambda_loss_amount=lambda_loss_amount,
                 training_iters=training_iters, batch_size=batch_size,
                 display_iter=display_iter)
    rnn.TrainingRNNandAccuracy()
if __name__ == "__main__":
    main()