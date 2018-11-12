from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Input,Dense,Conv1D,MaxPooling1D,Convolution1D,Flatten,Dropout,Activation,LSTM,CuDNNLSTM,Concatenate
from keras.layers.advanced_activations import LeakyReLU


def create_model(model_name, Ccands_shape, Ncands_shape, Pcands_shape, Globals_shape):
    if model_name is 'DeepJet':
        chg_inp = Input(shape=(Ccands_shape[1], Ccands_shape[2]), name='Charged_input')
        chg = Conv1D(64, 1, kernel_initializer='lecun_uniform', activation='relu')(chg_inp)
        chg = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = CuDNNLSTM(25,go_backwards=True)(chg) #go_backwards = representation of the lowest pT pfCand is fed in first

        neu_inp = Input(shape=(Ncands_shape[1], Ncands_shape[2]), name='Neutral_input')
        neu = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(neu_inp)
        neu = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = CuDNNLSTM(25,go_backwards=True)(neu) #go_backwards = representation of the lowest pT pfCand is fed in first

        pho_inp = Input(shape=(Pcands_shape[1], Pcands_shape[2]), name='Photon_input')
        pho = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(pho_inp)
        pho = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = CuDNNLSTM(25,go_backwards=True)(pho) #go_backwards = representation of the lowest pT pfCand is fed in first

        glo_inp = Input(shape=(Globals_shape[1],))

        concat = Concatenate()([chg, neu, pho, glo_inp])

        dense = Dense(128, activation='relu')(concat)
        dense = Dense(64, activation='relu')(dense)
        dense = Dense(32, activation='relu')(dense)
        dense = Dense(16, activation='relu')(dense)
        output = Dense(1, activation='relu')(dense)

        model = Model(inputs=[chg_inp, neu_inp, pho_inp, glo_inp], outputs=[output])
        model.compile(loss='logcosh', optimizer=optimizers.Adam())

        return model

    else:
        raise ValueError('Model not found')
