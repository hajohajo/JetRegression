from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Input,Dense,Conv1D,MaxPooling1D,Convolution1D,Flatten,Dropout,Activation,LSTM,CuDNNLSTM,Concatenate, Add
from keras.layers.advanced_activations import LeakyReLU


def create_model(model_name, Ccands_shape, Ncands_shape, Pcands_shape, Globals_shape):
    if model_name is 'DeepJet':
        chg_inp = Input(shape=(Ccands_shape[1], Ccands_shape[2]), name='Charged_input')
        chg = Conv1D(64, 1, kernel_initializer='lecun_uniform', activation='relu')(chg_inp)
        chg = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = Conv1D(2, 1, kernel_initializer='lecun_uniform', activation='relu')(chg)
        chg = LSTM(15, activation='relu', kernel_initializer='glorot_normal', go_backwards=True)(chg)

        neu_inp = Input(shape=(Ncands_shape[1], Ncands_shape[2]), name='Neutral_input')
        neu = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(neu_inp)
        neu = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = Conv1D(2, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = LSTM(15, activation='relu', kernel_initializer='glorot_normal', go_backwards=True)(neu)

        pho_inp = Input(shape=(Pcands_shape[1], Pcands_shape[2]), name='Photon_input')
        pho = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(pho_inp)
        pho = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = Conv1D(2, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = LSTM(15, activation='relu', kernel_initializer='glorot_normal', go_backwards=True)(pho)

        glo_inp = Input(shape=(Globals_shape[1],))

        concat = Concatenate()([chg, neu, pho, glo_inp])

        dense = Dense(32, activation='relu')(concat)
        dense = Dense(16, activation='relu')(dense)
        dense = Dense(4, activation='relu')(dense)
        output = Dense(1, activation='relu')(dense)

        model = Model(inputs=[chg_inp, neu_inp, pho_inp, glo_inp], outputs=[output])
        model.compile(loss='logcosh', optimizer=optimizers.Adam())

        return model

    if model_name is 'ResNet':
        chg_inp = Input(shape=(Ccands_shape[1], Ccands_shape[2]), name='Charged_input')
        chg1 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(chg_inp)
        chg2 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(chg1)
	add_chg1 = Add()([chg2, chg1])
        chg3 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(add_chg1)
        chg4 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(chg3)
        add_chg2 = Add()([chg4, add_chg1])
        chg5 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(add_chg2)
        chg6 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(chg5)
	add_chg3 = Add()([chg6, add_chg2])
#        chg = CuDNNLSTM(15,go_backwards=True, kernel_initializer='glorot_normal')(add_chg3) #go_backwards = representation of the lowest pT pfCand is fed in first
        chg = LSTM(15, activation='relu', kernel_initializer='glorot_normal', go_backwards=True)(add_chg3)

        neu_inp = Input(shape=(Ncands_shape[1], Ncands_shape[2]), name='Neutral_input')
        neu1 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(neu_inp)
        neu2 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(neu1)
        add_neu1 = Add()([neu2, neu1])
        neu3 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(add_neu1)
        neu4 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(neu3)
        add_neu2 = Add()([neu4, add_neu1])
        neu5 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(add_neu2)
        neu6 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(neu5)
	add_neu3 = Add()([neu6, add_neu2])
#        neu = CuDNNLSTM(15,go_backwards=True, kernel_initializer='glorot_normal')(add_neu3) #go_backwards = representation of the lowest pT pfCand is fed in first
	neu = LSTM(15, activation='relu', kernel_initializer='glorot_normal', go_backwards=True)(add_neu3)

        pho_inp = Input(shape=(Pcands_shape[1], Pcands_shape[2]), name='Photon_input')
        pho1 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(pho_inp)
        pho2 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(pho1)
        add_pho1 = Add()([pho2, pho1])
        pho3 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(add_pho1)
        pho4 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(pho3)
        add_pho2 = Add()([pho4, add_pho1])
        pho5 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(add_pho2)
        pho6 = Conv1D(64, 1, kernel_initializer='glorot_normal', activation='relu')(pho5)
	add_pho3 = Add()([pho6, add_pho2])
#        pho = CuDNNLSTM(15,go_backwards=True, kernel_initializer='glorot_normal')(add_pho3) #go_backwards = representation of the lowest pT pfCand is fed in first
        pho = LSTM(15, activation='relu', kernel_initializer='glorot_normal', go_backwards=True)(add_pho3)

        glo_inp = Input(shape=(Globals_shape[1],))

        concat = Concatenate()([chg, neu, pho, glo_inp])

        dense = Dense(32, activation='relu', kernel_initializer='glorot_normal')(concat)
        dense = Dense(16, activation='relu', kernel_initializer='glorot_normal')(dense)
        output = Dense(1, activation='relu', kernel_initializer='glorot_normal')(dense)

        model = Model(inputs=[chg_inp, neu_inp, pho_inp, glo_inp], outputs=[output])
        model.compile(loss='mae', optimizer=optimizers.Adam(lr=1e-3))

        return model

    if model_name is 'Test':
	glo_inp = Input(shape=(Globals_shape[1],))
	dense = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(glo_inp)
        dense = Dense(16, activation='relu', kernel_initializer='glorot_uniform')(dense)
        dense = Dense(8, activation='relu', kernel_initializer='glorot_uniform')(dense)
        output = Dense(1, activation='relu', kernel_initializer='glorot_uniform')(dense)

        model = Model(inputs=glo_inp, outputs=[output])
        model.compile(loss='mae', optimizer=optimizers.Adam(lr=1e-4))

        return model


    else:
	print model_name
        raise ValueError('Model not found')

