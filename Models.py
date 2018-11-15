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
        chg = CuDNNLSTM(25,go_backwards=True)(chg) #go_backwards = representation of the lowest pT pfCand is fed in first

        neu_inp = Input(shape=(Ncands_shape[1], Ncands_shape[2]), name='Neutral_input')
        neu = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(neu_inp)
        neu = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = Conv1D(2, 1, kernel_initializer='lecun_uniform', activation='relu')(neu)
        neu = CuDNNLSTM(25,go_backwards=True)(neu) #go_backwards = representation of the lowest pT pfCand is fed in first

        pho_inp = Input(shape=(Pcands_shape[1], Pcands_shape[2]), name='Photon_input')
        pho = Conv1D(32, 1, kernel_initializer='lecun_uniform', activation='relu')(pho_inp)
        pho = Conv1D(16, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = Conv1D(8, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = Conv1D(2, 1, kernel_initializer='lecun_uniform', activation='relu')(pho)
        pho = CuDNNLSTM(25,go_backwards=True)(pho) #go_backwards = representation of the lowest pT pfCand is fed in first

        glo_inp = Input(shape=(Globals_shape[1],))

        concat = Concatenate()([chg, neu, pho, glo_inp])

        dense = Dense(32, activation='relu')(concat)
        dense = Dense(16, activation='relu')(dense)
        output = Dense(1, activation='relu')(dense)

        model = Model(inputs=[chg_inp, neu_inp, pho_inp, glo_inp], outputs=[output])
        model.compile(loss='logcosh', optimizer=optimizers.Adam())

        return model

    if model_name is 'ResNet':
        chg_inp = Input(shape=(Ccands_shape[1], Ccands_shape[2]), name='Charged_input')
        chg1 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(chg_inp)
        chg2 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(chg1)
	add_chg1 = Add()([chg1, chg2])
        chg3 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(add_chg1)
        chg4 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(chg3)
        add_chg2 = Add()([chg3, chg4])
        chg5 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(add_chg2)
        chg6 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(chg5)
        chg = CuDNNLSTM(15,go_backwards=True)(chg6) #go_backwards = representation of the lowest pT pfCand is fed in first
#	chg = Flatten()(chg6)

        neu_inp = Input(shape=(Ncands_shape[1], Ncands_shape[2]), name='Neutral_input')
        neu1 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(neu_inp)
        neu2 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(neu1)
        add_neu1 = Add()([neu1, neu2])
        neu3 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(add_neu1)
        neu4 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(neu3)
        add_neu2 = Add()([neu3, neu4])
        neu5 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(add_neu2)
        neu6 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(neu5)
        neu = CuDNNLSTM(15,go_backwards=True)(neu6) #go_backwards = representation of the lowest pT pfCand is fed in first
#	neu = Flatten()(neu6)

        pho_inp = Input(shape=(Pcands_shape[1], Pcands_shape[2]), name='Photon_input')
        pho1 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(pho_inp)
        pho2 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(pho1)
        add_pho1 = Add()([pho1, pho2])
        pho3 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(add_pho1)
        pho4 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(pho3)
        add_pho2 = Add()([pho3, pho4])
        pho5 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(add_pho2)
        pho6 = Conv1D(4, 1, kernel_initializer='lecun_uniform', activation='relu')(pho5)
        pho = CuDNNLSTM(15,go_backwards=True)(pho6) #go_backwards = representation of the lowest pT pfCand is fed in first
#	pho = Flatten()(pho6)

        glo_inp = Input(shape=(Globals_shape[1],))

        concat = Concatenate()([chg, neu, pho, glo_inp])

        dense = Dense(32, activation='selu', kernel_initializer='lecun_uniform')(concat)
        dense = Dense(16, activation='selu', kernel_initializer='lecun_uniform')(dense)

	# The rationale behind using linear instead of relu (which makes sense since correction factor shouldnt be negative)
	# is to be able to propagate also the negative predicton errors to the network during training
        output = Dense(1, activation='linear', kernel_initializer='lecun_uniform')(dense)

        model = Model(inputs=[chg_inp, neu_inp, pho_inp, glo_inp], outputs=[output])
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4))

        return model

    if model_name is 'Test':
	glo_inp = Input(shape=(Globals_shape[1],))
	dense = Dense(32, activation='selu')(glo_inp)
        dense = Dense(16, activation='selu')(dense)
        dense = Dense(8, activation='selu')(dense)
        output = Dense(1, activation='selu')(dense)

        model = Model(inputs=glo_inp, outputs=[output])
        model.compile(loss='logcosh', optimizer=optimizers.Adam(lr=1e-3))

        return model


    else:
	print model_name
        raise ValueError('Model not found')

