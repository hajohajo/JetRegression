import matplotlib.pyplot as plt
from keras.callbacks import Callback, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

#Callback that produces training loss plots at the end of each epoch
class plot_learning(Callback):
    def on_train_begin(self, logs={}):
        self.epoch = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.logs = []
        self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs = {}):
        self.logs.append(logs)
        self.x.append(self.epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.epoch += 1

        plt.plot(self.x, self.losses, label = 'Training loss')
        plt.plot(self.x, self.val_losses, label = 'Validation loss')
        plt.legend()
        plt.yscale('log')
        plt.title('Training and validation losses')
        plt.ylabel('Epoch loss')
        plt.xlabel('Epoch')

	plt.savefig('Loss_plot.pdf')
	plt.clf()

def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=15, min_lr=1e-7)
    checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss',
                                 verbose=0, save_best_only=False,
                                 mode='auto')
    loss_plot=plot_learning()

    return [reduce_lr, checkpoint, loss_plot]
