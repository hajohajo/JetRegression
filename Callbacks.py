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

        plt.plot(self.x[1:], self.losses[1:], label = 'Training loss')
        plt.plot(self.x[1:], self.val_losses[1:], label = 'Validation loss')
        plt.legend()
        plt.yscale('log')
        plt.title('Training and validation losses')
        plt.ylabel('Epoch loss')
        plt.xlabel('Epoch')

	plt.savefig('plots/Loss_plot.pdf')
	plt.clf()

def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=15, min_lr=1e-7)
    checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss',
                                 verbose=0, save_best_only=False,
                                 mode='auto')
    loss_plot = plot_learning()

    monitoring = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True, write_grads=True) 

#    return [reduce_lr, checkpoint, loss_plot, monitoring]
    return [reduce_lr, checkpoint, loss_plot]

