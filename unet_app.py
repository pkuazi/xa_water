import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from cjfranzini_unet_model import get_unet,prediction_history, history_plot,generator
import pickle

# set network size params
N_CLASSES = 6 # farm, forest, grass, buildings, water,other
N_CHANNEL = 3

# define metrics
smooth = 1.

NUM_EPOCHS = 75
INPUT_SIZE = 256

# Define callback to save model checkpoints
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
model_checkpoint = ModelCheckpoint(os.path.join('checkpoints', 'weights.{epoch:02d}-{val_loss:.5f}.hdf5'), 
                                   monitor='loss', 
                                   save_best_only=True)

# Define callback to reduce learning rate when learning stagnates
# This won't actually kick in with only 5 training epochs, but I'm assuming you'll train for hundreds of epochs when you get serious about training this NN.
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5, 
                              patience=3, 
                              epsilon=0.002, 
                              cooldown=1)

# # Define rate scheduler callack (this is an alternative to ReduceLROnPlateau. There is no reason to use both.)
# schedule = lambda epoch_i: 0.01*np.power(0.97, i)
# schedule_lr = LearningRateScheduler(schedule)

# TensorBoard visualizations... this stuff is so freaking cool
tensorboard = TensorBoard(log_dir='/tmp/tboard_logs2', 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=True)

predictions=prediction_history()

lr = 1e-4
batch_size = 64
steps = 3000

BASE_DIR = '/mnt/rsimages/lulc/AISample'
X_train_file = os.path.join(BASE_DIR,'cache/train-set/X_train.p')
X_train = pickle.load(open(X_train_file,'rb'))

Y_train_file = os.path.join(BASE_DIR,'cache/train-set/Y_train.p')
Y_train = pickle.load(open(Y_train_file,'rb'))

X_test_file = os.path.join(BASE_DIR,'cache/test-set/X_test.p')
X_val = pickle.load(open(X_test_file,'rb'))

Y_test_file = os.path.join(BASE_DIR,'cache/test-set/Y_test.p')
Y_val = pickle.load(open(Y_test_file,'rb'))

# verify shape of sets
print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)

# Train the model
model = get_unet(lr,INPUT_SIZE, N_CHANNEL,N_CLASSES) #lr=0.001

fit = model.fit_generator(generator(x_train=X_train, y_train=Y_train, batch_size=batch_size), 
                    steps_per_epoch=steps, 
                    epochs=NUM_EPOCHS, 
                    callbacks=[model_checkpoint, reduce_lr, tensorboard, predictions],
                    validation_data=(X_val, Y_val))

# pickle model
model.save('../pickle_jar/unet_XXIX_{:.3f}_{:.3f}'.format(fit.history['val_loss'][-1],
                                                        fit.history['val_jacc_coef'][-1]))


# pred_img = ''
# pred = model.predict
# # # save prediction history for a few images
# # img_28_preds = [predictions.predhis[i][28,:,:,0] for i in range(NUM_EPOCHS)]
# # img_74_preds = [predictions.predhis[i][74,:,:,0] for i in range(NUM_EPOCHS)]
# # 
# # img_28_set = [X_val[28], Y_val[28], img_28_preds]
# # img_74_set = [X_val[74], Y_val[74], img_74_preds]
# # 
# # pickle.dump(img_28_set, open('../pickle_jar/img_28_set_X.p','wb'))
# # pickle.dump(img_74_set, open('../pickle_jar/img_74_set_X.p','wb'))
# # 
# # 
# # history_plot(fit)
# 
# for i in range(Y_val.shape[0]):
#     
#     x = X_val[i]
#     y = Y_val[i]
# 
#     # Pick out which target to look at
#     CLASS_NO = 0
#     targ = y[:, :, CLASS_NO]
# 
#     # Run the model on that sample
#     pred = model.predict(X_val)[i, :, :, CLASS_NO]
#     
# 
#     # Plot it
#     fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,8))
# #     ax1.imshow(scale_bands(x[:,:,[4,2,1]])) # This index starts at 0, so I had to decrement
#     ax2.imshow(targ, vmin=0, vmax=1)
#     ax3.imshow(pred, vmin=0, vmax=1)
# 
#     ax1.set_title('Image')
#     ax2.set_title('Ground Truth');
#     ax3.set_title('Prediction');
#     ax1.grid()
#     ax2.grid()
#     ax3.grid()
#     plt.show()
#     print('{}/{}'.format(i + 1, Y_val.shape[0]))
#     
#     break
#     
#     time.sleep(1)
#     display.clear_output(wait=True)
# 
# print('DONE')