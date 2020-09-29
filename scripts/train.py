import os
import numpy as np
from absl import logging
from absl import app
from absl import flags
import tensorflow as tf  
from utils.utils import *
from utils.trainer import ModelEnvironment
from utils.summary_utils import Summaries
from models.resnet import getResnetModel
from models.eval_functions.classifier_loss import EvalFunctions

HEIGHT = 32 
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
_NUM_DATA_FILES = 5

# TODO(tobyboyd): Change to best practice 45K(train)/5K(val)/10K(test) splits.
NUM_IMAGES = {
    'train': 45000,
    'validation': 5000,
    'test': 10000,
}


def preprocess_image(image):
    """Preprocess a single image of layout [height, width, depth]."""
    return image


def data_generator(data,batch_size,is_training,is_validation=False,take_n=None,skip_n=None):
   
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if is_training:
        shuffle_buffer=NUM_IMAGES['train']
    elif is_validation:
        shuffle_buffer=NUM_IMAGES['validation']
    else:
        shuffle_buffer=NUM_IMAGES['test']

    if skip_n != None:
        dataset = dataset.skip(skip_n)
    if take_n != None:
        dataset = dataset.take(take_n)

    if is_training:

        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(lambda img, lbl: (preprocess_image(img), lbl))
        dataset = dataset.map(lambda img, lbl: (img, tf.one_hot(lbl,NUM_CLASSES)))
        dataset = dataset.batch(batch_size,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(lambda img, lbl: (preprocess_image(img), lbl))
        dataset = dataset.map(lambda img, lbl: (img, tf.one_hot(lbl,NUM_CLASSES)))
        dataset = dataset.batch(100)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def learning_rate_fn(epoch):

    if epoch >= 0 and epoch <10:
        return 0.1
    elif epoch >=100 and epoch <120:
        return 0.1
    elif epoch >=120 and epoch < 150:
        return 0.01
    elif epoch >= 150:
        return 0.001
    else:
        return 1.0


FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '/tmp', 'save directory name')
flags.DEFINE_string('data_dir', '/tmp', 'data directory name')
flags.DEFINE_string('mode', 'local', 'Mode for the training local or cluster')
flags.DEFINE_float('dropout', 0.0, 'dropout rate for the dense blocks')
flags.DEFINE_float('weight_decay', 1e-4, 'weight decay parameter')
flags.DEFINE_float('learning_rate', 1e-1, 'learning rate')
flags.DEFINE_integer('epochs', 200, 'number of epochs')
flags.DEFINE_integer('start_epoch', 0, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size')
flags.DEFINE_integer('num_blocks', 10, 'Number of blocks for Network')
flags.DEFINE_integer('start_filter', 11, 'Filters for start')
flags.DEFINE_string('dataset','cifar10', 'Dataset')
flags.DEFINE_string('act','relu', 'Activation')
flags.DEFINE_string('aact','crelu', 'Advanced activation')
flags.DEFINE_string('model','complex', 'Model type')
flags.DEFINE_string('spectral_pool_scheme','none', 'Model type')
flags.DEFINE_string('spectral_param','store_true', 'Spectral parametrization')
flags.DEFINE_string('comp_init','complex_independent', 'Complex initialization')
flags.DEFINE_float('spectral_pool_gamma', 0.5, '')

flags.DEFINE_boolean('load_model', False, 'Bool indicating if the model should be loaded')
flags.DEFINE_integer('eval_every_n_th_epoch', 1, 'Integer discribing after how many epochs the test and validation set are evaluted')


def main(argv):
    
    try:

        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    except KeyError:

        task_id = 0
    
    
    model_save_dir = FLAGS.model_dir
    data_dir = FLAGS.data_dir
    print("Saving model to : " + str(model_save_dir))
    print("Loading data from : " + str(data_dir))
    test_data_dir = data_dir
    train_data_dir = data_dir
    epochs = FLAGS.epochs
    start_epoch = FLAGS.start_epoch
    dropout_rate = FLAGS.dropout
    weight_decay = FLAGS.weight_decay
    learning_rate = FLAGS.learning_rate
    load_model = FLAGS.load_model
    batch_size = FLAGS.batch_size
    model_save_dir+="_dropout_rate_"+str(dropout_rate)+"_learning_rate_"+str(learning_rate)+"_weight_decay_"+str(weight_decay)

    # Create parameter dict
    params = {}
    params["learning_rate"] = learning_rate
    params["model_dir"] = model_save_dir
    params["data_dir"] = data_dir
    params["weight_decay"] = weight_decay
    params["dropout_rate"] = dropout_rate
    
    #If load_model get old configuration
    if load_model:
        try:
            params = csv_to_dict(os.path.join(model_dir, "model_params.csv"))
        except:
            print("Could not find model hyperparameters!")


    #ResNet 18
    model = getResnetModel(FLAGS)
    

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    X_train    = X_train.astype('float32')/255.0
    X_test     = X_test.astype('float32')/255.0
    X_train_split = X_train[NUM_IMAGES['train']]

    pixel_mean = np.mean(X_train_split, axis=0)
    
    X_train    = X_train.astype(np.float32) - pixel_mean
    X_test     = X_test.astype(np.float32) - pixel_mean
    
    #Train data generator
    train_ds = data_generator((X_train,y_train),batch_size,is_training=True,take_n=NUM_IMAGES["train"])
    steps_train = NUM_IMAGES["train"]//batch_size
    #Validation data generator
    val_ds = data_generator((X_train,y_train),100,is_training=False,is_validation = True,skip_n=NUM_IMAGES["train"],take_n=NUM_IMAGES["validation"])
    #Test data generator
    test_ds = data_generator((X_test,y_test),100,is_training=False)
    #Create summaries to log
    scalar_summary_names = ["total_loss",
                            "class_loss",
                            "weight_decay_loss",
                            "accuracy"]

    summaries = Summaries(scalar_summary_names = scalar_summary_names,
                          learning_rate_names = ["learning_rate"],
                          save_dir = model_save_dir,
                          modes = ["train","val","test"],
                          summaries_to_print={"train": ["total_loss","weight_decay_loss", "accuracy"],
                                              "eval":["total_loss", "accuracy"]})

    #Create training setttings for models
    model_settings = [{'model': model,
            'optimizer_type': tf.keras.optimizers.SGD,
            'base_learning_rate': learning_rate,
            'learning_rate_fn': learning_rate_fn,
            'init_data': tf.random.normal([batch_size,HEIGHT,WIDTH,NUM_CHANNELS]),
            'trainable':True}]
    
    #Write training configuration into .csv file
    write_params_csv(model_save_dir, params)

    # Build training environment
    trainer = ModelEnvironment(train_ds,
                               val_ds,
                               test_ds,
                               epochs,
                               EvalFunctions,
                               model_settings=model_settings,
                               summaries=summaries,
                               eval_every_n_th_epoch = 1,
                               num_train_batches=steps_train,
                               load_model=load_model,
                               save_dir = model_save_dir,
                               input_keys=[0],
                               label_keys=[1],
                               start_epoch=start_epoch)
    
    trainer.train()

if __name__ == '__main__':
  app.run(main)
  
