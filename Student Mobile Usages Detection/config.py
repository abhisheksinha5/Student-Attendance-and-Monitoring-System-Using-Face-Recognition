import os

cwd = os.getcwd()

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(f'{cwd}/result'):
    os.mkdir(f'{cwd}/result')

TRAIN_DATASET_PATH = f'{cwd}/dataset_mobile/train'
VALID_DATASET_PATH = f'{cwd}/dataset_mobile/valid'
TEST_DATASET_PATH = f'{cwd}/dataset_mobile/test'
MODEL_PATH = f'{cwd}/model'

MODEL = 'efficientdet_lite0'
MODEL_NAME = 'mobile.tflite'
CLASSES = ['Mobile Phone']
EPOCHS = 300
BATCH_SIZE = 4