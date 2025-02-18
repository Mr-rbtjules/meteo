
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "./data/logs")
MODELS_DIR = os.path.join(BASE_DIR, "./data/saved_models")
RAW_DATA_DIR = os.path.join(BASE_DIR, "./data/lorenzData")
FIG_DIR = os.path.join(BASE_DIR, "./data/figures")






SEED_NP = 4
SEED_RAND = 5
SEED_SHUFFLE = 6
TEST_DATA_PROPORTION = 0.2
BATCH_SIZE = 16
#more than one if non trivial computation in getitem
NUM_WORKERS = 1 #more than 1 = slower wtf => keep low for low data


"""
#seed for data shuffle
np.random.seed(API.config.SEED_NP)

    #python seed to be sure
random.seed(API.config.SEED_RAND)
"""