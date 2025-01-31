
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "./data/logs")
MODELS_DIR = os.path.join(BASE_DIR, "./data/saved_models")
RAW_DATA_DIR = os.path.join(BASE_DIR, "./data/lorenzData")






SEED_NP = 4
SEED_RAND = 5

TEST_DATA_PROPORTION = 0.2

"""
#seed for data shuffle
np.random.seed(API.config.SEED_NP)

    #python seed to be sure
random.seed(API.config.SEED_RAND)
"""