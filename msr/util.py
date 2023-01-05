########## Incense ##########
from incense import ExperimentLoader

# Try to locate config file for Mongo DB
import importlib
spec = importlib.util.find_spec('mongodburi')
if spec is not None:
    from mongodburi import mongo_uri, db_name
else:
    mongo_uri, db_name = None, None


def get_loader(uri=mongo_uri, db=db_name):
    loader = ExperimentLoader(
        mongo_uri=uri,
        db_name=db
    )
    return loader


########## Util ##########
import numpy as np


