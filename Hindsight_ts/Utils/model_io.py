import codecs
import pickle

# Encode pickled object and decode to string
# Object is encoded using base64
def save_to_string(obj):
    return codecs.encode(pickle.dumps(obj), 'base64').decode()

# Loads model object from string
def load_model_from_string(pickled):
    return pickle.loads(codecs.decode(pickled.encode(), 'base64'))

# Loads model object from given txt file
def load_model_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.read()
    return load_model_from_string(lines)
