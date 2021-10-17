from pydybm.time_series.dybm import BinaryDyBM
from pydybm.base.generator import SequenceGenerator
from Utils.ld_parser import *
import scipy
from sklearn.preprocessing import OneHotEncoder

def add_one_hot_encoding(data, vals, mapping):
    assert len(mapping) == len(set(vals))
    r_data = copy.deepcopy(data)
    ohe = OneHotEncoder(sparse=False)
    res = ohe.fit_transform(vals.reshape(len(vals), 1))
    for i in range(len(mapping)):
        r_data[mapping[i]] = res[:, i]
    return r_data

# Receives a Dataframe and one-hot encodes non binary columns
# Returns the equivalents of each column provided in cols, if any
def parse_data(data, cols=[], ignore_first=False):
    b_data = pandas.DataFrame()
    new_cols = []
    if ignore_first:
        b_data[data.columns[0]] = data[data.columns[0]]
        start = 1
    else:
        start = 0
    for col in data.columns[start:]:
        col_dim = len(set(data[col].values))

        # If column has more than two possible values
        if col_dim > 2:
            mapping = []
            for i in range(col_dim):
                mapping.append(str(col) + '_' + str(i))
                if col in cols:
                    new_cols.append(str(col) + '_' + str(i))
            b_data = add_one_hot_encoding(b_data, data[col].values, mapping)

        # If column only has two possible values
        else:
            b_data[col] = data[col]
            if col in cols:
                new_cols.append(col)
    return b_data, new_cols

def parse_predictions(res):
    pred = []
    actual = []
    for i in range(len(res['prediction'])):
        pred.append(np.argmax(res['prediction'][i]))
        actual.append(np.argmax(res['actual'][i]))
    return pred, actual

def interpret_results(pred, actual):
    total_correct = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            total_correct += 1
    group_correct = int(scipy.stats.mode(pred).mode[0] == scipy.stats.mode(actual).mode[0])
    return total_correct, group_correct

def train_dybm(model, dataset, inp_vars, out_vars, iterations=5, id_col=None, test_dataset=None):
    recalls = []
    recalls_test = []
    if id_col is None:
        id_col = dataset.columns[0]
    for it in range(iterations):
        total_correct = 0
        total_pred = 0
        group_correct = 0
        group_pred = 0

        for s in subjects(dataset, id_col):
            current = get_subject(dataset, s, id_col)
            #current = current[current.columns[col_offset:]]
            in_seq = SequenceGenerator(current[inp_vars].values)
            out_seq = SequenceGenerator(current[out_vars].values)
            for i in range(5):
                model.init_state()
                in_seq.reset(0)
                out_seq.reset(0)
                res = model.learn(in_seq, out_seq, get_result=True)
            pred, actual = parse_predictions(res)
            total_res, group_res = interpret_results(pred, actual)
            total_correct += total_res
            group_correct += group_res
            total_pred += len(pred)
            group_pred += 1

        # Obtain training and testing recalls
        recalls.append(total_correct / total_pred)
        if not test_dataset is None:
            recalls_test.append(pred_dybm(model, test_dataset, inp_vars, out_vars, id_col=id_col))

    all_recalls = {'training': recalls, 'validation': recalls_test}

    return model, all_recalls

def pred_dybm(model, dataset, inp_vars, out_vars, id_col=None):
    total_correct = 0
    total_pred = 0
    group_correct = 0
    group_pred = 0

    if id_col is None:
        id_col = dataset.columns[0]

    for s in subjects(dataset, id_col):
        current = get_subject(dataset, s, id_col)
        #current = current[current.columns[col_offset:]]
        in_seq = SequenceGenerator(current[inp_vars].values)
        out_seq = SequenceGenerator(current[out_vars].values)
        model.init_state()
        actual = out_seq.to_list()
        pred = model.get_predictions(in_seq)
        res = {'prediction': pred, 'actual': actual}
        pred, actual = parse_predictions(res)
        total_res, group_res = interpret_results(pred, actual)
        total_correct += total_res
        group_correct += group_res
        total_pred += len(pred)
        group_pred += 1

    return total_correct / total_pred

def create_dybm(inp_vars, out_vars, delay, decay=[0.5]):
    in_dim = len(inp_vars)
    out_dim = len(out_vars)
    model = BinaryDyBM(in_dim, out_dim=out_dim, delay=2, decay_rates=decay)
    return model

def dybm_procedure(data, inp_vars, out_vars, delay, decay=[0.5], iterations=5, id_col=None, test_dataset=None):
    model = create_dybm(inp_vars, out_vars, delay, decay=decay)
    model, recalls = train_dybm(model, data, inp_vars, out_vars, iterations=iterations, id_col=None, test_dataset=test_dataset)
    return model, recalls
