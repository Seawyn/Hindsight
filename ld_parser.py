import pandas
import sys

# Handle version difference
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

class ld_parser:
    # Initializes class with a given .csv filename and column name of the subjects
    def __init__(self, filename, subject_col):
        self.data = pandas.read_csv(filename)
        self.subject_col = subject_col

    # Returns all subjects in the data
    def subjects(self):
        return list(set(self.data[self.subject_col]))

    # Returns a dataframe with data of a given subject
    def get_subject(self, subject_id):
        return self.data.where(self.data[self.subject_col] == subject_id).dropna(how='all')

    # Replaces values in a given column
    def replace_in_col(self, col, repl):
        self.data = self.data.replace({col: repl})

    # Presents a list of variables of the data, followed by general statistics
    def summary(self, subject=None):
        if subject is None:
            data = self.data
        else:
            data = self.get_subject(subject)

        for col in data.columns:
            if col == self.subject_col:
                continue
            else:
                print(col)
                number_of_nan = data[col].isna().sum()
                number_of_distinct = len(pandas.DataFrame(set(data[col])).dropna())
                print('Number of observations:', len(data[col]))
                print('Number of missing values:', number_of_nan)
                print('Percentage of missing values:', number_of_nan / len(data[col]))
                print('Number of distinct observations:', number_of_distinct)
                print('Percentage of distinct values:', number_of_distinct / len(data[col]))
                print('-----------------')

    # Detects and drops columns of data with percentage of missing values above 30%
    def drop_cols_with_high_missingness(self, data, threshold=0.3):
        for col in data.columns:
            if col == self.subject_col:
                continue
            percentage_of_nan = data[col].isna().sum() / len(data[col])
            if percentage_of_nan > threshold:
                data = data.drop(columns=[col])
                print("Dropped", col, " column")
        return data

    # Imputation of discrete variables based on MissForest
    def impute(self, data, imp_vars=None, categorical_variables=None):
        # Prepare input
        inp = data
        if not imp_vars is None:
            inp = data[imp_vars]
        if len(inp.columns) < 2:
            raise Exception("Multiple variables must be given as input")

        # Prepare MissForest Imputer
        imputer = MissForest()
        if not categorical_variables is None:
            cat_vars = []
            for categorical_variable in categorical_variables:
                cat_vars.append(list(inp.columns).index(categorical_variable))

        # Fit and Transform the input
        res = imputer.fit_transform(inp.values, cat_vars=cat_vars)
        return res

