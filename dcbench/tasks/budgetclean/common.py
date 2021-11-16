import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


class Preprocessor(object):
    """docstring for Preprocessor."""

    def __init__(self, num_strategy="mean"):
        super(Preprocessor, self).__init__()
        self.num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=num_strategy)),
                ("scaler", MinMaxScaler()),
            ]
        )
        self.feature_enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        self.label_enc = LabelEncoder()

    def fit(self, X_train, y_train, X_full=None):
        self.num_features = X_train.select_dtypes(include="number").columns
        self.cat_features = X_train.select_dtypes(exclude="number").columns

        if len(self.num_features) > 0:
            self.num_transformer.fit(X_train[self.num_features].values)

        if len(self.cat_features) > 0:
            if X_full is None:
                X_full = X_train
            # self.feature_enc.fit(X_full[self.cat_features].values)
            # self.cat_imputer.fit(X_train[self.cat_features].values)
            self.cat_transformer = Pipeline(
                steps=[("imputer", self.cat_imputer), ("onehot", self.feature_enc)]
            )
            self.cat_transformer.fit(X_full[self.cat_features].values)

        self.label_enc.fit(y_train.values.ravel())

    def transform(self, X=None, y=None):
        if X is not None:
            X_after = []
            if len(self.num_features) > 0:
                X_arr = X[self.num_features].values
                if len(X_arr.shape) == 1:
                    X_arr = X_arr.reshape(1, -1)
                X_num = self.num_transformer.transform(X_arr)
                X_after.append(X_num)

            if len(self.cat_features) > 0:
                X_arr = X[self.cat_features].values.astype(object)
                if len(X_arr.shape) == 1:
                    X_arr = X_arr.reshape(1, -1)
                X_cat = self.cat_transformer.transform(X_arr)
                X_after.append(X_cat)

            X = np.hstack(X_after)

        if y is not None:
            y = self.label_enc.transform(y.values.ravel())

        if X is None:
            return y
        elif y is None:
            return X
        else:
            return X, y
