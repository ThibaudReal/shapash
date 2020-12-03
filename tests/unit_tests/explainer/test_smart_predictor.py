"""
Unit test smart predictor
"""

import unittest
from shapash.explainer.smart_predictor import SmartPredictor
from shapash.explainer.smart_explainer import SmartExplainer
from shapash.explainer.smart_state import SmartState
from shapash.explainer.multi_decorator import MultiDecorator
import os
from os import path
from pathlib import Path
import pandas as pd
import numpy as np
import catboost as cb
from catboost import Pool
import category_encoders as ce
from unittest.mock import patch
import types
from sklearn.compose import ColumnTransformer
import sklearn.preprocessing as skp
import shap

def init_sme_to_pickle_test():
    """
    Init sme to pickle test
    TODO: Docstring
    Returns
    -------
    [type]
        [description]
    """
    current = Path(path.abspath(__file__)).parent.parent.parent
    pkl_file = path.join(current, 'data/predictor.pkl')
    xpl = SmartExplainer(features_dict={})
    y_pred = pd.DataFrame(data=np.array([1, 2]), columns=['pred'])
    dataframe_x = pd.DataFrame([[1, 2, 4], [1, 2, 3]])
    clf = cb.CatBoostClassifier(n_estimators=1).fit(dataframe_x, y_pred)
    xpl.compile(x=dataframe_x, y_pred=y_pred, model=clf)
    predictor = xpl.to_smartpredictor()
    return pkl_file, predictor

class TestSmartPredictor(unittest.TestCase):
    """
    Unit test Smart Predictor class
    """
    def setUp(self):
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        clf_explainer = shap.TreeExplainer(clf)

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}

        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}

        features_types = {features: str(df[features].dtypes) for features in df[['x1', 'x2']]}

        self.df_1 = df
        self.preprocessing_1 = encoder_fitted
        self.df_encoded_1 = df_encoded
        self.clf_1 = clf
        self.clf_explainer_1 = clf_explainer
        self.columns_dict_1 = columns_dict
        self.label_dict_1 = label_dict
        self.postprocessing_1 = postprocessing
        self.features_dict_1 = features_dict
        self.features_types_1 = features_types

        self.predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing)

        df['x2'] = np.random.randint(1, 100, df.shape[0])
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df[["x1", "x2"]])
        df_encoded = encoder_fitted.transform(df[["x1", "x2"]])

        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)
        features_dict = {"x1": "age", "x2": "weight"}
        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        self.df_2 = df
        self.preprocessing_2 = encoder_fitted
        self.df_encoded_2 = df_encoded
        self.clf_2 = clf
        self.clf_explainer_2 = clf_explainer
        self.columns_dict_2 = columns_dict
        self.label_dict_2 = label_dict
        self.postprocessing_2 = postprocessing
        self.features_dict_2 = features_dict
        self.features_types_2 = features_types

        self.predictor_2 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing)

        df['x1'] = [25, 39, 50, 43, 67]
        df['x2'] = [90, 78, 84, 85, 53]

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "No", 1: "Yes"}
        features_dict = {"x1": "age", "x2": "weight"}

        features_types = {features: str(df[features].dtypes) for features in df[['x1', 'x2']].columns}

        clf = cb.CatBoostRegressor(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)

        self.df_3 = df
        self.preprocessing_3 = None
        self.df_encoded_3 = df
        self.clf_3 = clf
        self.clf_explainer_3 = clf_explainer
        self.columns_dict_3 = columns_dict
        self.label_dict_3 = label_dict
        self.postprocessing_3 = None
        self.features_dict_3 = features_dict
        self.features_types_3 = features_types

        self.predictor_3 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict)

    def predict_proba(self, arg1, arg2):
        """
        predict_proba method
        """
        matrx = np.array(
            [[0.2, 0.8],
             [0.3, 0.7],
             [0.4, 0.6]]
        )
        return matrx

    def predict(self, arg1, arg2):
        """
        predict method
        """
        matrx = np.array(
            [12, 3, 7]
        )
        return matrx

    def test_init_1(self):
        """
        Test init smart predictor
        """
        predictor_1 = self.predictor_1

        mask_params = {
            'features_to_hide': None,
            'threshold': None,
            'positive': True,
            'max_contrib': 1
        }

        predictor_2 = predictor_1
        predictor_2.mask_params = mask_params

        assert hasattr(predictor_1, 'model')
        assert hasattr(predictor_1, 'explainer')
        assert hasattr(predictor_1, 'features_dict')
        assert hasattr(predictor_1, 'label_dict')
        assert hasattr(predictor_1, '_case')
        assert hasattr(predictor_1, '_classes')
        assert hasattr(predictor_1, 'columns_dict')
        assert hasattr(predictor_1, 'features_types')
        assert hasattr(predictor_1, 'preprocessing')
        assert hasattr(predictor_1, 'postprocessing')
        assert hasattr(predictor_1, 'mask_params')
        assert hasattr(predictor_2, 'mask_params')

        assert predictor_1.model == self.clf_1
        assert predictor_1.explainer == self.clf_explainer_1
        assert predictor_1.features_dict == self.features_dict_1
        assert predictor_1.label_dict == self.label_dict_1
        assert predictor_1._case == "classification"
        assert predictor_1._classes == [0,1]
        assert predictor_1.columns_dict == self.columns_dict_1
        assert predictor_1.preprocessing == self.preprocessing_1
        assert predictor_1.postprocessing == self.postprocessing_1

        assert predictor_2.mask_params == mask_params

    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_model')
    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_explainer')
    @patch('shapash.utils.check.check.check_preprocessing_options')
    @patch('shapash.utils.check.check_consistency_model_features')
    @patch('shapash.utils.check.check_consistency_model_label')
    def add_input_1(self, check_consistency_model_label,
                    check_consistency_model_features,
                    check_preprocessing_options,
                    check_explainer,
                    check_model):
        """
        Test add_input method from smart predictor
        """
        check_preprocessing_options.return_value = True
        check_consistency_model_features.return_value = True
        check_consistency_model_label.return_value = True
        check_explainer.return_value = self.clf_explainer_1
        check_model.return_value = "classification", [0, 1]

        ypred = self.df_1['y']
        shap_values = self.clf_1.get_feature_importance(Pool(self.df_encoded_1), type="ShapValues")

        predictor_1 = self.predictor_1
        predictor_1.add_input(x=self.df_1[["x1", "x2"]], contributions=shap_values[:, :-1])
        predictor_1_contrib = predictor_1.data["contributions"]

        assert all(attribute in predictor_1.data.keys()
                   for attribute in ["x", "x_preprocessed", "contributions", "ypred"])
        assert predictor_1.data["x"].shape == predictor_1.data["x_preprocessed"].shape
        assert all(feature in predictor_1.data["x"].columns
                   for feature in predictor_1.data["x_preprocessed"].columns)
        assert predictor_1_contrib.shape == predictor_1.data["x"].shape

        predictor_1.add_input(ypred=ypred)

        assert "ypred" in predictor_1.data.keys()
        assert predictor_1.data["ypred"].shape[0] == predictor_1.data["x"].shape[0]
        assert all(predictor_1.data["ypred"].index == predictor_1.data["x"].index)

    @patch('shapash.explainer.smart_predictor.SmartState')
    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_model')
    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_explainer')
    @patch('shapash.utils.check.check_preprocessing_options')
    @patch('shapash.utils.check.check_consistency_model_features')
    @patch('shapash.utils.check.check_consistency_model_label')
    def test_choose_state_1(self, check_consistency_model_label,
                    check_consistency_model_features,
                    check_preprocessing_options,
                    check_explainer,
                    check_model, mock_smart_state):
        """
        Unit test choose state 1
        Parameters
        ----------
        mock_smart_state : [type]
            [description]
        """
        check_preprocessing_options.return_value = True
        check_consistency_model_features.return_value = True
        check_consistency_model_label.return_value = True
        check_explainer.return_value = self.clf_explainer_1
        check_model.return_value = "classification", [0, 1]

        predictor_1 = self.predictor_1
        predictor_1.choose_state('contributions')
        mock_smart_state.assert_called()

    @patch('shapash.explainer.smart_predictor.MultiDecorator')
    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_model')
    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_explainer')
    @patch('shapash.utils.check.check_preprocessing_options')
    @patch('shapash.utils.check.check_consistency_model_features')
    @patch('shapash.utils.check.check_consistency_model_label')
    def test_choose_state_2(self, check_consistency_model_label,
                    check_consistency_model_features,
                    check_preprocessing_options,
                    check_explainer,
                    check_model, mock_multi_decorator):
        """
        Unit test choose state 2
        Parameters
        ----------
        mock_multi_decorator : [type]
            [description]
        """
        check_preprocessing_options.return_value = True
        check_consistency_model_features.return_value = True
        check_consistency_model_label.return_value = True
        check_explainer.return_value = self.clf_explainer_1
        check_model.return_value = "classification", [0, 1]

        predictor_1 = self.predictor_1
        predictor_1.choose_state('contributions')
        predictor_1.choose_state([1, 2, 3])
        mock_multi_decorator.assert_called()

    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_model')
    @patch('shapash.explainer.smart_predictor.SmartPredictor.check_explainer')
    @patch('shapash.utils.check.check_preprocessing_options')
    @patch('shapash.utils.check.check_consistency_model_features')
    @patch('shapash.utils.check.check_consistency_model_label')
    @patch('shapash.explainer.smart_predictor.SmartPredictor.choose_state')
    def test_validate_contributions_1(self, choose_state,
                                      check_consistency_model_label,
                                      check_consistency_model_features,
                                      check_preprocessing_options,
                                      check_explainer,
                                      check_model):
        """
        Unit test validate contributions 1
        """
        check_preprocessing_options.return_value = True
        check_consistency_model_features.return_value = True
        check_consistency_model_label.return_value = True
        check_explainer.return_value = self.clf_explainer_1
        check_model.return_value = "classification", [0, 1]
        choose_state.return_value = MultiDecorator(SmartState())

        predictor_1 = self.predictor_1

        contributions = [
            np.array([[2, 1], [8, 4]]),
            np.array([[5, 5], [0, 0]])
        ]
        predictor_1.state = predictor_1.choose_state(contributions)
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x_preprocessed"] = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        expected_output = [
            pd.DataFrame(
                [[2, 1], [8, 4]],
                columns=['Col1', 'Col2'],
                index=['Id1', 'Id2']
            ),
            pd.DataFrame(
                [[5, 5], [0, 0]],
                columns=['Col1', 'Col2'],
                index=['Id1', 'Id2']
            )
        ]
        output = predictor_1.validate_contributions(contributions)
        assert len(expected_output) == len(output)
        test_list = [pd.testing.assert_frame_equal(e, m) for e, m in zip(expected_output, output)]
        assert all(x is None for x in test_list)

    def test_check_contributions(self):
        """
        Unit test check_shape_contributions 1
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 100, df.shape[0])
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df[["x1", "x2"]])
        df_encoded = encoder_fitted.transform(df[["x1", "x2"]])
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        features_dict = {"x1": "age", "x2": "weight"}
        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        shap_values = self.clf_2.get_feature_importance(Pool(self.df_encoded_2), type="ShapValues")

        predictor_1 = self.predictor_2

        predictor_1.add_input(x=self.df_2[["x1", "x2"]], contributions=shap_values[:, :-1], ypred=self.df_2["y"])

        adapt_contrib = predictor_1.adapt_contributions(shap_values[:, :-1])
        predictor_1.state = predictor_1.choose_state(adapt_contrib)
        contributions = predictor_1.validate_contributions(adapt_contrib)
        predictor_1.check_contributions(contributions)

        with self.assertRaises(ValueError):
            predictor_1.check_contributions(shap_values[:, :-1])

    def test_check_model_1(self):
        """
        Unit test check model 1
        """
        predictor_1 = self.predictor_1

        model = lambda: None
        model.n_features_in_ = 2
        model.predict = types.MethodType(self.predict, model)

        predictor_1.model = model
        _case, _classes = predictor_1.check_model()
        assert _case == 'regression'
        assert _classes is None

    def test_check_model_2(self):
        """
        Unit test check model 2
        """
        predictor_1 = self.predictor_1

        model = lambda: None
        model._classes = np.array([1, 2])
        model.n_features_in_ = 2
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)

        predictor_1.model = model

        _case, _classes = predictor_1.check_model()
        assert _case == 'classification'
        self.assertListEqual(_classes, [1, 2])

    def test_check_preprocessing_1(self):
        """
        Test check preprocessing on multiple preprocessing
        """
        train = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D'],
                              'Binary1': ['E', 'F', 'E', 'F'], 'Binary2': ['G', 'H', 'G', 'H'],
                              'Ordinal1': ['I', 'J', 'I', 'J'], 'Ordinal2': ['K', 'L', 'K', 'L'],
                              'BaseN1': ['M', 'N', 'M', 'N'], 'BaseN2': ['O', 'P', 'O', 'P'],
                              'Target1': ['Q', 'R', 'Q', 'R'], 'Target2': ['S', 'T', 'S', 'T'],
                              'other': ['other', np.nan, 'other', 'other']})

        features_dict = None
        columns_dict = {i:features for i,features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        label_dict = None

        enc_ordinal_all = ce.OrdinalEncoder(cols=['Onehot1', 'Onehot2', 'Binary1', 'Binary2', 'Ordinal1', 'Ordinal2',
                                            'BaseN1', 'BaseN2', 'Target1', 'Target2', 'other']).fit(train)
        train_ordinal_all  = enc_ordinal_all.transform(train)

        y = pd.DataFrame({'y_class': [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal_all, y)
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)


        y = pd.DataFrame(data=[0, 1, 0, 0], columns=['y'])

        enc_onehot = ce.OneHotEncoder(cols=['Onehot1', 'Onehot2']).fit(train)
        train_onehot = enc_onehot.transform(train)
        enc_binary = ce.BinaryEncoder(cols=['Binary1', 'Binary2']).fit(train_onehot)
        train_binary = enc_binary.transform(train_onehot)
        enc_ordinal = ce.OrdinalEncoder(cols=['Ordinal1', 'Ordinal2']).fit(train_binary)
        train_ordinal = enc_ordinal.transform(train_binary)
        enc_basen = ce.BaseNEncoder(cols=['BaseN1', 'BaseN2']).fit(train_ordinal)
        train_basen = enc_basen.transform(train_ordinal)
        enc_target = ce.TargetEncoder(cols=['Target1', 'Target2']).fit(train_basen, y)

        input_dict1 = dict()
        input_dict1['col'] = 'Onehot2'
        input_dict1['mapping'] = pd.Series(data=['C', 'D', np.nan], index=['C', 'D', 'missing'])
        input_dict1['data_type'] = 'object'

        input_dict2 = dict()
        input_dict2['col'] = 'Binary2'
        input_dict2['mapping'] = pd.Series(data=['G', 'H', np.nan], index=['G', 'H', 'missing'])
        input_dict2['data_type'] = 'object'

        input_dict = dict()
        input_dict['col'] = 'state'
        input_dict['mapping'] = pd.Series(data=['US', 'FR-1', 'FR-2'], index=['US', 'FR', 'FR'])
        input_dict['data_type'] = 'object'

        input_dict3 = dict()
        input_dict3['col'] = 'Ordinal2'
        input_dict3['mapping'] = pd.Series(data=['K', 'L', np.nan], index=['K', 'L', 'missing'])
        input_dict3['data_type'] = 'object'
        list_dict = [input_dict2, input_dict3]

        y = pd.DataFrame(data=[0, 1], columns=['y'])

        train = pd.DataFrame({'city': ['chicago', 'paris'],
                              'state': ['US', 'FR'],
                              'other': ['A', 'B']})
        enc = ColumnTransformer(
            transformers=[
                ('onehot', skp.OneHotEncoder(), ['city', 'state'])
            ],
            remainder='drop')
        enc.fit(train, y)

        wrong_prepro = skp.OneHotEncoder().fit(train, y)

        predictor_1.preprocessing = [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1,
                                           list_dict]
        predictor_1.check_preprocessing()

        for preprocessing in [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target]:
            predictor_1.preprocessing = preprocessing
            predictor_1.check_preprocessing()

        predictor_1.preprocessing = input_dict2
        predictor_1.check_preprocessing()

        predictor_1.preprocessing = enc
        predictor_1.check_preprocessing()

        predictor_1.preprocessing = None
        predictor_1.check_preprocessing()

        with self.assertRaises(Exception):
            predictor_1.preprocessing = wrong_prepro
            predictor_1.check_preprocessing()

    def test_check_label_dict_1(self):
        """
        Unit test check label dict 1
        """
        predictor_1 = self.predictor_1

        predictor_1.check_label_dict()

    def test_check_label_dict_2(self):
        """
        Unit test check label dict 2
        """
        predictor_1 = self.predictor_1

        predictor_1.label_dict = None
        predictor_1._case = 'regression'
        predictor_1.check_label_dict()

    def test_check_mask_params(self):
        """
        Unit test check mask params
        """
        train = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D'],
                              'Binary1': ['E', 'F', 'E', 'F'], 'Binary2': ['G', 'H', 'G', 'H'],
                              'Ordinal1': ['I', 'J', 'I', 'J'], 'Ordinal2': ['K', 'L', 'K', 'L'],
                              'BaseN1': ['M', 'N', 'M', 'N'], 'BaseN2': ['O', 'P', 'O', 'P'],
                              'Target1': ['Q', 'R', 'Q', 'R'], 'Target2': ['S', 'T', 'S', 'T'],
                              'other': ['other', np.nan, 'other', 'other']})

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        label_dict = None

        enc_ordinal = ce.OrdinalEncoder(cols=['Onehot1', 'Onehot2', 'Binary1', 'Binary2', 'Ordinal1', 'Ordinal2',
                                                  'BaseN1', 'BaseN2', 'Target1', 'Target2', 'other']).fit(train)
        train_ordinal = enc_ordinal.transform(train)

        y = pd.DataFrame({'y_class': [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal, y)
        clf_explainer = shap.TreeExplainer(model)

        wrong_mask_params_1 = list()
        wrong_mask_params_2 = None
        wrong_mask_params_3 = {
            "features_to_hide": None,
            "threshold": None,
            "positive": None
        }
        wright_mask_params = {
            "features_to_hide": None,
            "threshold": None,
            "positive": True,
            "max_contrib": 5
        }
        with self.assertRaises(ValueError):
            predictor_1 = SmartPredictor(features_dict, model,
                                         columns_dict, clf_explainer, features_types, label_dict,
                                         mask_params=wrong_mask_params_1)
            predictor_1 = SmartPredictor(features_dict, model,
                                         columns_dict, clf_explainer, features_types, label_dict,
                                         mask_params=wrong_mask_params_2)
            predictor_1 = SmartPredictor(features_dict, model,
                                         columns_dict, clf_explainer, features_types, label_dict,
                                         mask_params=wrong_mask_params_3)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     mask_params=wright_mask_params)

    def test_check_ypred_1(self):
        """
        Unit test check y pred
        """
        predictor_1 = self.predictor_1
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1[["x1","x2"]]
        y_pred = None
        predictor_1.check_ypred(ypred=y_pred)

    def test_check_ypred_2(self):
        """
        Unit test check y pred 2
        """
        y_pred = pd.DataFrame(
            data=np.array(['1', 0, 0, 1, 0]),
            columns=['Y']
        )
        predictor_1 = self.predictor_1
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1

        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_check_ypred_3(self):
        """
        Unit test check y pred 3
        """
        predictor_1 = self.predictor_1

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1[["x1","x2"]]
        y_pred = pd.DataFrame(
            data=np.array([0]),
            columns=['Y']
        )
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_check_y_pred_4(self):
        """
        Unit test check y pred 4
        """
        predictor_1 = self.predictor_1
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        y_pred = [0, 1, 0, 1, 0]
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(ypred=y_pred)

    def test_check_ypred_5(self):
        """
        Unit test check y pred 5
        """
        predictor_1 = self.predictor_1
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1[["x1","x2"]]

        y_pred = pd.Series(
            data=np.array(['0'])
        )
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_predict_proba_1(self):
        """
        Unit test of predict_proba method.
        """
        predictor_1 = self.predictor_1

        clf = cb.CatBoostRegressor(n_estimators=1).fit(self.df_encoded_1[['x1', 'x2']], self.df_encoded_1['y'])
        clf_explainer = shap.TreeExplainer(clf)
        predictor_1.model = clf
        predictor_1.explainer = clf_explainer
        predictor_1._case = "regression"
        predictor_1._classes = None

        with self.assertRaises(AttributeError):
            predictor_1.predict_proba()

        predictor_1 = self.predictor_1

        with self.assertRaises(AttributeError):
            predictor_1.predict_proba()

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        with self.assertRaises(KeyError):
            predictor_1.predict_proba()

    def test_predict_proba_2(self):
        """
        Unit test of predict_proba method.
        """
        clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df_2[['x1', 'x2']], self.df_2['y'])
        predictor_1 = self.predictor_2

        predictor_1.model = clf
        predictor_1.explainer = shap.TreeExplainer(clf)
        predictor_1.preprocessing = None


        predictor_1.data = {"x": None, "ypred": None, "contributions": None, "x_preprocessed":None}
        predictor_1.data["x"] = self.df_2[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_2[["x1", "x2"]]

        prediction = predictor_1.predict_proba()
        assert prediction.shape[0] == predictor_1.data["x"].shape[0]

        predictor_1.data["ypred"] = pd.DataFrame(self.df_2["y"])
        prediction = predictor_1.predict_proba()

        assert prediction.shape[0] == predictor_1.data["x"].shape[0]

    def test_detail_contributions_1(self):
        """
        Unit test of detail_contributions method.
        """
        predictor_1 = self.predictor_1

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data["x_preprocessed"] = self.df_1[["x1", "x2"]]

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

    def test_detail_contributions_2(self):
        """
        Unit test 2 of detail_contributions method.
        """
        clf = cb.CatBoostRegressor(n_estimators=1).fit(self.df_2[['x1', 'x2']], self.df_2['y'])
        predictor_1 = self.predictor_2

        predictor_1.model = clf
        predictor_1.explainer = shap.TreeExplainer(clf)
        predictor_1.preprocessing = None
        predictor_1._case = "regression"
        predictor_1._classes = None

        predictor_1.data = {"x": None, "ypred": None, "contributions": None, "x_preprocessed": None}
        predictor_1.data["x"] = self.df_2[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_2[["x1", "x2"]]
        predictor_1.data["ypred"] = pd.DataFrame(self.df_2["y"])

        contributions = predictor_1.detail_contributions()

        assert contributions.shape[0] == predictor_1.data["x"].shape[0]
        assert all(contributions.index == predictor_1.data["x"].index)
        assert contributions.shape[1] == predictor_1.data["x"].shape[1] + 1

        clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df_2[['x1', 'x2']], self.df_2['y'])
        clf_explainer = shap.TreeExplainer(clf)

        predictor_1 = self.predictor_2
        predictor_1.preprocessing = None
        predictor_1.model = clf
        predictor_1.explainer = clf_explainer
        predictor_1._case = "classification"
        predictor_1._classes = [0, 1]

        false_y = pd.DataFrame({"y_false": [2, 2, 1, 1, 1]})
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x_preprocessed"] = self.df_2[["x1", "x2"]]
        predictor_1.data["x"] = self.df_2[["x1", "x2"]]
        predictor_1.data["ypred"] = false_y

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data["ypred"] = pd.DataFrame(self.df_2["y"])

        contributions = predictor_1.detail_contributions()

        assert contributions.shape[0] == predictor_1.data["x"].shape[0]
        assert all(contributions.index == predictor_1.data["x"].index)
        assert contributions.shape[1] == predictor_1.data["x"].shape[1] + 2

    def test_save_1(self):
        """
        Unit test save 1
        """
        pkl_file, predictor = init_sme_to_pickle_test()
        predictor.save(pkl_file)
        assert path.exists(pkl_file)
        os.remove(pkl_file)

    def test_apply_preprocessing_1(self):
        """
        Unit test for apply preprocessing method
        """
        y = pd.DataFrame(data=[0, 1], columns=['y'])
        train = pd.DataFrame({'num1': [0, 1],
                              'num2': [0, 2]})
        enc = ColumnTransformer(transformers=[('power', skp.QuantileTransformer(n_quantiles=2), ['num1', 'num2'])],
                                remainder='passthrough')
        enc.fit(train, y)
        train_preprocessed = pd.DataFrame(enc.transform(train), index=train.index)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)

        features_types = {features: str(train[features].dtypes) for features in train.columns}
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "num1", 1: "num2"}
        label_dict = {0: "Yes", 1: "No"}
        features_dict = {"num1": "city", "num2": "state"}

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict, enc)
        predictor_1.add_input(x=train)
        output_preprocessed = predictor_1.data["x_preprocessed"]
        assert output_preprocessed.shape == train_preprocessed.shape
        assert [column in clf.feature_names_ for column in output_preprocessed.columns]
        assert all(train.index == output_preprocessed.index)
        assert all([str(type_result) == str(train_preprocessed.dtypes[index])
                    for index, type_result in enumerate(output_preprocessed.dtypes)])

    def test_summarize_1(self):
        """
        Unit test 1 summarize method
        """
        clf = cb.CatBoostRegressor(n_estimators=1).fit(self.df_3[['x1', 'x2']], self.df_3['y'])
        clf_explainer = shap.TreeExplainer(clf)

        predictor_1 = self.predictor_3

        predictor_1.model = clf
        predictor_1._case = "regression"
        predictor_1._classes = None
        predictor_1.explainer = clf_explainer
        print(predictor_1.mask_params)
        predictor_1.add_input(x=self.df_3[["x1", "x2"]], ypred=pd.DataFrame(self.df_3["y"]), contributions=None)
        output = predictor_1.summarize()
        print(output)
        expected_output = pd.DataFrame({
            "y": [1, 1, 0, 0, 0],
            "feature_1": ["weight", "weight", "weight", "weight", "weight"],
            "value_1": ["90", "78", "84", "85", "53"],
            "contribution_1": ["0.0942857", "-0.0235714", "-0.0235714", "-0.0235714", "-0.0235714"],
            "feature_2": ["age", "age", "age", "age", "age"],
            "value_2": ["25", "39", "50", "43", "67"],
            "contribution_2": ["0", "0", "0", "0", "0"]
        }, dtype=object)
        expected_output["y"] = expected_output["y"].astype(int)

        feature_expected = [column for column in expected_output.columns if column.startswith("feature_")]
        feature_output = [column for column in output.columns if column.startswith("feature_")]

        value_expected = [column for column in expected_output.columns if column.startswith("value_")]
        value_output = [column for column in output.columns if column.startswith("value_")]

        contribution_expected = [column for column in expected_output.columns if column.startswith("contribution_")]
        contribution_output = [column for column in output.columns if column.startswith("contribution_")]

        assert expected_output.shape == output.shape
        assert len(feature_expected) == len(feature_output)
        assert len(value_expected) == len(value_output)
        assert len(contribution_expected) == len(contribution_output)

    def test_summarize_2(self):
        """
        Unit test 2 summarize method
        """
        predictor_1 = self.predictor_3
        predictor_1._case = "classification"
        predictor_1._classes = [0, 1]
        clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df_3[['x1', 'x2']], self.df_3['y'])
        clf_explainer = shap.TreeExplainer(clf)
        predictor_1.model = clf
        predictor_1.explainer = clf_explainer

        with self.assertRaises(ValueError):
            predictor_1.summarize()

        predictor_1.add_input(x=self.df_3[["x1", "x2"]], ypred=pd.DataFrame(self.df_3["y"]), contributions=None)
        output = predictor_1.summarize()

        expected_output = pd.DataFrame({
            "y": [1, 1, 0, 0, 0],
            "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209],
            "feature_1": ["weight", "weight", "weight", "weight", "weight"],
            "value_1": ["90", "78", "84", "85", "53"],
            "contribution_1": ["0.0942857", "-0.0235714", "-0.0235714", "-0.0235714", "-0.0235714"],
            "feature_2": ["age", "age", "age", "age", "age"],
            "value_2": ["25", "39", "50", "43", "67"],
            "contribution_2": ["0", "0", "0", "0", "0"]
        }, dtype=object)
        expected_output["y"] = expected_output["y"].astype(int)
        expected_output["proba"] = expected_output["proba"].astype(float)

        feature_expected = [column for column in expected_output.columns if column.startswith("feature_")]
        feature_output = [column for column in output.columns if column.startswith("feature_")]

        value_expected = [column for column in expected_output.columns if column.startswith("value_")]
        value_output = [column for column in output.columns if column.startswith("value_")]

        contribution_expected = [column for column in expected_output.columns if column.startswith("contribution_")]
        contribution_output = [column for column in output.columns if column.startswith("contribution_")]

        assert expected_output.shape == output.shape
        assert len(feature_expected) == len(feature_output)
        assert len(value_expected) == len(value_output)
        assert len(contribution_expected) == len(contribution_output)
        assert all(output.columns == expected_output.columns)

    def test_summarize_3(self):
        """
        Unit test 3 summarize method
        """
        predictor_1 = self.predictor_3
        predictor_1.mask_params = {"features_to_hide": None,
                                    "threshold": None,
                                    "positive": None,
                                    "max_contrib": 1
                                   }
        predictor_1.add_input(x=self.df_3[["x1", "x2"]], ypred=pd.DataFrame(self.df_3["y"]), contributions=None)
        output = predictor_1.summarize()

        expected_output = pd.DataFrame({
            "y": [1, 1, 0, 0, 0],
            "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209],
            "feature_1": ["weight", "weight", "weight", "weight", "weight"],
            "value_1": ["90", "78", "84", "85", "53"],
            "contribution_1": ["0.0942857", "-0.0235714", "-0.0235714", "-0.0235714", "-0.0235714"],
            "feature_2": ["age", "age", "age", "age", "age"],
            "value_2": ["25", "39", "50", "43", "67"],
            "contribution_2": ["0", "0", "0", "0", "0"]
        }, dtype=object)
        expected_output["y"] = expected_output["y"].astype(int)
        expected_output["proba"] = expected_output["proba"].astype(float)

        feature_expected = [column for column in expected_output.columns if column.startswith("feature_")]
        feature_output = [column for column in output.columns if column.startswith("feature_")]

        value_expected = [column for column in expected_output.columns if column.startswith("value_")]
        value_output = [column for column in output.columns if column.startswith("value_")]

        contribution_expected = [column for column in expected_output.columns if column.startswith("contribution_")]
        contribution_output = [column for column in output.columns if column.startswith("contribution_")]

        assert not expected_output.shape == output.shape
        assert not len(feature_expected) == len(feature_output)
        assert not len(value_expected) == len(value_output)
        assert not len(contribution_expected) == len(contribution_output)
        assert not len(output.columns) == len(expected_output.columns)

        predictor_1.mask_params = {"features_to_hide": None,
                                    "threshold": None,
                                    "positive": None,
                                    "max_contrib": None}

    def test_modfiy_mask(self):
        """
        Unit test modify_mask method
        """
        predictor_1 = self.predictor_2

        assert all([value is None for value in predictor_1.mask_params.values()])

        predictor_1.modify_mask(max_contrib=1)

        assert not all([value is None for value in predictor_1.mask_params.values()])
        assert predictor_1.mask_params["max_contrib"] == 1
        assert predictor_1.mask_params["positive"] == None

        predictor_1.modify_mask(max_contrib=2)

    def test_apply_postprocessing_1(self):
        """
        Unit test apply_postprocessing 1
        """
        predictor_1 = self.predictor_3
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = pd.DataFrame(
            [[1, 2],
            [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        assert np.array_equal(predictor_1.data["x"], predictor_1.apply_postprocessing())

    def test_apply_postprocessing_2(self):
        """
        Unit test apply_postprocessing 2
        """

        postprocessing = {'x1': {'type': 'suffix', 'rule': ' t'},
                          'x2': {'type': 'prefix', 'rule': 'test'}}

        predictor_1 = self.predictor_3
        predictor_1.postprocessing = postprocessing

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['x1', 'x2'],
            index=['Id1', 'Id2']
        )
        expected_output = pd.DataFrame(
            data=[['1 t', 'test2'],
                  ['3 t', 'test4']],
            columns=['x1', 'x2'],
            index=['Id1', 'Id2']
        )
        output = predictor_1.apply_postprocessing()
        assert np.array_equal(output, expected_output)









