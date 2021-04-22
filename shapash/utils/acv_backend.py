try:
    from acv_explainers import ACVTree
    from acv_explainers.utils import get_null_coalition
    is_acv_available = True
except ImportError:
    is_acv_available = False

import pandas as pd
import numpy as np
from shapash.utils.model_synoptic import simple_tree_model, catboost_model
from shapash.utils.transform import check_transformers, preprocessing_tolist


def active_shapley_values(model, x_df, explainer=None, preprocessing=None):
    """
    Compute the active Shapley values using ACV package.

    Parameters
    ----------
    model: model object from sklearn, catboost, xgboost or lightgbm library
        this model is used to choose a shap explainer and to compute
        active shapley values
    x_df: pd.DataFrame
    explainer : explainer object from shap, optional (default: None)
        this explainer is used to compute shapley values


    Returns
    -------
    np.array or list of np.array

    """
    if is_acv_available is False:
        raise ValueError(
            """
            Active Shapley values requires the ACV packag,
            which can be installed following instructions here :
                https://github.com/salimamoukou/acv00
            """
                    )

    if explainer is None:
        if str(type(model)) in simple_tree_model or str(type(model)) in catboost_model:
            explainer = ACVTree(model=model, data=x_df.values)
            print("Backend: ACV")
        else:
            raise NotImplementedError(
                """
                Model not supported for ACV backend.
                """
            )

    #  Get all dummies variables that should be grouped together
    if preprocessing:
        list_encoding = preprocessing_tolist(preprocessing)
        use_ct, use_ce = check_transformers(list_encoding)
        if use_ct:  # Case column transformers sklearn
            list_all_one_hot_features = get_list_group_one_hot_features_ct(list_encoding=list_encoding)
        elif use_ce:  # Case category_encoders
            list_all_one_hot_features = get_list_group_one_hot_features_ce(list_encoding=list_encoding)
        else:
            raise NotImplementedError
        c = [[x_df.columns.get_loc(feat) for feat in group_feat] for group_feat in list_all_one_hot_features]
    else:
        c = [[]]

    sdp_importance, sdp_index, size, sdp = explainer.importance_sdp_clf(
        X=x_df,
        data=np.array(x_df.values, dtype=np.double),
        C=c,
        global_proba=0.9,
        num_threads=1
    )
    s_star, n_star = get_null_coalition(sdp_index, size)
    contributions = explainer.shap_values_acv_adap(
        X=x_df,
        C=c,
        N_star=n_star,
        size=size,
        S_star=s_star,
        num_threads=1
    )

    contributions = [pd.DataFrame(contributions[:, :, i], columns=x_df.columns, index=x_df.index)
                     for i in range(contributions.shape[-1])]

    return contributions, sdp_importance, explainer


def get_list_group_one_hot_features_ct(list_encoding):
    if len(list_encoding) > 1:
        raise NotImplementedError("Shapash can only handle one ColumnTransformer.")

    list_one_hot = [list(enc[1].get_feature_names()) for enc in list_encoding[0].transformers_
                    if 'onehotencoder' in str(type(enc[1])).lower()]

    if len(list_one_hot) == 0:
        return [[]]

    list_one_hot_new_col_names = [feat for x in list_one_hot for feat in x]

    list_categories = [enc[1].categories_ for enc in list_encoding[0].transformers_
                       if 'onehotencoder' in str(type(enc[1])).lower()]
    list_categories = [list_cat for x in list_categories for list_cat in x]

    list_list_features_one_hot = []
    for cat_values in list_categories:
        new_features = []
        for cat_name in cat_values:
            assert cat_name in list_one_hot_new_col_names[0]
            new_features.append(list_one_hot_new_col_names[0])
            del list_one_hot_new_col_names[0]
        list_list_features_one_hot.append(new_features)

    return list_list_features_one_hot


def get_list_group_one_hot_features_ce(list_encoding):
    if isinstance(list_encoding, list):
        list_one_hot = [enc for enc in list_encoding if 'onehotencoder' in str(type(enc)).lower()]
    else:
        list_one_hot = [list_encoding] if 'onehotencoder' in str(type(list_encoding)).lower() else []

    if len(list_one_hot) == 0:
        return [[]]

    list_all_one_hot_cols = []
    for enc in list_one_hot:
        list_all_one_hot_cols.extend(
            [enc.mapping[i]['mapping'].columns.to_list() for i in range(len(enc.mapping))]
        )

    return list_all_one_hot_cols
