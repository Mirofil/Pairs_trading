import shap
import lightgbm as lgb
from simulations_database import generate_scenario



def transform_params_if_needed_row(params_row):
    if "training_delta" in params_row.index:
        params_row["training_delta"] = (
            30 * params_row["training_delta"][0] + params_row["training_delta"][1]
        )
    if "formation_delta" in params_row.index:
        params_row["formation_delta"] = (
            30 * params_row["formation_delta"][0] + params_row["formation_delta"][1]
        )
    
    return params_row


params_blacklist = [
    "method",
    "end",
    "start",
    "redo_prefiltered",
    "redo_preprocessed",
    "truncate",
    "name",
    "data_path",
    "save",
    "freq",
    "jump",
]

example_params = list(generate_scenario().keys())

params_row = [
    param
    for param in example_params
    if param not in params_blacklist
]

df = analysis.dataframe()
df.columns = [col_name.replace('config/', '') for col_name in df.columns]

df['formation_delta'] = df['formation_delta'].apply(lambda x: 30*x[0]+x[1])
df['training_delta'] = df['training_delta'].apply(lambda x: 30*x[0]+x[1])


x = df[params_row]
x= df[['lag', 'threshold']]
y = df["Annualized Sharpe"]

train_data = lgb.Dataset(x, label=y)

lgb_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": {"l2", "l1"},
    "num_leaves": 4,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "min_data_in_bin":1,"min_data":1,"min_hess":0
}

gbm = lgb.train(lgb_params, train_data, num_boost_round=20,)

shap.initjs()

explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(x)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[1,:], x.iloc[1,:])

shap.force_plot(explainer.expected_value, shap_values, x)
shap.dependence_plot("threshold", shap_values, x)


# IMPORTANCES OF ALL FEATURES
shap.summary_plot(shap_values, x, plot_type="bar")
shap.summary_plot(shap_values, x)
