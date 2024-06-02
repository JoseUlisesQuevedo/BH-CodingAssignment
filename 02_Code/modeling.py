import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,roc_auc_score
import sklearn
from xgboost import XGBClassifier

import optuna


def objective_xgb(trial):

    params = {
        "objective": "binary:logistic",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "reg_alpha":trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    model = XGBClassifier(**params)
    model.fit(training_data,training_labels)

    predictions = model.predict(validation_data)
    roc_auc = roc_auc_score(validation_labels,predictions)

    return roc_auc

def objective_lr(trial):

    params={
        "penalty": trial.suggest_categorical("penalty",[None,"l2"]),
        "C": trial.suggest_float("C",1e-3,5.0,log=True),
    }

    model = LogisticRegression(**params)
    model.fit(training_data.drop("Length",axis=1),training_labels.values.ravel())

    predictions = model.predict(validation_data.drop("Length",axis=1)) 
    roc_auc  = roc_auc_score(validation_labels,predictions)

    return roc_auc

if __name__ == "__main__":

    ##Data reading

    #Used to train the model
    training_data = pd.read_csv("01_Data/Processed/Training/training_set.csv")
    training_labels = pd.read_csv("01_Data/Processed/Training/training_labels.csv")

    #Used for hyperparameter tuning
    validation_data = pd.read_csv("01_Data/Processed/Validation/validation_set.csv")
    validation_labels = pd.read_csv("01_Data/Processed/Validation/validation_labels.csv")

    #Used for final model evaluation 
    testing_labels = pd.read_csv("01_Data/Processed/Testing/testing_labels.csv")
    testing_data = pd.read_csv("01_Data/Processed/Testing/testing_set.csv")

    #Model definition: We test 2 models, a vanilla logistic regression and a XGBoost model
    ## We tune both models using optuna for the same number of iterations and then compare results


    lr_study = optuna.create_study(direction="maximize")
    xgb_study = optuna.create_study(direction="maximize")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    xgb_study.optimize(objective_xgb, n_trials=100)
    lr_study.optimize(objective_lr, n_trials=100)


    ##Results comparison
    best_xgb = xgb_study.best_params
    best_lr = lr_study.best_params

    xgb_model = XGBClassifier(**best_xgb)
    lr_model = LogisticRegression(**best_lr)

    xgb_model.fit(training_data,training_labels)
    lr_model.fit(training_data.drop("Length",axis=1),training_labels.values.ravel())

    xgb_predictions = xgb_model.predict(testing_data)
    lr_predictions = lr_model.predict(testing_data.drop("Length",axis=1))

    print("XGBoost Model")  

    print(classification_report(testing_labels,xgb_predictions))

    print("Logistic Regression Model")  

    print(classification_report(testing_labels,lr_predictions))
          


