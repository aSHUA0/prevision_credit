from data_processing import processing_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

input_train, input_test, output_train, output_test = processing_data()

def training():
    #Instancias
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=47)
    lg_model = LogisticRegression()
    XGBC_model = XGBClassifier(max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    
    #Treinando o Algoritimo
    rf_model.fit(input_train, output_train)
    XGBC_model.fit(input_train, output_train)
    lg_model.fit(input_train, output_train)

    x = {'rf': rf_model,
         'XGBC': XGBC_model,
         'lg': lg_model}

    return x