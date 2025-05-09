import os
from model_training import training
from data_processing import processing_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

input_train, input_test, output_train, output_test = processing_data()
numeros_permitidos = ["1", "2", "3", "4"]
models = training()

while True:
    resp = input('Qual modelo deseja testar?\n [1]Random Forest\n [2]XGBoost \n [3]RegressionLogistic \n [4]Sair  ')
    if len(resp) > 1:
        print('Opção invalida, tente novamente')
        continue
    if resp not in numeros_permitidos:
        print('Opção invalida, tente novamente')
        continue

    if resp == "1":
        os.system('cls')
        
        rf_model = models['rf']

        rf_predict = rf_model.predict(input_test)

        print("Acurácia:", accuracy_score(output_test, rf_predict))
        print("Matriz de confusão:\n", confusion_matrix(output_test, rf_predict))
        print("Relatório de classificação:\n", classification_report(output_test, rf_predict))

    elif resp == "2":
        os.system('cls')
    
        XGBC_model = models['XGBC']

        XGBC_predict = XGBC_model.predict(input_test)

        print("Acurácia:", accuracy_score(output_test, XGBC_predict))
        print("Matriz de confusão:\n", confusion_matrix(output_test, XGBC_predict))
        print("Relatório de classificação:\n", classification_report(output_test, XGBC_predict))
    
    elif resp == "3":
        os.system('cls')

        lg_model = models['lg']

        lg_predict = lg_model.predict(input_test)

        print("Acurácia:", accuracy_score(output_test, lg_predict))
        print("Matriz de confusão:\n", confusion_matrix(output_test, lg_predict))
        print("Relatório de classificação:\n", classification_report(output_test, lg_predict))

    else: 
        os.system('cls')
        break