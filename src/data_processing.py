import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

caminho = os.path.join("..", "projeto", "data", 'UCC.csv')
credit = pd.read_csv(caminho)

def processing_data():
    input = credit.drop(columns='default.payment.next.month')
    output = credit['default.payment.next.month']

    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size = 0.2, random_state=47)

    #Escalonando os dados
    scaler = StandardScaler()
    input_train_scaled = scaler.fit_transform(input_train)

    input_test_scaled = scaler.transform(input_test)

    x = [input_train_scaled, input_test_scaled, output_train, output_test]

    return x