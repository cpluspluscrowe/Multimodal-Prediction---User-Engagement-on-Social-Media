from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def ApplyMinMaxScalar(data):
    data = pd.DataFrame(data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data.flatten()


if __name__ == '__main__':
    data = [1,30,50,4,7,4]
    data = ApplyMinMaxScalar(data)
    print(data)

