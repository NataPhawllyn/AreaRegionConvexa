from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

mlp_model = joblib.load("modelo_mlp.pkl")
print("Modelo cargado correctamente")

df = pd.read_csv("poligonos_data.csv")

#n=2 # 10 menos el  numero de vertices  
tabla_vertices=np.zeros((8,3))
tabla_vertices[:,0]=np.linspace(10,3,8) 

for n in range(1,8):
    
    df_numberVer=df.loc[((df==0).sum(axis=1))==n] 
    print(f" para {10-n} hay {len(df_numberVer)}")
    X_test = df_numberVer.drop(columns=["area"])  
    y_test = df_numberVer["area"] 

    y_pred = mlp_model.predict(X_test)

    tabla_vertices[n,1] = mean_squared_error(y_test, y_pred)
    error_absoluto_relativo = np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))
    tabla_vertices[n,2] = 100 - np.mean(error_absoluto_relativo) * 100

print(tabla_vertices)


# fig, ax1 = plt.subplots()
# ax1.plot(tabla_verices[:,0], tabla_verices[:,1], color="b", linestyle="-")
# ax1.set_xlabel("CP's number of vertices")
# ax1.set_ylabel("MSE", color="b")
# ax1.tick_params(axis='y', labelcolor="b")  

# # Crear un segundo eje Y que comparte el mismo eje X
# ax2 = ax1.twinx()
# ax2.plot(tabla_verices[:,0], tabla_verices[:,2], color="r", linestyle="--")
# ax2.set_ylabel("Accurancy", color="r")
# ax2.tick_params(axis='y', labelcolor="r")  

# plt.title("")
# fig.tight_layout() 
# plt.show()


     
    

