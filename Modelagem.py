# %%
import pandas as pd

from sklearn import metrics
from sklearn import model_selection

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.preprocessing import OneHotEncoder

import scikitplot as skplt
# %%
df = pd.read_csv("./data/teste_indicium_precificacao_clean.csv")
df


# %%
# Separando features e target
X = df.drop(columns=['id','nome','bairro', 'price', 'ultima_review'])
y = df['price']

# %%
#Pre-processamento dos dados
# %%
one_hot_encoder = OneHotEncoder(sparse_output=False)
room_type_encoded = pd.DataFrame(one_hot_encoder.fit_transform(X[['room_type']]), columns=one_hot_encoder.get_feature_names_out(['room_type']))
bairro_encoded = pd.DataFrame(one_hot_encoder.fit_transform(X[['bairro_group']]), columns=one_hot_encoder.get_feature_names_out(['bairro_group']))

# %%

X_final = pd.concat([X.drop(columns=['room_type','bairro_group']), room_type_encoded, bairro_encoded], axis=1)
# %%

#separando as varáveis entre treino e teste

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_final,
                                                                    y,
                                                                    test_size=0.2,
                                                                    random_state=42)

# %%
#Treinando diferentes modelos
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Support Vector Machines": SVR(),
    }

medidas_performance = pd.DataFrame(columns=["modelo", "MAE", "MSE", "RMSE", "R2"])

modelo_performance = []

for i, item in enumerate(models):
    model = models[item]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #MAE
    mae = metrics.mean_absolute_error(y_test, y_pred)

    #MSE
    mse = metrics.mean_squared_error(y_test, y_pred)

    #RMSE
    rmse = metrics.root_mean_squared_error(y_test, y_pred)

    #R²
    r2 = metrics.r2_score(y_test, y_pred)

    modelo_performance = [item,  mae, mse, rmse, r2]

    medidas_performance.loc[i] = modelo_performance

# 
# %%
medidas_performance
# %%
X_final
# %%
