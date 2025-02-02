import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Carregar os dados
data = pd.read_csv('daily-bike-share.csv')

# Verificar as primeiras linhas do DataFrame
print(data.head())

# Separar as variáveis independentes e dependentes
X = data[['temp', 'atemp', 'hum', 'windspeed']]  # Variáveis independentes
y = data['rentals']  # Variável dependente

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Exibir as previsões
print(predictions)