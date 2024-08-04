import streamlit as st
import pandas as pd
# scikit-learn is commonly imported as sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dados fictícios com risco do bairro atualizado
data = {
    'ano_fabricacao': [2015, 2018, 2020, 2017, 2016, 2019],
    'modelo': ['Sedan', 'SUV', 'Hatch', 'Sedan', 'SUV', 'Hatch'],
    'valor_seguro': [2000, 2500, 2200, 2400, 2600, 2300],
    'zona_cidade': ['Centro', 'Bairro', 'Subúrbio', 'Centro', 'Subúrbio', 'Bairro'],
    'risco_bairro': [1.1, 1.4, 1.8, 1.2, 1.7, 1.5]
}


df = pd.DataFrame(data)

# Dividir os dados em recursos (X) e alvo (y)
X = df[['ano_fabricacao', 'modelo', 'zona_cidade', 'risco_bairro']]
y = df['valor_seguro']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformador para variáveis categóricas
categorical_features = ['modelo', 'zona_cidade']
categorical_transformer = OneHotEncoder()

# Pipeline de pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

# Pipeline do modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Treinar o modelo
model.fit(X_train, y_train)

# Título da página
st.title('Previsão de Seguro de Carro')
#st.write('Previsão de Seguro de Carro')
# Legenda para risco do bairro
st.markdown("""
  Legenda para Risco do Bairro:
- **Centro**: Risco baixo (1.0 a 1.2)
- **Bairro**: Risco médio (1.3 a 1.5)
- **Subúrbio**: Risco alto (1.6 a 2.0)
""")

# Entradas do usuário
ano_fabricacao = st.number_input('Ano de Fabricação', min_value=1990, max_value=2024, value=2020)
modelo = st.selectbox('Modelo', ['Sedan', 'SUV', 'Hatch'])
zona_cidade = st.selectbox('Zona da Cidade', ['Centro', 'Bairro', 'Subúrbio'])
risco_bairro = st.slider('Risco do Bairro', min_value=1.0, max_value=2.0, value=1.5, step=0.1)

# Prever o valor do seguro
if st.button('Calcular Seguro'):
    input_data = pd.DataFrame({
        'ano_fabricacao': [ano_fabricacao],
        'modelo': [modelo],
        'zona_cidade': [zona_cidade],
        'risco_bairro': [risco_bairro]
    })
   
    previsao = model.predict(input_data)[0]
    st.write(f'O valor previsto do seguro é: R$ {previsao:.2f}')
