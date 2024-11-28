import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analisar', methods=['POST'])
def analisar():
    data = pd.read_csv('data/entregas.csv')
    data.dropna(inplace=True)

    # Codificação One-Hot
    data = pd.get_dummies(data, columns=['localidade'])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tamanho', y='preco', data=data)
    plt.savefig('static/scatterplot.png')
    
    X = data.drop(columns=['preco'])
    y = data['preco']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return render_template('index.html', mse=mse, image='static/scatterplot.png')

@app.route('/filtrar', methods=['POST'])
def filtrar():
    localidade = request.form.get('localidade')
    tamanho = request.form.get('tamanho')
    
    data = pd.read_csv('data/entregas.csv')
    data.dropna(inplace=True)
    
    if localidade:
        data = data[data['localidade'] == localidade]
    
    if tamanho:
        data = data[data['tamanho'] == int(tamanho)]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tamanho', y='preco', data=data)
    plt.savefig('static/scatterplot_filtrado.png')
    
    return render_template('index.html', image_filtrado='static/scatterplot_filtrado.png')

@app.route('/prever', methods=['POST'])
def prever():
    tamanho = float(request.form.get('tamanho'))
    localidade = request.form.get('localidade')
    
    data = pd.read_csv('data/entregas.csv')
    data.dropna(inplace=True)
    
    X = data[['tamanho', 'localidade']]
    y = data['preco']
    X = pd.get_dummies(X, columns=['localidade'])
    
    model = LinearRegression()
    model.fit(X, y)
    
    X_novo = pd.DataFrame({'tamanho': [tamanho], 'localidade': [localidade]})
    X_novo = pd.get_dummies(X_novo, columns=['localidade'])
    
    # Alinhar as colunas para corresponder ao treino
    X_novo = X_novo.reindex(columns=model.coef_.shape[0], fill_value=0)
    
    previsao = model.predict(X_novo)[0]
    
    return render_template('index.html', previsao=previsao)

@app.route('/visualizacao_avancada', methods=['POST'])
def visualizacao_avancada():
    data = pd.read_csv('data/entregas.csv')
    data.dropna(inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='localidade', y='preco', data=data)
    plt.savefig('static/barplot.png')
    
    plt.figure(figsize=(10, 6))
    data['localidade'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.savefig('static/piechart.png')
    
    return render_template('index.html', barplot='static/barplot.png', piechart='static/piechart.png')

@app.route('/exportar', methods=['POST'])
def exportar():
    data = pd.read_csv('data/entregas.csv')
    data.dropna(inplace=True)
    
    data.to_csv('data/entregas_exportadas.csv', index=False)
    
    return render_template('index.html', message='Dados exportados com sucesso!')

if __name__ == '__main__':
    app.run(debug=True)
