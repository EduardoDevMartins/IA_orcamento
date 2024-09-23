import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicializar o Flask
app = Flask(__name__)

# Carregar os dados
tabela = pd.read_excel('tabela_precos.xlsx', sheet_name='Hidráulica')
tabela.columns = tabela.columns.str.strip()  # Remove espaços em branco dos nomes das colunas

# Remover linhas com valores NaN na coluna de serviços (C)
tabela = tabela[tabela.iloc[:, 2].notna()]

# Função para encontrar o serviço mais semelhante
def encontrar_servico_mais_semelhante(servico_solicitado):
    # Usar TF-IDF para calcular similaridade
    vectorizer = TfidfVectorizer()
    
    # Ajustar e transformar a coluna de serviços
    tfidf_matrix = vectorizer.fit_transform(tabela.iloc[:, 2])  # Coluna C
    
    # Adicionar o serviço solicitado à matriz
    servico_solicitado_tfidf = vectorizer.transform([servico_solicitado])
    
    # Calcular similaridade
    similaridade = cosine_similarity(servico_solicitado_tfidf, tfidf_matrix)
    
    # Encontrar o índice do serviço mais semelhante
    indice_servico_mais_semelhante = similaridade.argmax()
    
    # Retornar o preço correspondente (supondo que M seja o índice 12)
    return tabela.iloc[indice_servico_mais_semelhante, 12]  # Coluna M

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        servico = request.form['servico']
        
        # Encontrar o preço do serviço mais semelhante
        preco = encontrar_servico_mais_semelhante(servico)
        
        return render_template('index.html', preco=preco)

    return render_template('index.html', preco=None)

if __name__ == '__main__':
    app.run(debug=True)
