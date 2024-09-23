import pandas as pd
from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicializar o Flask
app = Flask(__name__)

# Carregar os dados
tabela = pd.read_excel('tabela_precos.xlsx', sheet_name='Hidráulica')
tabela.columns = tabela.columns.str.strip()  # Remove espaços em branco dos nomes das colunas

# Remover linhas com valores NaN na coluna de serviços (C)
tabela = tabela[tabela.iloc[:, 2].notna()]

# Concatenar as colunas C, D, E e F para criar uma representação completa de cada serviço
tabela['descricao_completa'] = tabela.iloc[:, 2].astype(str) + ' ' + tabela.iloc[:, 3].astype(str) + ' ' + tabela.iloc[:, 4].astype(str) + ' ' + tabela.iloc[:, 5].astype(str)

# Função para encontrar o serviço mais semelhante
def encontrar_servico_mais_semelhante(servico_solicitado, caracteristicas):
    # Usar TF-IDF para calcular similaridade
    vectorizer = TfidfVectorizer()

    # Concatenar as características fornecidas pelo usuário
    if caracteristicas:
        servico_solicitado = f"{servico_solicitado} {' '.join(caracteristicas)}"
    
    # Ajustar e transformar a nova coluna de descrição completa (C + D + E + F)
    tfidf_matrix = vectorizer.fit_transform(tabela['descricao_completa'])
    
    # Adicionar o serviço solicitado à matriz
    servico_solicitado_tfidf = vectorizer.transform([servico_solicitado])
    
    # Calcular similaridade
    similaridade = cosine_similarity(servico_solicitado_tfidf, tfidf_matrix)
    
    # Encontrar o índice do serviço mais semelhante
    indice_servico_mais_semelhante = similaridade.argmax()
    
    # Retornar o preço total, mão de obra, material, diagnóstico e solução
    preco_total = tabela.iloc[indice_servico_mais_semelhante, 12]  # Coluna M
    preco_mao_de_obra = tabela.iloc[indice_servico_mais_semelhante, 10]  # Coluna K
    preco_material = tabela.iloc[indice_servico_mais_semelhante, 11]  # Coluna L
    diagnostico = tabela.iloc[indice_servico_mais_semelhante, 13]  # Coluna N
    solucao = tabela.iloc[indice_servico_mais_semelhante, 14]  # Coluna O
    
    return preco_total, preco_mao_de_obra, preco_material, diagnostico, solucao

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        servico = request.form['servico']
        caracteristicas = [
            request.form.get('caracteristica1', ''),
            request.form.get('caracteristica2', ''),
            request.form.get('caracteristica3', '')
        ]
        
        # Encontrar os preços do serviço mais semelhante
        preco_total, preco_mao_de_obra, preco_material, diagnostico, solucao = encontrar_servico_mais_semelhante(servico, caracteristicas)
        
        return render_template('index.html', preco_total=preco_total, preco_mao_de_obra=preco_mao_de_obra, preco_material=preco_material, diagnostico=diagnostico, solucao=solucao)

    return render_template('index.html', preco_total=None, preco_mao_de_obra=None, preco_material=None, diagnostico=None, solucao=None)

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_text = request.form['feedback']
    expectativa_text = request.form['expectativa']
    
    # Aqui você pode processar o feedback e armazenar a resposta correta em uma nova planilha ou arquivo
    with open('feedback.txt', 'a') as f:
        f.write(f"Feedback: {feedback_text}\nExpectativa: {expectativa_text}\n\n")
    
    # Aqui você pode armazenar a expectativa no mesmo formato que as descrições completas
    # Vamos imaginar que 'expectativa_text' tenha o mesmo formato que um serviço completo (C + D + E + F)
    with open('dados_corrigidos.csv', 'a') as f:
        f.write(f"{expectativa_text}\n")  # Armazene a resposta correta
    
    # Você pode então reprocessar esse arquivo na próxima vez que o modelo for executado
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
