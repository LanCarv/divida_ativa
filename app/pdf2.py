import os
import camelot

def extrair_tabelas_pdf(caminho_pdf):
    tabelas = camelot.read_pdf(caminho_pdf, flavor='stream', pages='all')
    return tabelas

def salvar_tabelas_como_csv(tabelas, diretorio_saida):
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)

    for i, tabela in enumerate(tabelas):
        nome_arquivo = os.path.join(diretorio_saida, f'tbl_{i + 1}.csv')
        tabela.to_csv(nome_arquivo, index=False)

# Caminho do arquivo PDF
arquivo_pdf = r"C:\Users\Consultor\Downloads\divida_pdf_1.pdf"

# Diretório de saída
diretorio_saida = r'C:\Users\Consultor\Documents\bases_pesquisa\verificacao\skus'

# Extrair tabelas do PDF usando camelot
tabelas_extrair = extrair_tabelas_pdf(arquivo_pdf)

# Salvar as tabelas como CSV
salvar_tabelas_como_csv(tabelas_extrair, diretorio_saida)
