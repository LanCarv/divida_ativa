import tabula
import os


area = [10, 10, 800, 600]
arquivo_pdf = r"C:\Users\Consultor\Downloads\divida_pdf_1.pdf"

tabelas = tabula.read_pdf(arquivo_pdf, pages='all', multiple_tables=True, area=area)

diretorio_saida = r'C:\Users\Consultor\Documents\bases_pesquisa\verificacao\skus'

if not os.path.exists(diretorio_saida):
    os.makedirs(diretorio_saida)

for i, tabela in enumerate(tabelas):
    nome_arquivo = os.path.join(diretorio_saida, f'tbl_{i + 1}.xlsx')
    tabela.to_excel(nome_arquivo, index=False)
