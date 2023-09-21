import os
import dotenv
import zipfile
import pandas as pd
import boto3
from io import BytesIO
import pickle

import warnings
warnings.filterwarnings("ignore")

rootPath = os.path.dirname(os.getcwd())
dataPath = os.path.join(rootPath, 'data')
modelsPath = os.path.join(rootPath, 'models')
env = os.path.join(rootPath, '.env')
dotenv.load_dotenv(dotenv_path=env)

print("Iniciando carregamento dos dados")
zip_file = os.path.join(dataPath, 'base_treino.zip')
z = zipfile.ZipFile(zip_file)

def ler_bases_exportadas(nome_arquivo):
    z.extract(nome_arquivo)
    df = pd.read_csv(nome_arquivo, sep=',')
    os.remove(nome_arquivo)
    return df

base_conjunta = ler_bases_exportadas('imovel_mercantil.csv')
# base_imovel = ler_bases_exportadas('imovel.csv')
# base_mercantil = ler_bases_exportadas('mercantil.csv')
base_notas_fiscais = ler_bases_exportadas('emissao_notas.csv')


base_conjunta['data_divida'] = pd.to_datetime(base_conjunta['inscricao_divida'], infer_datetime_format = True)
base_conjunta['ano_inscricao_da'] = base_conjunta['data_divida'].dt.year

#base_conjunta = pd.concat([base_imovel, base_mercantil])
# base_conjunta = base_conjunta.query("cpf_cnpj_existe == 1")
#base_conjunta

# Agrego os valores de parcelas para evitar as duplicações
# base_parcelas_agg = base_parcelas.groupby(['cda', 'id_pessoa', 'tipo_divida'])[['total_valor_pago']].sum()
# base_parcelas_agg.reset_index(inplace=True)
# base_parcelas_agg.head()

# Realiza o merge das das bases de imovel e mercantil com a base de parcelas trazendo as correspondências da base parcelas que tem na base conjunta
# base_conjunta = base_conjunta.merge(
#     base_parcelas_agg,
#     how='left',
#     on=['cda', 'tipo_divida', 'id_pessoa'],
#     suffixes=('', '_PARC')
# )

# trata os valores nulos alterando para zero para não enviezar a soma
# base_conjunta.loc[ base_conjunta['total_valor_pago'].isna(), 'total_valor_pago'] = 0

print("Gerando variáveis para identificação dos grupos de contribuintes")

dados_pessoas = base_conjunta[['cda', 'id_contribuinte', 'situacao', 'cpf_cnpj_existe', 'edificacao', 'deb_totais', 'deb_pagos', 'valor_tot', 'vlr_pago', 'tipo_divida']]
dados_pessoas['id_contribuinte'] = dados_pessoas['id_contribuinte'].astype(str)  # Convertendo para string

dados_pessoas.dropna(subset=['id_contribuinte'], inplace=True)
dados_pessoas.rename(columns={'id_contribuinte': 'id_pessoa'}, inplace=True)

# Renomeio a coluna criada anteriormente para valor_pago
# dados_pessoas.rename(columns={'valor_pago_vista_parc':'valor_pago'}, inplace=True)

# Calculo que apresenta quantas cdas o contribuinte tem

frequencia_da_pessoa = dados_pessoas.groupby(['id_pessoa'])['cda'].nunique()
total_debitos_pessoa = dados_pessoas.groupby(['id_pessoa'])['deb_totais'].sum()
debitos_pagos_pessoa = dados_pessoas.groupby(['id_pessoa'])['deb_pagos'].sum()
valor_total_pessoa = dados_pessoas.groupby(['id_pessoa'])['valor_tot'].sum()
valor_pago_pessoa = dados_pessoas.groupby(['id_pessoa'])['vlr_pago'].sum()

# Agrega informação da base de notas fiscais
dados_pessoas = pd.merge(
    left=dados_pessoas, left_on='id_pessoa', right=base_notas_fiscais, right_on='id_pessoa', how='left'
)

# Substituindo por zero os valores nulos
# dados_pessoas['edificacao'] = dados_pessoas['edificacao'].fillna(0)
dados_pessoas['qtd_notas_2anos'] = dados_pessoas['qtd_notas_2anos'].fillna(0)

# Cria variável de situação do contribuinte tratando mercantil e imovel em suas respectivas variáveis ( DEBATER COM EQUIPE )


# MERCANTIL
dados_pessoas.loc[(dados_pessoas['tipo_divida'] == 'mercantil' ) & (dados_pessoas['qtd_notas_2anos'] > 0) & (dados_pessoas['situacao'] == 'ATIVO'), 'situacao_ativa'] = 2
dados_pessoas.loc[(dados_pessoas['tipo_divida'] == 'mercantil' ) & (dados_pessoas['qtd_notas_2anos'] > 0) & (dados_pessoas['situacao'] != 'ATIVO'), 'situacao_ativa'] = 1
dados_pessoas.loc[(dados_pessoas['tipo_divida'] == 'mercantil' ) & (dados_pessoas['qtd_notas_2anos'] == 0) & (dados_pessoas['situacao'] == 'ATIVO'), 'situacao_ativa'] = 1
dados_pessoas.loc[(dados_pessoas['tipo_divida'] == 'mercantil' ) & (dados_pessoas['qtd_notas_2anos'] == 0) & (dados_pessoas['situacao'] != 'ATIVO'), 'situacao_ativa'] = 0

# IMOVEL
dados_pessoas.loc[(dados_pessoas['tipo_divida'] == 'imovel' ) & (dados_pessoas['edificacao'] == 1), 'situacao_ativa'] = 2
dados_pessoas.loc[(dados_pessoas['tipo_divida'] == 'imovel' ) & (dados_pessoas['edificacao'] == 0), 'situacao_ativa'] = 1
dados_pessoas.loc[(dados_pessoas['tipo_divida'] == 'imovel' ) & (dados_pessoas['edificacao'] == 0) & (dados_pessoas['cpf_cnpj_existe'] == 0), 'situacao_ativa'] = 0

# O QUE É NULO COLOCAMOS PESO 0
dados_pessoas['situacao_ativa'] = dados_pessoas['situacao_ativa'].fillna(0)

# Cria variável com peso sobre a situação real tratando cpf

dados_pessoas['status_situacao'] = dados_pessoas['situacao_ativa'] + dados_pessoas['cpf_cnpj_existe']
dados_pessoas.loc[dados_pessoas['situacao_ativa'] == 0, 'status_situacao'] = 0
# dados_pessoas.loc[(dados_pessoas['cpf_cnpj_existe'] == 0), 'status_situacao'] = 0

# Remove duplicatas dos dados finalizando o dataframe
dados_pessoas.drop_duplicates(subset=['id_pessoa'], inplace=True)
dados_pessoas = dados_pessoas.set_index('id_pessoa')

dados_pessoas['frequencia_da_pessoa'] = frequencia_da_pessoa
dados_pessoas['total_debitos_pessoa'] = total_debitos_pessoa
dados_pessoas['debitos_pagos_pessoa'] = debitos_pagos_pessoa
dados_pessoas['valor_total_pessoa'] = valor_total_pessoa
dados_pessoas['valor_pago_pessoa'] = valor_pago_pessoa

# Faz o calculo do historico de pagamento

dados_pessoas.loc[(dados_pessoas['total_debitos_pessoa'].isna()) | (dados_pessoas['total_debitos_pessoa'] == 0) , 'total_debitos_pessoa'] = 1
dados_pessoas.loc[(dados_pessoas['valor_total_pessoa'].isna()) | (dados_pessoas['valor_total_pessoa'] == 0) , 'valor_total_pessoa'] = 1

dados_pessoas['historico_pagamento_em_qtd'] = dados_pessoas['debitos_pagos_pessoa'] / (dados_pessoas['total_debitos_pessoa'])
dados_pessoas['historico_pagamento_em_valor'] = dados_pessoas['valor_pago_pessoa'] / (dados_pessoas['valor_total_pessoa'])

print("Realizando a classificação do contribuinte")

# fiz alterações de quando ele chama o modelo gerado no notebook
def abre_modelo(nome_modelo, path_modelo, zip_name=None):
    if zip_name:
        zip_file = os.path.join(path_modelo, zip_name)
        z = zipfile.ZipFile(zip_file)
        z.extract(nome_modelo)
    else:
        nome_modelo = os.path.join(path_modelo, nome_modelo)

    modelo = pickle.load(open(nome_modelo, 'rb'))
    return modelo

# Lê o modelo diretamente da pasta 'models'
print("Le o modelo de clusterização salvo do notebook")
model_predict_contribuinte = abre_modelo("classificador-contribuinte-v2.pkl", modelsPath)


matriz_previsao_class = dados_pessoas[['status_situacao', 'historico_pagamento_em_qtd', 'historico_pagamento_em_valor']]
dados_pessoas['class_contribuinte'] = model_predict_contribuinte.predict(matriz_previsao_class)
dados_pessoas.loc[dados_pessoas['frequencia_da_pessoa'] == 1, 'class_contribuinte'] = 4

# Nomeando a classificação com label de prioridade
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 4, 'class_contribuinte_nome'] = 'PRIMEIRA DIVIDA'
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 0, 'class_contribuinte_nome'] = 'PIOR PAGADOR'
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 1, 'class_contribuinte_nome'] = 'PAGADOR INTERMEDIARIO'
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 3, 'class_contribuinte_nome'] = 'BOM PAGADOR'
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 2, 'class_contribuinte_nome'] = 'MELHOR PAGADOR'

# Dando pesos para os tipos de contribuintes
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 4, 'class_contribuinte_peso'] = 1
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 0, 'class_contribuinte_peso'] = -0.98031
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 1, 'class_contribuinte_peso'] = 1
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 3, 'class_contribuinte_peso'] = 2.54487 # esse é o peso do primeira dívida na análise discriminante
dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 2, 'class_contribuinte_peso'] = 4.68032


df_feature_store_contribuinte = dados_pessoas.reset_index()

df_feature_store_contribuinte = df_feature_store_contribuinte[['id_pessoa', 'situacao', 'cpf_cnpj_existe', 'edificacao', 'qtd_notas_2anos',
                                                               'situacao_ativa', 'status_situacao', 
                                                               'deb_totais','deb_pagos', 'valor_tot', 'vlr_pago',
                                                               'frequencia_da_pessoa', 'total_debitos_pessoa', 'debitos_pagos_pessoa', 'valor_total_pessoa', 'valor_pago_pessoa', 
                                                               'historico_pagamento_em_qtd', 'historico_pagamento_em_valor', 
                                                               'class_contribuinte', 'class_contribuinte_nome', 'class_contribuinte_peso']]

print("Inicia a conexão com S3 para inscrição dos dados")

# Cria conexão ao s3
s3_resource = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.getenv("AWS_ACESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACESS_KEY")
    )

# Preenche com o arquivo gerado
def up_s3_files(dataframe, bucket_name, folder_name, file_name):
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, sep=';', index=False)
    file_key_aws = folder_name + file_name
    s3_resource.Object(bucket_name, file_key_aws).put(Body=csv_buffer.getvalue())


# Salva o arquivo em feature_store_contribuinte_2.csv
up_s3_files(dataframe=df_feature_store_contribuinte, 
            bucket_name=os.getenv("S3_BUCKET_NAME"), 
            folder_name=os.getenv("S3_FOLDER_NAME"), 
            file_name='feature_store_contribuinte_2.csv')

print("Dados atualizados e persistidos no bucket S3")
print("Processo finalizado")
