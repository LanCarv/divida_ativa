import os
import dotenv
import boto3
from io import BytesIO
import pandas as pd
from datetime import date
import zipfile
from sklearn.preprocessing import StandardScaler
import pickle

import warnings

warnings.filterwarnings("ignore")

print("Processo iniciado")
# Define diretorios
rootPath = os.path.dirname(os.getcwd())
dataPath = os.path.join(rootPath, 'data')
modelsPath = os.path.join(rootPath, 'models')
env = os.path.join(rootPath, '.env')
dotenv.load_dotenv(dotenv_path=env)

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.getenv("AWS_ACESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACESS_KEY")
)

print("Carregando dados para previsões")

# Prepara os dados para realizar as previsões

zip_file = os.path.join(dataPath, 'estoque_da3.zip')
z = zipfile.ZipFile(zip_file)


def ler_bases_exportadas(nome_arquivo):
    z.extract(nome_arquivo)
    df = pd.read_csv(nome_arquivo, sep=',')
    os.remove(nome_arquivo)
    return df


base_conjunta = ler_bases_exportadas('estoque-divida.csv')
base_parcelas = ler_bases_exportadas('parcelas.csv')

print("Inicia transformação das variáveis sobre a dívida")
print("Processamento das variáveis sobre a dívida")

# Gera as variáveis de tempo
base_conjunta['data_divida'] = pd.to_datetime(base_conjunta['inscricao_divida'], infer_datetime_format = True)
base_conjunta['ano_inscricao_da'] = base_conjunta['data_divida'].dt.year

# Seleciona dados sobre a divida
dados_divida = base_conjunta[
    ['cda', 'id_contribuinte', 'da_aberto', 'valor_tot', 'vlr_tributo', 'vlr_taxa', 'da_paga', 'arrecadacao_divida',
     'refis', 'ajuizamento_divida', 'atividade_principal', 'tipo_divida', 'ano_inscricao_da', 'protesto',
     'ajuizamento', 'base']]
dados_divida.dropna(subset=['id_contribuinte'], inplace=True)
dados_divida['id_contribuinte'] = dados_divida['id_contribuinte'].astype(str)  # persistindo tipo de dados

# Calcula a idade da dívida ativa
dados_divida['ano_atual'] = date.today().year
dados_divida['anos_idade_da'] = dados_divida['ano_atual'] - dados_divida['ano_inscricao_da']
dados_divida = dados_divida.drop(columns=['ano_atual'])

# Renomeia colunas para nome mais adequados e filtra dataframe
colunas_nome = {
    'valor_tot': 'valor_total_da',
}
df_divida_ativa = dados_divida.rename(columns=colunas_nome)

# Obtem os dados de parcelamento das dívidas ativas

base_parcelas.loc[base_parcelas['id_imovel'].isnull(), 'tipo_divida'] = 'mercantil'
base_parcelas['tipo_divida'] = base_parcelas['tipo_divida'].fillna('imovel')
base_parcelas['cda'] = base_parcelas['tipo_divida'] + '-' + base_parcelas['cda']

df_parcelas = base_parcelas[['cda', 'quantidade_reparcelamentos']]
df_parcelas.drop_duplicates(subset='cda', inplace=True)
df_parcelas.dropna(subset='cda', inplace=True)

df_divida_ativa = pd.merge(
    left=df_divida_ativa,
    left_on='cda',
    right=df_parcelas,
    right_on='cda',
    how='left'
)

print("Processamento das variáveis sobre o contribuinte")


# Cria conexão com o banco e prepara os dados

def read_s3_files(bucket_name, folder_name, file_name):
    file_key_aws = folder_name + file_name
    obj = s3.Bucket(bucket_name).Object(file_key_aws).get()
    df = pd.read_csv(obj['Body'], sep=';')
    return df


dados_contribuinte = read_s3_files(bucket_name=os.getenv("S3_BUCKET_NAME"), folder_name=os.getenv("S3_FOLDER_NAME"),
                                   file_name='feature_store_contribuinte_2.csv')

dados_totais = pd.merge(
    left=df_divida_ativa,
    right=dados_contribuinte,
    left_on='id_contribuinte',
    right_on='id_pessoa',
    how='left')

df = dados_totais[
    ['cda', 'id_contribuinte', 'da_aberto', 'da_paga', 'arrecadacao_divida', 'tipo_divida', 'valor_total_da',
     'vlr_tributo', 'vlr_taxa', 'ano_inscricao_da', 'protesto', 'ajuizamento', 'refis', 'ajuizamento_divida', 'cpf_cnpj_existe',
    'atividade_principal', 'base', 'anos_idade_da', 'quantidade_reparcelamentos', 'qtd_notas_2anos',
     'situacao', 'status_situacao', 'frequencia_da_pessoa', 'historico_pagamento_em_qtd','historico_pagamento_em_valor',
     'class_contribuinte', 'class_contribuinte_nome', 'class_contribuinte_peso']]

# Ajustando pesos dos contribuintes, foi dado ao primeira dívida o mesmo peso do bom pagador, isso procede?
df.loc[df['frequencia_da_pessoa'] <= 1, 'class_contribuinte_nome'] == 'PRIMEIRA DIVIDA'
df.loc[df['class_contribuinte_nome'] == 'PRIMEIRA DIVIDA', 'class_contribuinte_peso'] == 4.12324

df['class_contribuinte_nome'] = df['class_contribuinte_nome'].fillna('PIOR PAGADOR')
df.loc[df['class_contribuinte_nome'] == 'PIOR PAGADOR', 'class_contribuinte_peso'] == -1.57627
df.fillna(0, inplace=True)

print("Preparação do pipeline de previsões")
# Filtra os dados que precisamos para previsão
df_feature_store = df[['valor_total_da', 'anos_idade_da', 'quantidade_reparcelamentos',
                       'frequencia_da_pessoa', 'historico_pagamento_em_qtd', 'status_situacao',
                       'historico_pagamento_em_valor', 'class_contribuinte_peso']]

# Normalização desses dados
normalizador = StandardScaler()
normalizador.fit(df_feature_store)
dados_normalizados = normalizador.fit_transform(df_feature_store)

colunas = list(normalizador.get_feature_names_out())
df_normalizado = pd.DataFrame(dados_normalizados, columns=colunas)

print("Carrega o modelo e realiza a previsão do Índice Geral de Recuperação (IGR)")


def abre_modelo(nome_modelo, path_modelo, zip_name=None):
    if zip_name:
        zip_file = os.path.join(path_modelo, zip_name)
        z = zipfile.ZipFile(zip_file)
        z.extract(nome_modelo)
    else:
        nome_modelo = os.path.join(path_modelo, nome_modelo)

    modelo = pickle.load(open(nome_modelo, 'rb'))
    return modelo


model_predict_igr = abre_modelo("modeloDA-igr-divida-v2.pkl", modelsPath)

# Realizando previsões
previsoes = model_predict_igr.predict(df_normalizado)
df['igr'] = previsoes
df.loc[df['status_situacao'] == 0, 'igr'] = 0
df.loc[df['anos_idade_da'] >= 15, 'igr'] = 0

print("Início do processo de classificação do rating da dívida parametrizando junto a classificação do contribuinte")

def make_rating_divida(dataframe):
    dataframe.loc[dataframe['igr'] == 0, 'rating_divida'] = 'BAIXISSIMA'

    # Melhor Pagador
    dataframe.loc[(dataframe['rating_divida'].isnull()) & (dataframe['class_contribuinte'] == 2), 'rating_divida'] = 'ALTISSIMA'

    # Pior Pagador
    dataframe.loc[(dataframe['class_contribuinte'] == 0) & (dataframe['igr'] != 0), 'rating_divida'] = 'BAIXA'

    # dataframe.loc[(dataframe['class_contribuinte'] == 3) & (dataframe['igr'] >= 0.5), 'rating_divida'] = 'ALTISSIMA'
    # dataframe.loc[(dataframe['class_contribuinte'] == 3) & (dataframe['igr'] < 0.5) & (dataframe['igr'] >= 0.1), 'rating_divida'] = 'ALTA'
    # dataframe.loc[(dataframe['class_contribuinte'] == 3) & (dataframe['igr'] < 0.1) & (dataframe['igr'] != 0), 'rating_divida'] = 'MEDIA'

    # Pagador intermediario
    dataframe.loc[(dataframe['class_contribuinte'] == 3) & (dataframe['igr'] >= 0.5), 'rating_divida'] = 'ALTA'
    dataframe.loc[(dataframe['class_contribuinte'] == 3) & (dataframe['igr'] < 0.5) & (dataframe['igr'] >= 0.05), 'rating_divida'] = 'MEDIA'
    dataframe.loc[(dataframe['class_contribuinte'] == 3) & (dataframe['igr'] < 0.1) & (dataframe['igr'] != 0), 'rating_divida'] = 'BAIXA'

    # Bom pagador
    dataframe.loc[(dataframe['class_contribuinte'] == 1) & (dataframe['igr'] >= 0.5), 'rating_divida'] = 'ALTISSIMA'
    dataframe.loc[(dataframe['class_contribuinte'] == 1) & (dataframe['igr'] < 0.5) & (dataframe['igr'] >= 0.05), 'rating_divida'] = 'ALTA'
    dataframe.loc[(dataframe['class_contribuinte'] == 1) & (dataframe['igr'] < 0.05) & (dataframe['igr'] != 0), 'rating_divida'] = 'MEDIA'

    # Melhor Pagador
    dataframe.loc[(dataframe['class_contribuinte'] == 4) & (dataframe['igr'] >= 0.3), 'rating_divida'] = 'ALTA'
    dataframe.loc[(dataframe['class_contribuinte'] == 4) & (dataframe['igr'] < 0.3) & dataframe['igr'] >= 0.1, 'rating_divida'] = 'MEDIA'
    dataframe.loc[(dataframe['class_contribuinte'] == 4) & (dataframe['igr'] < 0.1) & (dataframe['igr'] != 0), 'rating_divida'] = 'BAIXA'
    return dataframe


df = make_rating_divida(df)

print("Inicia a conexão com S3 para inscrição dos dados com as previsões")


# Cria conexão ao s3 e preenche a tabela com os dados

def up_s3_files(dataframe, bucket_name, folder_name, file_name):
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, sep=';', index=False)
    file_key_aws = folder_name + file_name
    s3.Object(bucket_name, file_key_aws).put(Body=csv_buffer.getvalue())


up_s3_files(dataframe=df,
            bucket_name=os.getenv("S3_BUCKET_NAME"),
            folder_name=os.getenv("S3_FOLDER_NAME"),
            file_name='previsoes_igr_status_situacao.csv')

print("Upload de dados efetuados no s3")
print("Processo finalizado")
print("Arquivo disponível para download e análise")
