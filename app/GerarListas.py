import os
import dotenv

import boto3
import pandas as pd
import zipfile
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')
import warnings
warnings.filterwarnings("ignore")


rootPath = os.getcwd()
dataPath = os.path.join(rootPath, '../data')
modelsPath = os.path.join(rootPath, '../models')
env = os.path.join(rootPath, '../.env')
dotenv.load_dotenv(dotenv_path=env)

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.getenv("AWS_ACESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACESS_KEY")
    )

def read_s3(bucket_name, folder_name, file_name):
    file_key_aws = folder_name + file_name
    obj = s3.Bucket(bucket_name).Object(file_key_aws).get()
    df = pd.read_csv(obj['Body'], sep=';')
    return df

def formatar_moeda(valor):
    return locale.currency(valor, grouping=True)

print("Carregando dados do feature_store_contribuinte")
dados_contribuinte = read_s3(
    bucket_name=os.getenv("S3_BUCKET_NAME"),
    folder_name=os.getenv("S3_FOLDER_NAME"),
    file_name='feature_store_contribuinte_prime_k5_dez.csv')

print("Carregando dados do feature_store_igr")
dados_feat_store_igr = read_s3(
     bucket_name=os.getenv("S3_BUCKET_NAME"),
     folder_name=os.getenv("S3_FOLDER_NAME"),
     file_name='feature_store_igr_prime_ab1_k5_dez.csv')

print("Carregando dados da predição realizada")
dados_igr = read_s3(
    bucket_name=os.getenv("S3_BUCKET_NAME"),
    folder_name=os.getenv("S3_FOLDER_NAME"),
    file_name='prime_k5_dez.csv')

print("Carregando dados de imovel e mercantil")
zip_file = os.path.join(dataPath, 'base_treino.zip')
z = zipfile.ZipFile(zip_file)

def ler_bases_exportadas(nome_arquivo):
    z.extract(nome_arquivo)
    df = pd.read_csv(nome_arquivo, sep=',')
    os.remove(nome_arquivo)
    return df

base_conjunta = ler_bases_exportadas('imovel_mercantil.csv')


# Filtrando apenas o Estoque:
dados_igr = dados_igr.loc[(dados_igr['da_aberto'] == 1)]

# Colunas/variáveis que exportaremos no csv
col = ['cda', 'id_pessoa', 'atividade_principal', 'idade_divida', 'vlr_tributo', 'vlr_taxa', 'valor_total_da', 'num_dist_cda',
       'quantidade_reparcelamento', 'historico_pagamento_em_valor', 'qtd_notas_2anos', 'edificacao', 'situacao', 'situacao_cobranca',
       'endereco_existe', 'cpf_cnpj_existe', 'class_contribuinte_nome', 'class_contribuinte_perfil', 'igr', 'rating_divida']

# Filtro para manter apenas registros de DAs abertas não ajuizadas
base_conjunta_aux = base_conjunta[['cda', 'tipo_divida', 'id_pessoa', 'da_aberto','atividade_principal', 'situacao',
                                   'tipo_tributo', 'vlr_tributo', 'vlr_taxa', 'edificacao', 'cpf_cnpj_existe',
                                   'protesto', 'ajuizamento', 'refis', 'endereco_existe']]
chave = ['cda', 'tipo_divida', 'id_pessoa', 'da_aberto']


dados_igr3 = pd.merge(dados_igr, base_conjunta_aux, on = chave, how = "left")
dados_contribuinte_aux = dados_contribuinte[['id_pessoa', 'tipo_divida', 'da_aberto', 'qtd_notas_2anos']]

dados_igr3 = pd.merge(dados_igr3, dados_contribuinte_aux, on = ['tipo_divida', 'id_pessoa', 'da_aberto'], how = "left")

# Filtrando para o dataframe desejado:
dados_igr3 = dados_igr3[(dados_igr3['ajuizamento'] == 0)]
dados_igr3 = dados_igr3[(dados_igr3['protesto'] == 0)]
dados_igr3 = dados_igr3[(dados_igr3['idade_divida'] >= 0) & (dados_igr3['idade_divida'] <= 3)]

valores_filtrar = ["MEDIA", "ALTA", "ALTISSIMA"]
dados_igr3 = dados_igr3[dados_igr3['rating_divida'].isin(valores_filtrar)]

# Diferenciando por tipo de dívida
dados_igr_imovel = dados_igr3[dados_igr3['tipo_divida'] == 'imovel']
dados_igr_merc = dados_igr3[dados_igr3['tipo_divida'] == 'mercantil']

# Filtro para Geração das listas IMÓVEL
imovel_ajuizamento = dados_igr_imovel[dados_igr_imovel['valor_total_da'] >= 50000]
imovel_protesto =  dados_igr_imovel[(dados_igr_imovel['valor_total_da'] >= 3000) & (dados_igr_imovel['valor_total_da'] <50000)]
imovel_negativar = dados_igr_imovel[(dados_igr_imovel['valor_total_da'] >= 1000) & (dados_igr_imovel['valor_total_da'] <3000)]

# Dataframe 1
imovel_negativar = imovel_negativar.sort_values(by="igr", ascending=False)
imovel_negativar = imovel_negativar[col]

# Dataframe 2
imovel_protesto = imovel_protesto.sort_values(by="igr", ascending=False)
imovel_protesto = imovel_protesto[col]

# Dataframe 3
imovel_ajuizamento = imovel_ajuizamento.sort_values(by="igr", ascending=False)
imovel_ajuizamento = imovel_ajuizamento[col]


# Filtro para Geração das listas Mercantil
mercantil_ajuizar = dados_igr_merc[dados_igr_merc['valor_total_da'] >= 200000]
mercantil_protestar =  dados_igr_merc[(dados_igr_merc['valor_total_da'] >= 3000) & (dados_igr_merc['valor_total_da'] < 200000)]
mercantil_negativar = dados_igr_merc[(dados_igr_merc['valor_total_da'] >= 1000) & (dados_igr_merc['valor_total_da'] < 3000)]

# Dataframe 4
mercantil_negativar = mercantil_negativar.sort_values(by="igr", ascending=False)
mercantil_negativar = mercantil_negativar[col]

# Dataframe 5
mercantil_protestar = mercantil_protestar.sort_values(by="igr", ascending=False)
mercantil_protestar = mercantil_protestar[col]

# Dataframe 6
mercantil_ajuizar = mercantil_ajuizar.sort_values(by="igr", ascending=False)
mercantil_ajuizar = mercantil_ajuizar[col]