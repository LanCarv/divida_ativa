import os
import pandas as pd
import zipfile
from datetime import date
import dotenv
import boto3
from io import BytesIO
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')

import warnings
warnings.filterwarnings("ignore")

rootPath = os.path.dirname(os.getcwd())
dataPath = os.path.join(rootPath, 'data')
modelsPath = os.path.join(rootPath, 'models')
env = os.path.join(rootPath, '.env')
dotenv.load_dotenv(dotenv_path=env)

print("Iniciando carregamento dos dados")
zip_file = os.path.join(dataPath, 'rating_igr_18_09.zip')
z = zipfile.ZipFile(zip_file)

def ler_bases_exportadas(nome_arquivo):
    z.extract(nome_arquivo)
    df = pd.read_csv(nome_arquivo, sep=',')
    os.remove(nome_arquivo)
    return df

# Transforma as chaves de tempo em data
# def coleta_datas(df, left_on, nova_coluna_data):
#     df = pd.merge(
#         left=df, right=base_dim_datas,
#         left_on=left_on, right_on='chave_tempo',
#         how='left'
#     ).rename(
#         columns={"date": nova_coluna_data})
#     return df

def up_s3_files(dataframe, bucket_name, folder_name, file_name):
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, sep=';', index=False)
    file_key_aws = folder_name + file_name
    s3_resource.Object(bucket_name, file_key_aws).put(Body=csv_buffer.getvalue())

base_imovel = ler_bases_exportadas('imovel.csv')
base_mercantil = ler_bases_exportadas('mercantil.csv')
#base_parcelas = ler_bases_exportadas('parcelas.csv')
base_notas_fiscais = ler_bases_exportadas('emissao_notas.csv')
#base_dim_datas = ler_bases_exportadas('dim_datas.csv')

base_conjunta = pd.concat([base_imovel, base_mercantil])

print("Inicia transformação das variáveis sobre a dívida")
# Transforma as chaves de tempo em data
# def coleta_datas(df, left_on, nova_coluna_data):
#
#     df = pd.merge(
#         left=df, right=base_dim_datas,
#         left_on=left_on, right_on='chave_tempo',
#         how= 'left'
#         ).rename(
#             columns={"date": nova_coluna_data})
#     return df

# lista_chaves_tempos = ['inscricao_divida']
# lista_nomes_campos_data = ['data_inscricao_da']
#
# for i in range(len(lista_chaves_tempos)):
#     base_conjunta = coleta_datas(base_conjunta, lista_chaves_tempos[i], lista_nomes_campos_data[i])

# Gera as variáveis de dimensão tempo
# base_conjunta['data_inscricao_da'] = pd.to_datetime(base_conjunta['data_inscricao_da'] ,infer_datetime_format=True)
# base_conjunta['ano_inscricao_da'] = base_conjunta['data_inscricao_da'].dt.year

# Gera as variáveis de tempo
base_conjunta['data_divida'] = pd.to_datetime(base_conjunta['inscricao_divida'], infer_datetime_format = True)
base_conjunta['ano_inscricao_da'] = base_conjunta['data_divida'].dt.year

base_conjunta.drop_duplicates(subset='cda', inplace=True) #Garantia que não houve duplicatas de linhas

# Seleciona dados sobre a divida
dados_divida = base_conjunta[[ 'cda', 'id_pessoa', 'tipo_divida', 'valor_tot', 'valor_pago', 'protesto', 'divida_ajuizada', 'ano_inscricao_da']]
dados_divida.dropna(subset=['id_pessoa'], inplace=True)
dados_divida['id_pessoa'] = dados_divida['id_pessoa'].astype(str) # persistindo tipo de dados

# cria coluna e instancia como zero para preencher com a base de parcelas
# dados_divida['quantidade_reparcelamentos'] = 0

# concateno as bases mergeando para atribuir os valores aos seus devidos locais
# dados_divida = base_conjunta.merge(base_parcelas,how='left',on=['cda', 'tipo_divida', 'id_pessoa'],suffixes=('', '_PARC'))
# dados_divida.loc[ dados_divida['total_valor_pago'].isna(), 'total_valor_pago' ] = 0

# Somando valores a vista e parcelados em uma nova coluna
# dados_divida['valor_pago_vista_parc'] = dados_divida['valor_pago'] + dados_divida['total_valor_pago']
# dados_divida['valor_pago_vista_parc'].sum() / 1000000

# tratamento dos dados de parcelamento para concatenação das bases
valor_tot = dados_divida.groupby(['cda', 'id_pessoa', 'tipo_divida'])['valor_tot'].sum()
valor_pago = dados_divida.groupby(['cda', 'id_pessoa', 'tipo_divida'])['valor_pago'].sum()
divida_protestada = dados_divida.groupby(['cda', 'id_pessoa', 'tipo_divida'])['protesto'].max()
divida_ajuizada = dados_divida.groupby(['cda', 'id_pessoa', 'tipo_divida'])['divida_ajuizada'].max()
ano_inscricao_da = dados_divida.groupby(['cda', 'id_pessoa', 'tipo_divida'])['ano_inscricao_da'].max()
# quantidade_reparcelamentos = dados_divida.groupby(['cda', 'id_pessoa', 'tipo_divida'])['quantidade_reparcelamentos'].sum()

aux = pd.merge(valor_tot, valor_pago, on = ['cda', 'id_pessoa', 'tipo_divida'], how = "left")
aux2 = pd.merge(divida_protestada, divida_ajuizada, on = ['cda', 'id_pessoa', 'tipo_divida'], how = "left")
# aux3 = pd.merge(ano_inscricao_da, on = ['cda', 'id_pessoa', 'tipo_divida'], how = "left")

aux4 = pd.merge(aux, aux2, on = ['cda', 'id_pessoa', 'tipo_divida'], how = "left")
aux5 = pd.merge(aux4, ano_inscricao_da, on = ['cda', 'id_pessoa', 'tipo_divida'], how = "left")

# renomeia a coluna criada para valor_pago usada no modelo
# aux5.rename( columns={'valor_pago_vista_parc':'valor_pago'}, inplace=True)

# Calcula a idade da dívida ativa
aux5['ano_atual'] = date.today().year
aux5['anos_idade_da'] = aux5['ano_atual'] - aux5['ano_inscricao_da']
dados_divida = aux5.drop(columns=['ano_atual'])
dados_divida = dados_divida.reset_index()

# Renomeia colunas para nome mais adequados e filtra dataframe
colunas_nome = {
    'valor_tot': 'valor_total_da'
}
df_divida_ativa = dados_divida.rename(columns=colunas_nome)


# Criando variável target Y que será predita
df_divida_ativa['percentual_pago_cda'] = df_divida_ativa['valor_pago'] / df_divida_ativa['valor_total_da']


print("Inicia a conexão com S3 para inscrição dos dados")
# Cria conexão ao s3 e preenche a tabela com os dados
s3_resource = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.getenv("AWS_ACESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACESS_KEY")
    )

def up_s3_files(dataframe, bucket_name, folder_name, file_name):
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, sep=';', index=False)
    file_key_aws = folder_name + file_name
    s3_resource.Object(bucket_name, file_key_aws).put(Body=csv_buffer.getvalue())

up_s3_files(dataframe=df_divida_ativa, 
            bucket_name=os.getenv("S3_BUCKET_NAME"),
            folder_name=os.getenv("S3_FOLDER_NAME"), 
            file_name='feature_store_divida.csv')

print("Dados atualizados e persistidos no bucket S3")
print("Processo finalizado")