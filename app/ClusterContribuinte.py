import os
import pickle
import shutil
import warnings

import boto3
import dotenv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from app.ProcessaDadosContribuinte import pessoas_da_aberto_0, pessoas_da_aberto_1

warnings.filterwarnings("ignore")

rootPath = os.path.dirname(os.getcwd())
dataPath = os.path.join(rootPath, 'data')
modelsPath = os.path.join(rootPath, 'models')
env = os.path.join(rootPath, '.env')
dotenv.load_dotenv(dotenv_path=env)


def criar_pipeline_treinamento(self):
    df_pipe_cluster = pessoas_da_aberto_0.query("num_dist_cda > 1")
    df_pipe_cluster = df_pipe_cluster[['id_pessoa',
                                       'tipo_divida',
                                       'situacao_cobranca',
                                       'num_dist_cda',
                                       'quantidade_reparcelamento',
                                       'historico_pagamento_em_valor'
                                       ]]
    df_pipe_cluster = df_pipe_cluster.set_index(['id_pessoa', 'tipo_divida'])
    return df_pipe_cluster

def criar_pipeline_predicao(self):
    df_pipe_predict = pessoas_da_aberto_1[['id_pessoa',
                                           'tipo_divida',
                                           'situacao_cobranca',
                                           'num_dist_cda',
                                           'quantidade_reparcelamento',
                                           'historico_pagamento_em_valor'
                                           ]]
    df_pipe_predict = df_pipe_predict.set_index(['id_pessoa', 'tipo_divida'])
    return df_pipe_predict
print("Iniciando clusterização")
faixa_n_clusters = [i for i in range(2, 16)]
valores_inercia = []
valores_score = []

for k in faixa_n_clusters:
    agrupador = KMeans(n_clusters=k, random_state=1337)
    label = agrupador.fit_predict(criar_pipeline_treinamento)
    print(f"Treinamento do agrupador para K= {k} finalizado")

    media_inercia = agrupador.inertia_
    valores_inercia.append(media_inercia)
    print(f"Inércia calculada para o agrupador de K= {k}. Inércia: {media_inercia}")

    media_score = agrupador.score(criar_pipeline_treinamento)
    valores_score.append(media_score)
    print(f"Score calculado para o agrupador de K= {k}. Socre: {media_score}")
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 15, wcss[len(wcss) - 1]
    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 2

valor_ideal_k = optimal_number_of_clusters(valores_inercia)
print("Melhor valor de K:", valor_ideal_k)

VALOR_K = 5
# Construindo o melhor agrupador de clusteres
agrupador = KMeans(n_clusters=VALOR_K, random_state=1337)
agrupador.fit_transform(df_pipe_cluster)

# Calculando o Silhouete (qualidade dos clusters, separação intra cluster e inter cluster)
labels = agrupador.labels_
silhouette_avg = silhouette_score(df_pipe_cluster, labels)
silhouette_vals = silhouette_samples(df_pipe_cluster, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Obtendo o ponto central dos clusteres
centros = agrupador.cluster_centers_
df_centroide = pd.DataFrame(centros, columns=df_pipe_cluster.columns).round(3)
df_centroide['cluster'] = df_centroide.index

# Obtendo o label para cada pessoa
df_pipe_cluster['label_cluster'] = agrupador.labels_

dicionario_clusteres = {
    'class_contribuinte': [0, 3, 1, 4, 2, 5],
    'class_contribuinte_nome': ['DEVEDOR PESSIMO',
                                'DEVEDOR RUIM',
                                'DEVEDOR CONTUMAZ',
                                'DEVEDOR BOM',
                                'DEVEDOR EXCEL',
                                'PRIMEIRA DIVIDA'],

    'class_contribuinte_perfil': ['NAO PAGA INACESSIVEL',
                                  'NAO PAGA ACESSIVEL',
                                  'NEGOCIADOR MTS PARC',
                                  'PAGADOR NEGOCIADOR',
                                  'PAGADOR ORGANICO',
                                  'NOVO EM DIVIDA']
}

df_dicionario_clusteres = pd.DataFrame(dicionario_clusteres)

print("Iniciando o treinamento do modelo")
x_cluster = df_pipe_cluster.drop(columns=['label_cluster'])
y_cluster = df_pipe_cluster['label_cluster']

X_train, X_test, y_train, y_test = train_test_split(x_cluster, y_cluster, test_size=0.3, random_state=1337)
model_predict_contribuinte = RandomForestClassifier(random_state=1337)
model_predict_contribuinte.fit(X_train, y_train)

score_validacao = model_predict_contribuinte.score(X_test, y_test)
print("Score de validacao:", score_validacao)

# Previsão da classificação para a base DA_ABERTO == 0
matriz_previsao_class_da_aberto_0 = pessoas_da_aberto_0[['situacao_cobranca',
                                 'num_dist_cda',
                                 'quantidade_reparcelamento',
                                 'historico_pagamento_em_valor']]
pessoas_da_aberto_0['class_contribuinte'] = model_predict_contribuinte.predict(matriz_previsao_class_da_aberto_0)
pessoas_da_aberto_0.loc[pessoas_da_aberto_0['num_dist_cda'] == 1, 'class_contribuinte'] = VALOR_K

# Previsão da classificação para a base DA_ABERTO == 1
matriz_previsao_class_da_aberto_1 = pessoas_da_aberto_1[['situacao_cobranca',
                                 'num_dist_cda',
                                 'quantidade_reparcelamento',
                                 'historico_pagamento_em_valor']]
pessoas_da_aberto_1['class_contribuinte'] = model_predict_contribuinte.predict(matriz_previsao_class_da_aberto_1)
pessoas_da_aberto_1.loc[pessoas_da_aberto_1['num_dist_cda'] == 1, 'class_contribuinte'] = VALOR_K


pessoas_da_aberto_0['da_aberto'] = 0
pessoas_da_aberto_1['da_aberto'] = 1
pessoas_completo_com_previsao = pd.concat([pessoas_da_aberto_0, pessoas_da_aberto_1])

print("Classificando os clusters")
pessoas_completo_com_previsao = pd.merge(pessoas_completo_com_previsao,
         df_dicionario_clusteres,
         on = "class_contribuinte",
         how = "left")

df_classificao_contribuinte = pessoas_completo_com_previsao[['id_pessoa', 'tipo_divida', 'class_contribuinte_nome']]

print("Iniciando a análise discriminante")
pessoas_completo_com_previsao.loc[(pessoas_completo_com_previsao['valor_tot'].isna()) | (pessoas_completo_com_previsao['valor_tot'] == 0) , 'valor_tot'] = 1
pessoas_completo_com_previsao['percentual_pago_cda'] = pessoas_completo_com_previsao['valor_pago'] / pessoas_completo_com_previsao['valor_tot']

# Imputando historico_pagamento_em_valor = 1 nos casos que passa de 1
pessoas_completo_com_previsao.loc[pessoas_completo_com_previsao['percentual_pago_cda'] > 1, 'percentual_pago_cda'] = 1

df_analise_discriminante = pessoas_completo_com_previsao[['id_pessoa', 'tipo_divida', 'percentual_pago_cda', 'class_contribuinte_nome']]
df_analise_discriminante = df_analise_discriminante.set_index(['id_pessoa', 'tipo_divida'])
df_analise_discriminante = df_analise_discriminante.reset_index()

print("Dummyzando a variável de classificação")
ohe = OneHotEncoder(dtype=int)
colunas_ohe = ohe.fit_transform(df_analise_discriminante[['class_contribuinte_nome']]).toarray()
df_2 = pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['class_contribuinte_nome']))

df_n_categorico = df_analise_discriminante.drop(columns=['class_contribuinte_nome'], axis=1)
df_pipe_discriminante = pd.concat([df_n_categorico, df_2], axis=1)
df_pipe_discriminante = df_pipe_discriminante.set_index(['id_pessoa', 'tipo_divida'])

print("Executando algoritmo LinearDiscriminantAnalysis")
x_analise_discriminante = df_pipe_discriminante.drop(columns=['percentual_pago_cda'])
y_analise_discriminante = df_pipe_discriminante['percentual_pago_cda'].astype('int')

analise_discriminante = LinearDiscriminantAnalysis()
analise_discriminante.fit(x_analise_discriminante, y_analise_discriminante)

dados_analise_disc = {'variavel': analise_discriminante.feature_names_in_, 'coeficiente' : analise_discriminante.coef_[0].round(5)}
pesos_analise_disc = pd.DataFrame(dados_analise_disc).sort_values('variavel').reset_index().drop(columns=['index'])

pesos_ord = pesos_analise_disc.sort_values(by='coeficiente')['coeficiente'].reset_index().drop(columns="index")
pesos_ord.sort_values(by="coeficiente", inplace=True)

print("Adicionando pesos ao label do cluster correspondente")
pesos_analise_disc2 = df_dicionario_clusteres

pesos_analise_disc2.loc[pesos_analise_disc2['class_contribuinte_nome'] == "DEVEDOR PESSIMO", 'coeficiente'] = pesos_ord.loc[0, "coeficiente"]
pesos_analise_disc2.loc[pesos_analise_disc2['class_contribuinte_nome'] == "DEVEDOR RUIM", 'coeficiente'] = pesos_ord.loc[1, "coeficiente"]
pesos_analise_disc2.loc[pesos_analise_disc2['class_contribuinte_nome'] == "DEVEDOR CONTUMAZ", 'coeficiente'] = pesos_ord.loc[2, "coeficiente"]
pesos_analise_disc2.loc[pesos_analise_disc2['class_contribuinte_nome'] == "DEVEDOR BOM", 'coeficiente'] = pesos_ord.loc[3, "coeficiente"]
pesos_analise_disc2.loc[pesos_analise_disc2['class_contribuinte_nome'] == "DEVEDOR EXCEL", 'coeficiente'] = pesos_ord.loc[5, "coeficiente"]
pesos_analise_disc2.loc[pesos_analise_disc2['class_contribuinte_nome'] == "PRIMEIRA DIVIDA", 'coeficiente'] = pesos_ord.loc[4, "coeficiente"]
pesos_analise_disc2.sort_values(by="coeficiente", inplace=True)

print("Salvando pickle do modelo treinado")
def salva_modelo_serializado(nome_modelo_serializado, modelo):
    sav_best_model = open(nome_modelo_serializado, 'wb')
    pickle.dump(modelo, sav_best_model)
    sav_best_model.close()

    pathModelo = modelsPath+"\\"+os.path.join(nome_modelo_serializado)
    shutil.move(os.path.abspath(nome_modelo_serializado), pathModelo)

salva_modelo_serializado("classificador-contribuinte_prime_k5_dez.pkl", model_predict_contribuinte)


print("Preparando feature store do contribuinte")
pessoas_completo_com_previsao = pd.merge(pessoas_completo_com_previsao,
         pesos_analise_disc2,
         on = ["class_contribuinte_nome"],
         how = "left")

pessoas_completo_com_previsao = pessoas_completo_com_previsao.rename(columns = {'coeficiente':'class_contribuinte_peso'})
pessoas_completo_com_previsao = pessoas_completo_com_previsao.drop(columns = ['class_contribuinte_perfil_x', 'class_contribuinte_y','class_contribuinte_perfil_y'])
df_feature_store_contribuinte = pessoas_completo_com_previsao

print("Inicia a conexão com S3 para inscrição dos dados")
# Cria conexão ao s3 e preenche a tabela com os dados
s3_resource = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.getenv("AWS_ACESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACESS_KEY")
    )

NOME_ARQ_SALVAR_S3 = 'feature_store_contribuinte.csv'

up_s3_files(dataframe=df_feature_store_contribuinte,
            bucket_name=os.getenv("S3_BUCKET_NAME"),
            folder_name=os.getenv("S3_FOLDER_NAME"),
            file_name= NOME_ARQ_SALVAR_S3)

print("Dados upados no s3")
print("Processo finalizado")
