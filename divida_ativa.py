import os
import dotenv
import psycopg2 as pg
from sqlalchemy import create_engine
import boto3
from io import BytesIO
import zipfile
from datetime import date, datetime
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import shutil

folder_root = os.getcwd()
folder_data = os.path.join(folder_root, 'data')
folder_models = os.path.join(folder_root, 'models')
env = os.path.join(folder_root, '.env')
dotenv.load_dotenv(env)

def connect_database_cliente():
        host = os.getenv("da_db_host")
        port = os.getenv("da_db_port")
        db = os.getenv("da_db_dbnm")
        usr = os.getenv("da_db_usr")
        psw = os.getenv("da_db_psw")
        engine = create_engine(f'postgresql+psycopg2://{usr}:{psw}@{host}:{port}/{db}')
    
        return engine

class etl:

    s3_bucket = os.getenv("S3_BUCKET_NAME")
    s3_nome_pasta = os.getenv("S3_FOLDER_NAME")
    
    s3 = boto3.resource(
            service_name='s3',
            region_name='us-east-1',
            aws_access_key_id=os.getenv("AWS_ACESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACESS_KEY")
            )

    def up_s3_files(self, dataframe, file_name, bucket_name=s3_bucket, folder_name=s3_nome_pasta, s3_resource=s3):
        csv_buffer = BytesIO()
        dataframe.to_csv(csv_buffer, sep=';', index=False)
        file_key_aws = folder_name + file_name
        s3_resource.Object(bucket_name, file_key_aws).put(Body=csv_buffer.getvalue())

    def read_s3_files(self, file_name, bucket_name=s3_bucket, folder_name=s3_nome_pasta, s3_resource=s3):
        file_key_aws = folder_name + file_name
        obj = s3_resource.Bucket(bucket_name).Object(file_key_aws).get()
        df = pd.read_csv(obj['Body'], sep=';')
        return df

    # def coleta_datas(df, df_dim_datas, left_on, nova_coluna_data):
    #         df = pd.merge(
    #             left=df, right=df_dim_datas,
    #             left_on=left_on, right_on='chave_tempo',
    #             how= 'left'
    #             ).rename(
    #                 columns={"date": nova_coluna_data})
    #         return df
    
    def ler_bases_exportadas(nome_arquivo):
        zip_file = os.path.join(folder_data, 'rating_igr.zip')
        z = zipfile.ZipFile(zip_file)
        z.extract(nome_arquivo)
        df = pd.read_csv(nome_arquivo, sep=',', low_memory=False)
        os.remove(nome_arquivo)
        return df

    def ler_estoque_dividas(nome_arquivo):
        zip_file = os.path.join(folder_data, 'estoque_da.zip')
        z = zipfile.ZipFile(zip_file)
        z.extract(nome_arquivo)
        df = pd.read_csv(nome_arquivo, sep=',', low_memory=False)
        os.remove(nome_arquivo)
        return df

class models:

    def open_model(nome_modelo, path_modelo):
        zip_file = os.path.join(path_modelo, 'models.zip')
        z = zipfile.ZipFile(zip_file)
        z.extract(nome_modelo)
        modelo = pickle.load(open(nome_modelo, 'rb'))
        os.remove(nome_modelo)
        return modelo
    
    def save_model(model, models_path, nome_modelo_serializado):
        sav_best_model = open(nome_modelo_serializado, 'wb')
        pickle.dump(model, sav_best_model)
        sav_best_model.close()

        pathModelo = models_path+"\\"+os.path.join(nome_modelo_serializado)
        shutil.move(os.path.abspath(nome_modelo_serializado), pathModelo)
    
class feature_store:

    def load_data_fs_banco(self, folder_data):

        engine = connect_database_cliente()

        base_imovel = etl.query_database(engine, folder_data, sql_file_name='igr_imovel.sql')
        base_mercantil = etl.query_database(engine, folder_data, sql_file_name='igr_mercantil.sql')
        base_parcelas = etl.query_database(engine, folder_data, sql_file_name='etl_parcelamento.sql')
        base_notas_fiscais = etl.query_database(engine, folder_data, sql_file_name='etl_notas_fiscais.sql')
        base_dim_datas = etl.query_database(engine, folder_data, sql_file_name='dicionario_datas.sql')

        return base_imovel, base_mercantil, base_parcelas, base_notas_fiscais, base_dim_datas
    
    def load_data_fs():

        base_imovel = etl.ler_bases_exportadas('imovel.csv')
        base_mercantil = etl.ler_bases_exportadas('mercantil.csv')
        base_parcelas = etl.ler_bases_exportadas('parcelas.csv')
        base_notas_fiscais = etl.ler_bases_exportadas('emissao_notas.csv')
        base_dim_datas = etl.ler_bases_exportadas('dim_datas.csv')

        return base_imovel, base_mercantil, base_parcelas, base_notas_fiscais, base_dim_datas

    def build_fs_contribuinte():
        
        base_imovel, base_mercantil, base_parcelas, base_notas_fiscais, base_dim_datas = feature_store.load_data_fs()

        base_conjunta = pd.concat([base_imovel, base_mercantil])
        dados_pessoas = base_conjunta[['cda', 'id_pessoa', 'situacao', 'cpf_existe', 'edificacao', 'deb_totais', 'deb_pagos', 'valor_tot', 'valor_pago']]
        dados_pessoas['id_pessoa'] = dados_pessoas['id_pessoa'].astype(int) # persistindo tipo de dados

        dados_pessoas.dropna(subset=['id_pessoa'], inplace=True)

        # Calculo que apresenta quantas cdas o contribuinte tem

        frequencia_da_pessoa = dados_pessoas.groupby(['id_pessoa']).count()['cda']
        total_debitos_pessoa = dados_pessoas.groupby(['id_pessoa'])['deb_totais'].sum()
        debitos_pagos_pessoa = dados_pessoas.groupby(['id_pessoa'])['deb_pagos'].sum()
        valor_total_pessoa = dados_pessoas.groupby(['id_pessoa'])['valor_tot'].sum()
        valor_pago_pessoa = dados_pessoas.groupby(['id_pessoa'])['valor_pago'].sum()

        # Agrega informação da base de notas fiscais
        dados_pessoas = pd.merge(
            left=dados_pessoas, left_on='id_pessoa', right=base_notas_fiscais, right_on='id_pessoa', how='left'
        )

        # Substituindo por zero os valores nulos
        dados_pessoas['edificacao'] = dados_pessoas['edificacao'].fillna(1)
        dados_pessoas['qtd_notas_2anos'] = dados_pessoas['qtd_notas_2anos'].fillna(0)

        # Cria variável de situação do contribuinte

        dados_pessoas.loc[dados_pessoas['situacao'] == 'ATIVO', 'situacao_ativa'] = 1
        dados_pessoas.loc[(dados_pessoas['qtd_notas_2anos'] > 0) & (dados_pessoas['situacao'] != 'ATIVO'), 'situacao_ativa'] = 1
        dados_pessoas['situacao_ativa'] = dados_pessoas['situacao_ativa'].fillna(0)

        # Cria variável com peso sobre a situação real

        dados_pessoas['status_situacao'] = dados_pessoas['situacao_ativa'] + dados_pessoas['cpf_existe'] + dados_pessoas['edificacao']
        dados_pessoas.loc[dados_pessoas['situacao_ativa'] == 0, 'status_situacao'] = 0

        # Remove duplicatas dos dados finalizando o dataframe
        dados_pessoas.drop_duplicates(subset=['id_pessoa'], inplace=True)
        dados_pessoas = dados_pessoas.set_index('id_pessoa')

        dados_pessoas['frequencia_da_pessoa'] = frequencia_da_pessoa
        dados_pessoas['total_debitos_pessoa'] = total_debitos_pessoa
        dados_pessoas['debitos_pagos_pessoa'] = debitos_pagos_pessoa
        dados_pessoas['valor_total_pessoa'] = valor_total_pessoa
        dados_pessoas['valor_pago_pessoa'] = valor_pago_pessoa

        # Faz o calculo do historico de pagamento

        dados_pessoas['historico_pagamento_em_qtd'] = (dados_pessoas['debitos_pagos_pessoa'] / dados_pessoas['total_debitos_pessoa']) + 1
        dados_pessoas['historico_pagamento_em_valor'] = (dados_pessoas['valor_pago_pessoa'] / dados_pessoas['valor_total_pessoa']) + 1

        model_predict_contribuinte = models.open_model(nome_modelo="classificador-contribuinte-v1.pkl", path_modelo=folder_models)

        matriz_previsao_class = dados_pessoas[['status_situacao', 'historico_pagamento_em_qtd', 'historico_pagamento_em_valor']]
        dados_pessoas['class_contribuinte'] = model_predict_contribuinte.predict(matriz_previsao_class)
        dados_pessoas.loc[dados_pessoas['frequencia_da_pessoa'] == 1, 'class_contribuinte'] = 4

        # Nomeando a classificação com label de prioridade
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 1, 'class_contribuinte_nome'] = 'PIOR PAGADOR'
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 0, 'class_contribuinte_nome'] = 'PAGADOR INTERMEDIARIO'
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 2, 'class_contribuinte_nome'] = 'BOM PAGADOR'
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 3, 'class_contribuinte_nome'] = 'MELHOR PAGADOR'
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 4, 'class_contribuinte_nome'] = 'PRIMEIRA DIVIDA'

        # Dando pesos para os tipos de contribuintes
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 1, 'class_contribuinte_peso'] = -1.37166 # pior pagador
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 0, 'class_contribuinte_peso'] = 1 # pagador intermediario
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 2, 'class_contribuinte_peso'] = 3.79174 # bom pagador
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 3, 'class_contribuinte_peso'] = 11.19087 # melhor pagador
        dados_pessoas.loc[dados_pessoas['class_contribuinte'] == 4, 'class_contribuinte_peso'] = 1 # primeira divida


        df_feature_store_contribuinte = dados_pessoas.reset_index()

        df_feature_store_contribuinte = df_feature_store_contribuinte[['id_pessoa', 'situacao', 'cpf_existe', 'edificacao', 'qtd_notas_2anos', 
                                                                    'situacao_ativa', 'status_situacao', 
                                                                    'deb_totais','deb_pagos', 'valor_tot', 'valor_pago', 
                                                                    'frequencia_da_pessoa', 'total_debitos_pessoa', 'debitos_pagos_pessoa', 'valor_total_pessoa', 'valor_pago_pessoa', 
                                                                    'historico_pagamento_em_qtd', 'historico_pagamento_em_valor', 
                                                                    'class_contribuinte', 'class_contribuinte_nome', 'class_contribuinte_peso']]
        return df_feature_store_contribuinte
    
    def build_fs_igr():
        
        base_imovel, base_mercantil, base_parcelas, base_notas_fiscais, base_dim_datas = feature_store.load_data_fs()

        base_conjunta = pd.concat([base_imovel, base_mercantil])

        lista_chaves_tempos = ['inscricao_divida']
        lista_nomes_campos_data = ['data_inscricao_da']

        for i in range(len(lista_chaves_tempos)):
            base_conjunta = etl.coleta_datas(base_conjunta, base_dim_datas, lista_chaves_tempos[i], lista_nomes_campos_data[i])

        # Gera as variáveis de dimensão tempo
        base_conjunta['data_inscricao_da'] = pd.to_datetime(base_conjunta['data_inscricao_da'] ,infer_datetime_format=True)
        base_conjunta['ano_inscricao_da'] = base_conjunta['data_inscricao_da'].dt.year

        base_conjunta.drop_duplicates(subset='cda', inplace=True) #Garantia que não houve duplicatas de linhas

        # Seleciona dados sobre a divida
        dados_divida = base_conjunta[[ 'cda', 'id_pessoa', 'tipo_divida', 'valor_tot', 'valor_pago', 'tipo_tributo', 'divida_protestada', 'divida_ajuizada', 'ano_inscricao_da']]
        dados_divida.dropna(subset=['id_pessoa'], inplace=True)
        dados_divida['id_pessoa'] = dados_divida['id_pessoa'].astype(int) # persistindo tipo de dados

        # Calcula a idadae da dívida ativa
        dados_divida['ano_atual'] = date.today().year
        dados_divida['anos_idade_da'] = dados_divida['ano_atual'] - dados_divida['ano_inscricao_da']
        dados_divida = dados_divida.drop(columns=['ano_atual'])

        # Renomeia colunas para nome mais adequados e filtra dataframe
        colunas_nome = {
            'valor_tot': 'valor_total_da',
            'tipo_tributo': 'tipo_tributo_da'
        }
        df_divida_ativa = dados_divida.rename(columns=colunas_nome)

        # Criando variável target Y
        df_divida_ativa['percentual_pago_cda'] = df_divida_ativa['valor_pago'] / df_divida_ativa['valor_total_da']

        # Obtem os dados de parcelamento das dívidas ativas

        base_parcelas.loc[base_parcelas['id_imovel'].isnull(), 'tipo_divida'] = 'mercantil'
        base_parcelas['tipo_divida'] = base_parcelas['tipo_divida'].fillna('imovel')
        base_parcelas['cda'] = base_parcelas['tipo_divida'] + '-' + base_parcelas['cda']

        base_parcelas['parcela_taxa_pagamento_valor'] = base_parcelas['total_valor_pago'] / base_parcelas['valor_lancado_total']
        df_parcelas = base_parcelas[['cda', 'quantidade_reparcelamentos', 'parcela_taxa_pagamento_valor']]
        df_parcelas.drop_duplicates(subset='cda', inplace=True)
        df_parcelas.dropna(subset='cda', inplace=True)

        df_divida_ativa = pd.merge(
            left=df_divida_ativa,
            left_on='cda',
            right=df_parcelas,
            right_on='cda',
            how='left'
        )

        return df_divida_ativa

class train_model:
    def select_data(self):

        dados_divida = etl().read_s3_files(file_name='feature_store_igr.csv')

        dados_contribuinte = etl().read_s3_files(file_name='feature_store_contribuinte.csv')
        
        return dados_divida, dados_contribuinte
    
    def normalize_data(self, dataframe):
        
        normalizador = StandardScaler()
        normalizador.fit(dataframe)
        dados_normalizados = normalizador.fit_transform(dataframe)

        colunas = list(normalizador.get_feature_names_out())
        df_normalizado = pd.DataFrame(dados_normalizados, columns=colunas)

        return df_normalizado


    def filter_enginier_igr(self, dados_divida, dados_contribuinte):
        
        # Filtrando variáveis de interesse para a modelagem
        df_divida = dados_divida[['cda', 'id_pessoa', 'percentual_pago_cda', 'valor_total_da', 'anos_idade_da', 'quantidade_reparcelamentos']]
        df_contribuinte = dados_contribuinte[['id_pessoa', 'frequencia_da_pessoa', 'historico_pagamento_em_qtd', 'status_situacao', 'historico_pagamento_em_valor', 'class_contribuinte_peso']]
        df = pd.merge(left=df_divida, right=df_contribuinte, left_on='id_pessoa', right_on='id_pessoa')

        # Filtrando apenas dados recentes para treinamento
        df = df.query("anos_idade_da < 10")

        df_feature_store = df.drop(columns=['cda', 'id_pessoa'])
        # Substituindo valores vazios
        df_feature_store['quantidade_reparcelamentos'] = df_feature_store['quantidade_reparcelamentos'].fillna(0)
        return df_feature_store
    
    def train_model_igr(self, dataframe, target_feature_name, n_estimators, test_size=0.3):
        seed = 1337

        x = dataframe.drop(columns=[target_feature_name])
        y = dataframe[target_feature_name]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
        
        algoritimo_regressao = RandomForestRegressor(
                                        random_state=seed, 
                                        n_estimators=n_estimators)

        algoritimo_regressao.fit(X_train, y_train)
        
        previsoes = algoritimo_regressao.predict(X_test)
        
        r2 = r2_score(y_test, previsoes)
        mse = mean_squared_error(y_test, previsoes)
        
        variaveis_preditoras = algoritimo_regressao.feature_names_in_
        importancia_variaveis = algoritimo_regressao.feature_importances_
        features_importance = {'features': variaveis_preditoras, 'importancia': importancia_variaveis}
        matriz_importancia = json.loads(pd.DataFrame(features_importance).to_json(orient='index'))
        return algoritimo_regressao, r2, mse, matriz_importancia
    
    def filter_enginier_rg_contribuinte(self, dados_contribuinte):

        # Monta dataframe para clusterização

        df_pipe_cluster = dados_contribuinte.query("frequencia_da_pessoa > 1")
        df_pipe_cluster = df_pipe_cluster.set_index('id_pessoa')
        df_pipe_cluster = df_pipe_cluster[['status_situacao', 'historico_pagamento_em_qtd', 'historico_pagamento_em_valor']]

        return df_pipe_cluster
    
    def train_model_rg_contribuinte(self, df_pipe_cluster, num_cluster):
        faixa_n_clusters = [i for i in range(2, num_cluster)]
        valores_inercia = []
        valores_score = []

        for k in faixa_n_clusters:
            agrupador = KMeans(n_clusters=k, random_state=1337)
            label = agrupador.fit_predict(df_pipe_cluster)

            media_inercia = agrupador.inertia_
            valores_inercia.append(media_inercia)

            media_score = agrupador.score(df_pipe_cluster)
            valores_score.append(media_score)

        def optimal_number_of_clusters(wcss):
            x1, y1 = 2, wcss[0]
            x2, y2 = num_cluster, wcss[len(wcss)-1]

            distances = []
            for i in range(len(wcss)):
                x0 = i+2
                y0 = wcss[i]
                numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
                denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                distances.append(numerator/denominator)
            
            return distances.index(max(distances)) + 2
        valor_ideal_k = optimal_number_of_clusters(valores_inercia)

        # Construindo o melhor agrupador de clusteres
        agrupador = KMeans(n_clusters=valor_ideal_k, random_state=1337)
        agrupador.fit_transform(df_pipe_cluster)
        mean_inertia = agrupador.inertia_

        # Obtendo o ponto central dos clusteres
        centros = agrupador.cluster_centers_
        df_centroide = pd.DataFrame(centros, columns = df_pipe_cluster.columns).round(3)
        df_centroide['cluster'] = df_centroide.index
        df_pipe_cluster['label_cluster'] = agrupador.labels_

        json_centroide = json.loads(pd.DataFrame(df_centroide).to_json(orient='index')) # Pontos centrais dos clusteres

        # Modelo de decisão para encontrar os clusteres
        x_cluster = df_pipe_cluster.drop(columns=['label_cluster'])
        y_cluster = df_pipe_cluster['label_cluster']
        X_train, X_test, y_train, y_test = train_test_split(x_cluster, y_cluster, test_size=0.3, random_state=1337)

        model_predict_contribuinte = RandomForestClassifier(random_state=1337)
        model_predict_contribuinte.fit(X_train, y_train)

        return valor_ideal_k, mean_inertia, json_centroide, model_predict_contribuinte
    
class unitary_tests:
    def validation_model_igr(self, modelo_regressao, r2, mse):

        data_today = str(datetime.today()).replace(" ","_")
        
        if (r2 > 0.82) and (mse < 0.019):
            nome_modelo = f"modeloDA-igr-divida{data_today}.pkl"
            models.save_model(model=modelo_regressao, models_path=folder_models, nome_modelo_serializado=nome_modelo)
            test_value = {
                 'modelo' : str(nome_modelo),
                 'r2': r2,
                 'mse': mse
            }
        else:
            test_value = "Modelo nao possui R2 e MSE melhores que o modelo em producao"
        return test_value
    
    def validation_model_rg_contribuinte(self, modelo_decisao, k_cluster, json_centroide):
        
        data_today = str(datetime.today()).replace(" ","_")

        if k_cluster > 4:
            nome_modelo = f"classificador-contribuinte-{data_today}.pkl"
            models.save_model(model=modelo_decisao, models_path=folder_models, nome_modelo_serializado=nome_modelo)
            test_value = {
                 'modelo' : str(nome_modelo),
                 'numero_clusteres': k_cluster,
                 'pontos_centrais': json_centroide
            }
        else:
            test_value = "Nao foram encontrados diferentes grupos para os contribuintes."
        return test_value
    
class predict:
    
    def batch_select_data(self):
        base_conjunta = etl.ler_estoque_dividas('estoque-divida.csv')
        base_parcelas = etl.ler_estoque_dividas('dados_parcelamento_divida.csv')

        # Gera as variáveis de dimensão tempo
        base_conjunta['data_inscricao_da'] = pd.to_datetime(base_conjunta['inscricao_divida'] ,infer_datetime_format=True)
        base_conjunta['ano_inscricao_da'] = base_conjunta['data_inscricao_da'].dt.year

        # Seleciona dados sobre a divida
        dados_divida = base_conjunta[[ 'cda', 'id_contribuinte', 'da_aberto', 'da_paga', 'arrecadacao_divida', 'refis', 'ajuizamento_divida', 'bairro', 'atividade_principal', 'tipo_divida', 'valor_tot', 'tipo_tributo', 'ano_inscricao_da', 'protesto', 'ajuizamento', 'base']]
        dados_divida.dropna(subset=['id_contribuinte'], inplace=True)
        dados_divida['id_contribuinte'] = dados_divida['id_contribuinte'].astype(int) # persistindo tipo de dados

        # Calcula a idadae da dívida ativa
        dados_divida['ano_atual'] = date.today().year
        dados_divida['anos_idade_da'] = dados_divida['ano_atual'] - dados_divida['ano_inscricao_da']
        dados_divida = dados_divida.drop(columns=['ano_atual'])

        # Renomeia colunas para nome mais adequados e filtra dataframe
        colunas_nome = {
            'valor_tot': 'valor_total_da',
            'tipo_tributo': 'tipo_tributo_da'
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

        dados_divida, dados_contribuinte = train_model().select_data()

        dados_totais = pd.merge(
            left=df_divida_ativa,
            right=dados_contribuinte,
            left_on='id_contribuinte',
            right_on='id_pessoa',
            how='left')
        return dados_totais
    
    def batch_filter_enginer(self, dataframe):

        df = dataframe[['cda', 'id_contribuinte', 'da_aberto', 'da_paga', 'arrecadacao_divida', 'tipo_divida', 'valor_total_da', 
        'tipo_tributo_da', 'ano_inscricao_da', 'protesto', 'ajuizamento', 'refis', 'ajuizamento_divida',
        'bairro', 'atividade_principal', 'base', 'anos_idade_da', 'quantidade_reparcelamentos', 'qtd_notas_2anos',
        'situacao', 'status_situacao', 'frequencia_da_pessoa', 'historico_pagamento_em_qtd', 'historico_pagamento_em_valor', 
        'class_contribuinte', 'class_contribuinte_nome', 'class_contribuinte_peso']]

        # Ajustando pesos dos contribuintes
        df.loc[df['frequencia_da_pessoa'] <= 1, 'class_contribuinte_nome'] == 'PRIMEIRA DIVIDA'
        df.loc[df['class_contribuinte_nome'] == 'PRIMEIRA DIVIDA' , 'class_contribuinte_peso'] == 3.79174
        df['class_contribuinte_nome'] = df['class_contribuinte_nome'].fillna('PIOR PAGADOR')
        df.loc[df['class_contribuinte_nome'] == 'PIOR PAGADOR' , 'class_contribuinte_peso'] == -1.37166
        df.fillna(0, inplace=True)

        # Filtra os dados que precisamos para previsão
        df_feature_store = df[['valor_total_da', 'anos_idade_da', 'quantidade_reparcelamentos', 
                            'frequencia_da_pessoa', 'historico_pagamento_em_qtd', 'status_situacao', 'historico_pagamento_em_valor', 'class_contribuinte_peso']]
        
        return df, df_feature_store
    
    def batch_make_predict(self, prod_model, df, df_normalizado):
        
        igr_model = models.open_model(nome_modelo=prod_model, path_modelo=folder_models)
        previsoes = igr_model.predict(df_normalizado)
        df['igr'] = previsoes
        df.loc[df['status_situacao'] == 0, 'igr'] = 0
        df.loc[df['anos_idade_da'] >= 15, 'igr'] = 0

        return df
    
    def solo_filter_enginer(self, dataframe, modelo_contribuinte_prod):
        
        dataframe['status_situacao'] = dataframe['situacao_ativa'] + dataframe['edificacao'] + dataframe['tem_cpf']
        dataframe.loc[dataframe['situacao_ativa'] == 0, 'status_situacao'] = 0

        dataframe['historico_pagamento_em_qtd'] = dataframe['historico_pagamento_em_qtd'] + 1
        dataframe['historico_pagamento_em_valor'] = dataframe['historico_pagamento_em_valor'] + 1

        model_predict_contribuinte = models.open_model(nome_modelo=modelo_contribuinte_prod, path_modelo=folder_models)

        matriz_previsao_class = dataframe[['status_situacao', 'historico_pagamento_em_qtd', 'historico_pagamento_em_valor']]
        dataframe['class_contribuinte'] = model_predict_contribuinte.predict(matriz_previsao_class)
        dataframe.loc[dataframe['frequencia_da_pessoa'] == 1, 'class_contribuinte'] = 4

        # Dando pesos para os tipos de contribuintes
        dataframe.loc[dataframe['class_contribuinte'] == 1, 'class_contribuinte_peso'] = -1.37166
        dataframe.loc[dataframe['class_contribuinte'] == 0, 'class_contribuinte_peso'] = 1
        dataframe.loc[dataframe['class_contribuinte'] == 2, 'class_contribuinte_peso'] = 3.79174
        dataframe.loc[dataframe['class_contribuinte'] == 3, 'class_contribuinte_peso'] = 11.19087
        dataframe.loc[dataframe['class_contribuinte'] == 4, 'class_contribuinte_peso'] = 1 
        
        dataframe = dataframe.drop(columns=['class_contribuinte', 'situacao_ativa', 'edificacao', 'tem_cpf'])
        
        return dataframe

    def solo_make_predict(self, dataframe, modelo_igr_prod, modelo_contribuinte_prod):

        igr_model = models.open_model(nome_modelo=modelo_igr_prod, path_modelo=folder_models)
        previsoes = igr_model.predict(dataframe)
        dataframe['igr'] = previsoes
        dataframe.loc[dataframe['status_situacao'] == 0, 'igr'] = 0 # O igr se tornar zero quando o resultado da coluna status situação é zero
        dataframe.loc[dataframe['anos_idade_da'] >= 15, 'igr'] = 0 # se a dívida for maio que 15 ou igual o igr será zero 

        model_predict_contribuinte = models.open_model(path_modelo=folder_models, nome_modelo=modelo_contribuinte_prod)
        matriz_previsao_class = dataframe[['status_situacao', 'historico_pagamento_em_qtd', 'historico_pagamento_em_valor']]
        dataframe['class_contribuinte'] = model_predict_contribuinte.predict(matriz_previsao_class)
        dataframe.loc[dataframe['frequencia_da_pessoa'] == 1, 'class_contribuinte'] = 4

        dataframe = predict().make_rating_divida(dataframe=dataframe)
        dataframe = dataframe.drop(columns=['class_contribuinte'])
        json_predict = json.loads(dataframe.to_json(orient='index'))
        
        return json_predict

    def make_rating_divida(self, dataframe):
        # Quando o IGR é zero classifica o rating como BAIXISSIMA RECUPERAÇÃO
        dataframe.loc[dataframe['igr'] == 0, 'rating_divida'] = 'BAIXISSIMA'

        # Se a Coluna do Rating estiver como nula e o contribuinte for MELHOR PAGADOR classifica o rating da dívida como ALTISSIMA RECUPERAÇÃO
        dataframe.loc[(dataframe['rating_divida'].isnull()) & (dataframe['class_contribuinte'] ==  3), 'rating_divida'] = 'ALTISSIMA'

        # Se o contribuinte for PIOR PAGADOR e o IGR for diferente de 0 classifica o rating da divida como BAIXA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 1) & (dataframe['igr']!=0), 'rating_divida'] = 'BAIXA'

        # Se o contribuinte for BOM PAGADOR e o IGR for maior ou igual a 0.5 classifica o rating da divida como ALTISSIMA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 2) & (dataframe['igr']>=0.5), 'rating_divida'] = 'ALTISSIMA'

        # Se o contribuinte for BOM PAGADOR e o IGR for menor que 0.5 e maior ou igual a 0.1 classifica o rating da divida como ALTA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 2) & (dataframe['igr']<0.5) & (dataframe['igr']>=0.1), 'rating_divida'] = 'ALTA'

        # Se o contribuinte for BOM PAGADOR e o IGR for menor que 0.1 e diferente de zero classifica o rating da divida como MEDIA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 2) & (dataframe['igr']<0.1) & (dataframe['igr']!=0), 'rating_divida'] = 'MEDIA'

        # Se o contribuinte for PAGADOR INTERMEDIARIO e o IGR for maior ou igual a 0.5 classifica o rating da divida como ALTA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 0) & (dataframe['igr']>=0.5), 'rating_divida'] = 'ALTA'

        # Se o contribuinte for PAGADOR INTERMEDIARIO e o IGR for menor que 0.5 e maior ou igual a 0.05 classifica o rating da divida como MEDIA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 0) & (dataframe['igr']<0.5) & (dataframe['igr']>=0.05), 'rating_divida'] = 'MEDIA'

        # Se o contribuinte for PAGADOR INTERMEDIARIO e o IGR for menor que 0.05 e diferente de zero classifica o rating da divida como BAIXA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 0) & (dataframe['igr']<0.05) & (dataframe['igr']!=0), 'rating_divida'] = 'BAIXA'
        
        # Se o contribuinte for PRIMEIRA DIVIDA e o IGR for maior ou igual a 0.3 classifica o rating da divida como ALTA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 4) & (dataframe['igr']>=0.3), 'rating_divida'] = 'ALTA'

        # Se o contribuinte for PRIMEIRA DIVIDA e o IGR for menor que 0.3 e maior ou igual a 0.1 classifica o rating da divida como MEDIA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 4) & (dataframe['igr']<0.3) & dataframe['igr']>=0.1, 'rating_divida'] = 'MEDIA'

        # Se o contribuinte for PRIMEIRA DIVIDA e o IGR for menor que 0.1 e diferente de 0 classifica o rating da divida como BAIXA RECUPERAÇÃO
        dataframe.loc[(dataframe['class_contribuinte'] == 4) & (dataframe['igr']<0.1) & (dataframe['igr']!=0), 'rating_divida'] = 'BAIXA'
        return dataframe
    
    def  make_rating_dteste(self, dataframe):
        dataframe.loc