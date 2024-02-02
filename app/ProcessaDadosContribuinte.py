import os

import boto3
import dotenv
import zipfile
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

class ProcessaDadosContribuinte:
    s3 = boto3.resource(
        service_name='s3',
        region_name='us-east-1',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    def __init__(self):
        self.rootPath = os.path.dirname(os.getcwd())
        self.dataPath = os.path.join(self.rootPath, 'data')
        self.modelsPath = os.path.join(self.rootPath, 'models')
        env = os.path.join(self.rootPath, '.env')
        dotenv.load_dotenv(dotenv_path=env)

        self.base_notas_fiscais = None
        self.base_conjunta = None
        self.base_fechada = None
        self.base_aberta = None
        self.valor_pago_fechado = None
        self.valor_pago_aberto = None
        self.pessoas_da_aberto_0 = None
        self.pessoas_da_aberto_1 = None

    def read_s3(self, bucket_name, folder_name, file_name):
        file_key_aws = folder_name + file_name
        obj = self.s3.Bucket(bucket_name).Object(file_key_aws).get()
        df = pd.read_csv(obj['Body'], sep=';')
        return df

    def carregar_dados_do_s3(self ,s3_bucket_name, s3_folder_name, zip_file='base_treino.zip'):
        print("Iniciando carregamento dos dados")

        # Nome do arquivo no S3
        s3_file_key = os.path.join(s3_folder_name, zip_file)

        # Leitura do arquivo zip do S3
        local_zip_path = self.read_s3(s3_bucket_name, s3_folder_name, zip_file)

        # Extrair o arquivo zip
        with zipfile.ZipFile(local_zip_path, 'r') as z:
            # Modificado: Adicionado o diretório de extração
            z.extractall()

            # Ler os arquivos CSV extraídos
            self.base_notas_fiscais = pd.read_csv('emissao_notas.csv', sep=';')
            self.base_conjunta = pd.read_csv('imovel_mercantil.csv', sep=';')

        # Renomeando coluna de idade da dívida
        self.base_conjunta = self.base_conjunta.rename(columns={'idade_divida': 'anos_idade_da'})

        self.base_fechada = self.base_conjunta[self.base_conjunta['da_aberto'] == 0]
        self.base_aberta = self.base_conjunta[self.base_conjunta['da_aberto'] == 1]

        # Remover os arquivos CSV locais após o uso
        os.remove('emissao_notas.csv')
        os.remove('imovel_mercantil.csv')

    def ler_bases_exportadas(self, zip_file, nome_arquivo):
        zip_file.extract(nome_arquivo)
        df = pd.read_csv(nome_arquivo, sep=',')
        os.remove(nome_arquivo)
        return df

    def criar_obj_valor_tot_pago_aberto(self, base_conjunta):
        valor_tot = base_conjunta.groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_tot'].sum().to_frame().reset_index()
        valor_pago = base_conjunta.groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_pago'].sum().to_frame().reset_index()

        # O que está em aberto
        valor_aberto_tot = base_conjunta[base_conjunta['da_aberto'] == 1].groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_tot'].sum().to_frame().reset_index()
        valor_aberto_pg = base_conjunta[base_conjunta['da_aberto'] == 1].groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_pago'].sum().to_frame().reset_index()
        valor_aberto = pd.merge(valor_aberto_tot, valor_aberto_pg, on=['cda', 'tipo_divida', 'id_pessoa'], how="left")
        valor_aberto['valor_aberto'] = valor_aberto['valor_tot'] - valor_aberto['valor_pago']
        valor_aberto.drop(columns=['valor_tot', 'valor_pago'], inplace=True)

        # Merge de valor_tot & valor_pago
        valor_tot_pago = pd.merge(valor_tot, valor_pago, on=['cda', 'tipo_divida', 'id_pessoa'], how="left")
        # Merge de (valor_tot e valor_pago) & valor_aberto
        valor_tot_pago_aberto = pd.merge(valor_tot_pago, valor_aberto, on=['cda', 'tipo_divida', 'id_pessoa'], how="left")

        # O que a gente esperava receber: dif_tot_pago
        valor_tot_pago_aberto['dif_tot_pago'] = valor_tot_pago_aberto['valor_tot'] - valor_tot_pago_aberto['valor_pago']
        # O quanto perdeu entre o que a gente esperava receber e o que foi efetivamente pago
        valor_tot_pago_aberto['dif_tot_pago_aberto'] = round(
            valor_tot_pago_aberto['dif_tot_pago'] - valor_tot_pago_aberto['valor_aberto'], 5)

        valor_tot_pago_aberto.sort_values(by='dif_tot_pago_aberto', ascending=False)

        if base_conjunta['da_aberto'].iloc[0] == 0:
            self.valor_pago_fechado = valor_tot_pago_aberto
        elif base_conjunta['da_aberto'].iloc[0] == 1:
            self.valor_pago_aberto = valor_tot_pago_aberto

    def quantidades_por_id_pessoa_e_divida(self, dados_pessoas, base_notas_fiscais):
        CHAVE = ['id_pessoa', 'tipo_divida']

        frequencia_da_pessoa = dados_pessoas.groupby(CHAVE)['cda'].nunique().to_frame().reset_index().rename(
            columns={'cda': 'num_dist_cda'})
        total_reparcelamentos_pessoa = dados_pessoas.groupby(CHAVE)[
            'quantidade_reparcelamento'].sum().to_frame().reset_index()
        valor_pessoa = dados_pessoas.groupby(CHAVE)['valor_tot', 'valor_pago'].sum().reset_index()

        # Remove as linhas duplicadas
        edificacao = dados_pessoas[['id_pessoa', 'tipo_divida', 'edificacao']].drop_duplicates()
        situacao = dados_pessoas[['id_pessoa', 'tipo_divida', 'situacao']].drop_duplicates()
        cpf_cnpj_existe = dados_pessoas[['id_pessoa', 'tipo_divida', 'cpf_cnpj_existe']].drop_duplicates()
        endereco_existe = dados_pessoas[['id_pessoa', 'tipo_divida', 'endereco_existe']].drop_duplicates()

        # Merges
        notas_edif = pd.merge(base_notas_fiscais, edificacao, on='id_pessoa', how='outer')
        situ_doc = pd.merge(situacao, cpf_cnpj_existe, on=CHAVE, how='outer')
        valor_notas_edif = pd.merge(valor_pessoa, notas_edif, on=CHAVE, how='left')
        situ_doc_endereco_existe = pd.merge(situ_doc, endereco_existe, on=CHAVE, how='left')

        # Merge Final
        pessoas = pd.merge(frequencia_da_pessoa, total_reparcelamentos_pessoa, on=CHAVE, how='left')
        pessoas = pd.merge(pessoas, valor_notas_edif, on=CHAVE, how='left')
        pessoas = pd.merge(pessoas, situ_doc_endereco_existe, on=CHAVE, how='left')

        if dados_pessoas['da_aberto'].iloc[0] == 0:
            self.pessoas_da_aberto_0 = pessoas
        elif dados_pessoas['da_aberto'].iloc[0] == 1:
            self.pessoas_da_aberto_1 = pessoas

    def variaveis_cluster(self, pessoas):
        # Seu código para calcular as variáveis do cluster...
        # Substituindo por zero os valores nulos
        pessoas['qtd_notas_2anos'] = pessoas['qtd_notas_2anos'].fillna(0)
        pessoas['edificacao'] = pessoas['edificacao'].fillna(0)
        pessoas['cpf_cnpj_existe'] = pessoas['cpf_cnpj_existe'].fillna(0)

        # REGRAS DE APLICAÇÃO REFERENTE A ACESSIBILIDADE DE COBRANÇA DO CONTRIBUINTE

        # MERCANTIL
        pessoas.loc[
            (pessoas['tipo_divida'] == 'mercantil') & (pessoas['qtd_notas_2anos'] > 0) & (
                        pessoas['situacao'] == 'ATIVO'), 'perfil_acessivel'] = 2
        pessoas.loc[
            (pessoas['tipo_divida'] == 'mercantil') & (pessoas['qtd_notas_2anos'] > 0) & (
                        pessoas['situacao'] != 'ATIVO'), 'perfil_acessivel'] = 1
        pessoas.loc[
            (pessoas['tipo_divida'] == 'mercantil') & (pessoas['qtd_notas_2anos'] == 0) & (
                        pessoas['situacao'] == 'ATIVO'), 'perfil_acessivel'] = 1
        pessoas.loc[
            (pessoas['tipo_divida'] == 'mercantil') & (pessoas['qtd_notas_2anos'] == 0) & (
                        pessoas['situacao'] != 'ATIVO'), 'perfil_acessivel'] = 0

        # IMOVEL
        pessoas.loc[(pessoas['tipo_divida'] == 'imovel') & (pessoas['edificacao'] == 1), 'perfil_acessivel'] = 2
        pessoas.loc[(pessoas['tipo_divida'] == 'imovel') & (pessoas['edificacao'] == 0), 'perfil_acessivel'] = 0  # terreno

        pessoas['situacao_cobranca'] = pessoas['perfil_acessivel'] + pessoas['cpf_cnpj_existe'] + pessoas[
            'endereco_existe']
        pessoas.loc[(pessoas['cpf_cnpj_existe'] == 0), 'situacao_cobranca'] = 0
        pessoas.loc[(pessoas['tipo_divida'] == 'mercantil') & (pessoas['perfil_acessivel'] == 0),
                    'situacao_cobranca'] = 0
        pessoas.loc[(pessoas['tipo_divida'] == 'imovel') & (pessoas['perfil_acessivel'] == 0),
                    'situacao_cobranca'] = 1

        # Faz o calculo do historico de pagamento
        pessoas.loc[(pessoas['valor_tot'].isna()) | (pessoas['valor_tot'] == 0), 'valor_tot'] = 1

        pessoas['historico_pagamento_em_valor'] = pessoas['valor_pago'] / (pessoas['valor_tot'])
        pessoas = pessoas.sort_values(by='historico_pagamento_em_valor', ascending=False)

        if pessoas['da_aberto'].iloc[0] == 0:
            self.pessoas_da_aberto_0 = pessoas
        elif pessoas['da_aberto'].iloc[0] == 1:
            self.pessoas_da_aberto_1 = pessoas

    def adicionar_classificacao_situacao_cobranca(df_aberto_0, df_aberto_1):
        # Dicionário de acessibilidade do contribuinte para cobrança
        dicionario_situacao_cobranca = {
            'situacao_cobranca': [0, 1, 2, 3, 4],
            'class_situacao_cobranca': ['INACESSÍVEL',
                                        'POUQUISSIMO ACESSÍVEL',
                                        'POUCO ACESSÍVEL',
                                        'BEM ACESSÍVEL',
                                        'MUITO ACESSÍVEL']
        }
        dicionario_situacao_cobranca = pd.DataFrame(dicionario_situacao_cobranca)

        # DA fechada
        df_aberto_0_atualizado = pd.merge(df_aberto_0,
                                          dicionario_situacao_cobranca,
                                          on="situacao_cobranca",
                                          how="left")
        # DA aberta
        df_aberto_1_atualizado = pd.merge(df_aberto_1,
                                          dicionario_situacao_cobranca,
                                          on="situacao_cobranca",
                                          how="left")

        return df_aberto_0_atualizado, df_aberto_1_atualizado

    def criar_pipeline_treino_modelo(pessoas_da_aberto_0):
        print("Criando Pipeline de treino do modelo")
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
    def criar_pipeline_predict_modelo(pessoas_da_aberto_1):
        print("Criando Pipeline de predição do modelo")
        df_pipe_predict = pessoas_da_aberto_1.query("num_dist_cda > 1")
        df_pipe_predict = df_pipe_predict[['id_pessoa',
                                               'tipo_divida',
                                               'situacao_cobranca',
                                               'num_dist_cda',
                                               'quantidade_reparcelamento',
                                               'historico_pagamento_em_valor'
                                               ]]
        df_pipe_predict = df_pipe_predict.set_index(['id_pessoa', 'tipo_divida'])
        return df_pipe_predict

if __name__ == "__main__":
    processa = ProcessaDadosContribuinte()
    processa.carregar_dados_do_s3(
        s3_bucket_name=os.getenv("S3_BUCKET_NAME"),
        s3_folder_name=os.getenv("S3_FOLDER_NAME"),
        zip_file='base_treino.zip'
    )

    try:
        print("Antes de carregar dados do S3")
        processa.carregar_dados_do_s3(s3_bucket_name="S3_BUCKET_NAME", s3_folder_name="S3_FOLDER_NAME")

        print("Antes de criar obj_valor_tot_pago_aberto")
        processa.criar_obj_valor_tot_pago_aberto(processa.base_aberta)

        print("Antes de quantidades_por_id_pessoa_e_divida")
        processa.quantidades_por_id_pessoa_e_divida(processa.pessoas_da_aberto_1, processa.base_notas_fiscais)

        print("Antes de variaveis_cluster")
        processa.variaveis_cluster(processa.pessoas_da_aberto_1)

        print("Antes de adicionar_classificacao_situacao_cobranca")
        processa.adicionar_classificacao_situacao_cobranca(processa.pessoas_da_aberto_0, processa.pessoas_da_aberto_1)

        print("Antes de criar_pipeline_treino_modelo")
        df_pipe_cluster = processa.criar_pipeline_treino_modelo(processa.pessoas_da_aberto_0)

        print("Antes de criar_pipeline_predict_modelo")
        df_pipe_predict = processa.criar_pipeline_predict_modelo(processa.pessoas_da_aberto_1)

        print("Processo finalizado com sucesso")
    except Exception as e:
        print(f"Erro: {e}")

