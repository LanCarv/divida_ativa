import os
import dotenv
import zipfile
import numpy as np
import pandas as pd


class ProcessaDadosContribuinte:
    def __init__(self, data_path, models_path):
        self.data_path = data_path
        self.models_path = models_path
        self.base_notas_fiscais = None
        self.base_conjunta = None
        self.base_fechada = None
        self.base_aberta = None
        self.da_aberto_0 = None
        self.da_aberto_1 = None
        self.pessoas_da_aberto_0 = None
        self.pessoas_da_aberto_1 = None
        self.df_pipe_cluster = None
        self.df_pipe_predict = None

    rootPath = os.path.dirname(os.getcwd())
    dataPath = os.path.join(rootPath, 'data')
    modelsPath = os.path.join(rootPath, 'models')
    env = os.path.join(rootPath, '.env')
    dotenv.load_dotenv(dotenv_path=env)

    print("Iniciando carregamento dos dados")
    zip_file = os.path.join(dataPath, 'base_treino.zip')
    z = zipfile.ZipFile(zip_file)

    def ler_bases_exportadas(self, nome_arquivo):
        with zipfile.ZipFile(os.path.join(self.data_path, 'base_treino.zip')) as z:
            z.extract(nome_arquivo)
            df = pd.read_csv(nome_arquivo, sep=',')
            os.remove(nome_arquivo)
        return df

    def carregar_dados(self):
        print("Iniciando carregamento dos dados")
        self.base_notas_fiscais = self.ler_bases_exportadas('emissao_notas.csv')
        self.base_conjunta = self.ler_bases_exportadas('imovel_mercantil.csv.csv')
        self.base_conjunta = self.base_conjunta.rename(columns={'idade_divida': 'anos_idade_da'})
        self.base_fechada = self.base_conjunta[self.base_conjunta['da_aberto'] == 0]
        self.base_aberta = self.base_conjunta[self.base_conjunta['da_aberto'] == 1]


    def criar_obj_valor_tot_pago_aberto(base_conjunta):
        valor_tot = base_conjunta.groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_tot'].sum().to_frame().reset_index()
        valor_pago = base_conjunta.groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_pago'].sum().to_frame().reset_index()

        # O que está em aberto
        valor_aberto_tot = base_conjunta[base_conjunta['da_aberto'] == 1].groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_tot'].sum().to_frame().reset_index()
        valor_aberto_pg = base_conjunta[base_conjunta['da_aberto'] == 1].groupby(['cda', 'tipo_divida', 'id_pessoa'])['valor_pago'].sum().to_frame().reset_index()
        valor_aberto = pd.merge(valor_aberto_tot, valor_aberto_pg, on = ['cda', 'tipo_divida', 'id_pessoa'], how = "left")
        valor_aberto['valor_aberto'] = valor_aberto['valor_tot'] - valor_aberto['valor_pago']
        valor_aberto.drop(columns = ['valor_tot', 'valor_pago'], inplace = True)

        # Merge de valor_tot & valor_pago
        valor_tot_pago = pd.merge(valor_tot, valor_pago, on = ['cda', 'tipo_divida', 'id_pessoa'], how = "left")
        # Merge de (valor_tot e valor_pago) & valor_aberto
        valor_tot_pago_aberto = pd.merge(valor_tot_pago, valor_aberto, on = ['cda', 'tipo_divida', 'id_pessoa'], how = "left")

        # O que a gente esperava receber: dif_tot_pago
        valor_tot_pago_aberto['dif_tot_pago'] = valor_tot_pago_aberto['valor_tot'] - valor_tot_pago_aberto['valor_pago']
        # O quanto perdeu entre o que a gente esperava receber e o que foi efetivamente pago
        valor_tot_pago_aberto['dif_tot_pago_aberto'] = round(valor_tot_pago_aberto['dif_tot_pago'] - valor_tot_pago_aberto['valor_aberto'], 5)

        valor_tot_pago_aberto.sort_values(by = 'dif_tot_pago_aberto', ascending = False)

        return valor_tot_pago_aberto

    def processar_valor_pago(self):
        # Aplicando a função nos dois universos entre fechado e aberto
        valor_pago_fechado = self.criar_obj_valor_tot_pago_aberto(self.base_fechada)
        valor_pago_aberto = self.criar_obj_valor_tot_pago_aberto(self.base_aberta)

        # Para base fechada
        valor_pago_fechado.loc[
            (valor_pago_fechado['valor_tot'].isna()) | (valor_pago_fechado['valor_tot'] == 0), 'valor_tot'] = 1
        valor_pago_fechado['perc_pago'] = np.round(valor_pago_fechado['valor_pago'] / valor_pago_fechado['valor_tot'],
                                                   5)

        # Para base aberta
        valor_pago_aberto.loc[
            (valor_pago_aberto['valor_tot'].isna()) | (valor_pago_aberto['valor_tot'] == 0), 'valor_tot'] = 1
        valor_pago_aberto['perc_pago'] = np.round(valor_pago_aberto['valor_pago'] / valor_pago_aberto['valor_tot'], 5)

        # Manipulação para DA's fechadas ( Usamos para treinar )
        print("Gerando variáveis para identificação dos grupos de contribuintes com instâncias de da_aberto == 0 FECHADAS")

        base_conjunta_aux = self.base_conjunta[
            ['cda', 'tipo_divida', 'id_pessoa', 'atividade_principal', 'situacao', 'tipo_tributo',
             'vlr_tributo', 'vlr_taxa', 'competencia_divida', 'inscricao_divida', 'arrecadacao_divida',
             'ajuizamento_divida',
             'edificacao', 'cpf_cnpj_existe', 'protesto', 'ajuizamento', 'refis', 'anos_idade_da',
             'quantidade_reparcelamento',
             'da_aberto', 'endereco_existe']]

        valor_pago_fechado = pd.merge(base_conjunta_aux, valor_pago_fechado, on=['cda', 'tipo_divida', 'id_pessoa'],
                                      how='left')
        valor_pago_fechado = valor_pago_fechado[valor_pago_fechado['da_aberto'] == 0]

        print("Gerando variáveis para identificação dos grupos de contribuintes com instâncias de da_aberto == 0")
        # DA FECHADA
        da_aberto_0 = valor_pago_aberto[['tipo_divida', 'cda', 'id_pessoa', 'situacao', 'cpf_cnpj_existe', 'edificacao',
                                         'valor_tot', 'valor_pago', 'valor_aberto', 'quantidade_reparcelamento',
                                         'anos_idade_da',
                                         'endereco_existe', 'da_aberto']]
        da_aberto_0.dropna(subset=['id_pessoa'], inplace=True)
        da_aberto_0['perc_pago'] = np.round(da_aberto_0['valor_pago'] / da_aberto_0['valor_tot'], 5)

        # DA ABERTA
        valor_pago_aberto = pd.merge(base_conjunta_aux, valor_pago_aberto, on=['cda', 'tipo_divida', 'id_pessoa'],how='left')
        valor_pago_aberto = valor_pago_aberto[valor_pago_aberto['da_aberto'] == 1]

        print("Gerando variáveis para identificação dos grupos de contribuintes com instâncias de da_aberto == 1")
        da_aberto_1 = valor_pago_aberto[
            ['tipo_divida', 'cda', 'id_pessoa', 'situacao', 'cpf_cnpj_existe', 'edificacao', 'valor_tot',
             'valor_pago', 'valor_aberto', 'quantidade_reparcelamento', 'anos_idade_da', 'endereco_existe',
             'da_aberto']]

        da_aberto_1.dropna(subset=['id_pessoa'], inplace=True)
        da_aberto_1['perc_pago'] = np.round(da_aberto_1['valor_pago'] / da_aberto_1['valor_tot'], 5)

    def criar_variaveis_cluster(self):
        print("Iniciando agrupamento para o CLUSTER")
        self.pessoas_da_aberto_0 = self._quantidades_por_id_pessoa_e_divida(self.da_aberto_0)
        self.pessoas_da_aberto_1 = self._quantidades_por_id_pessoa_e_divida(self.da_aberto_1)

        self.pessoas_da_aberto_0 = self._variaveis_cluster(self.pessoas_da_aberto_0)
        self.pessoas_da_aberto_1 = self._variaveis_cluster(self.pessoas_da_aberto_1)

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
        self.pessoas_da_aberto_0 = pd.merge(self.pessoas_da_aberto_0,
                   dicionario_situacao_cobranca,
                   on="situacao_cobranca",
                   how="left")

        # DA aberta
        self.pessoas_da_aberto_1 = pd.merge(self.pessoas_da_aberto_1,
                   dicionario_situacao_cobranca,
                   on="situacao_cobranca",
                   how="left")

        print("Criando Pipeline do modelo")

    def quantidades_por_id_pessoa_e_divida(self, dados_pessoas):
        CHAVE = ['id_pessoa', 'tipo_divida']

        frequencia_da_pessoa = dados_pessoas.groupby(CHAVE)['cda'].nunique().to_frame().reset_index().rename(columns={'cda': 'num_dist_cda'})
        total_reparcelamentos_pessoa = dados_pessoas.groupby(CHAVE)['quantidade_reparcelamento'].sum().to_frame().reset_index()
        valor_pessoa = dados_pessoas.groupby(CHAVE)['valor_tot', 'valor_pago'].sum().reset_index()

        # Remove as linhas duplicadas
        edificacao = dados_pessoas[['id_pessoa', 'tipo_divida', 'edificacao']].drop_duplicates()
        situacao = dados_pessoas[['id_pessoa', 'tipo_divida', 'situacao']].drop_duplicates()
        cpf_cnpj_existe = dados_pessoas[['id_pessoa', 'tipo_divida', 'cpf_cnpj_existe']].drop_duplicates()
        endereco_existe = dados_pessoas[['id_pessoa', 'tipo_divida', 'endereco_existe']].drop_duplicates()

        # Merges
        notas_edif = pd.merge(self.base_notas_fiscais, edificacao, on='id_pessoa', how='outer')
        situ_doc = pd.merge(situacao, cpf_cnpj_existe, on=CHAVE, how='outer')
        valor_notas_edif = pd.merge(valor_pessoa, notas_edif, on=CHAVE, how='left')
        situ_doc_endereco_existe = pd.merge(situ_doc, endereco_existe, on=CHAVE, how='left')

        # Merge Final
        pessoas = pd.merge(frequencia_da_pessoa, total_reparcelamentos_pessoa, on=CHAVE, how='left')
        pessoas = pd.merge(pessoas, valor_notas_edif, on=CHAVE, how='left')
        pessoas = pd.merge(pessoas, situ_doc_endereco_existe, on=CHAVE, how='left')
        return pessoas

pessoas_da_aberto_0 = quantidades_por_id_pessoa_e_divida(da_aberto_0, base_notas_fiscais)
pessoas_da_aberto_1 = quantidades_por_id_pessoa_e_divida(da_aberto_1, base_notas_fiscais)

    def variaveis_cluster(pessoas):
        # Substituindo por zero os valores nulos
        pessoas['qtd_notas_2anos'] = pessoas['qtd_notas_2anos'].fillna(0)
        pessoas['edificacao'] = pessoas['edificacao'].fillna(0)
        pessoas['cpf_cnpj_existe'] = pessoas['cpf_cnpj_existe'].fillna(0)

        # REGRAS DE APLICAÇÃO REFERENTE A ACESSIBILIDADE DE COBRANÇA DO CONTRIBUINTE

        # MERCANTIL
        pessoas.loc[(pessoas['tipo_divida'] == 'mercantil' ) & (pessoas['qtd_notas_2anos'] > 0) & (pessoas['situacao'] == 'ATIVO'), 'perfil_acessivel'] = 2
        pessoas.loc[(pessoas['tipo_divida'] == 'mercantil' ) & (pessoas['qtd_notas_2anos'] > 0) & (pessoas['situacao'] != 'ATIVO'), 'perfil_acessivel'] = 1
        pessoas.loc[(pessoas['tipo_divida'] == 'mercantil' ) & (pessoas['qtd_notas_2anos'] == 0) & (pessoas['situacao'] == 'ATIVO'), 'perfil_acessivel'] = 1
        pessoas.loc[(pessoas['tipo_divida'] == 'mercantil' ) & (pessoas['qtd_notas_2anos'] == 0) & (pessoas['situacao'] != 'ATIVO'), 'perfil_acessivel'] = 0

        # IMOVEL
        pessoas.loc[(pessoas['tipo_divida'] == 'imovel' ) & (pessoas['edificacao'] == 1), 'perfil_acessivel'] = 2
        pessoas.loc[(pessoas['tipo_divida'] == 'imovel' ) & (pessoas['edificacao'] == 0), 'perfil_acessivel'] = 0 # terreno

        pessoas['situacao_cobranca'] = pessoas['perfil_acessivel'] + pessoas['cpf_cnpj_existe'] + pessoas['endereco_existe']
        pessoas.loc[(pessoas['cpf_cnpj_existe'] == 0), 'situacao_cobranca'] = 0
        pessoas.loc[(pessoas['tipo_divida'] == 'mercantil' ) & (pessoas['perfil_acessivel'] == 0), 'situacao_cobranca'] = 0
        pessoas.loc[(pessoas['tipo_divida'] == 'imovel' ) & (pessoas['perfil_acessivel'] == 0), 'situacao_cobranca'] = 1

        # Faz o calculo do historico de pagamento
        pessoas.loc[(pessoas['valor_tot'].isna()) | (pessoas['valor_tot'] == 0), 'valor_tot'] = 1

        pessoas['historico_pagamento_em_valor'] = pessoas['valor_pago'] / (pessoas['valor_tot'])
        pessoas = pessoas.sort_values(by='historico_pagamento_em_valor', ascending=False)

        return pessoas

if __name__ == "__main__":
    root_path = os.path.dirname(os.getcwd())
    data_path = os.path.join(root_path, 'data')
    models_path = os.path.join(root_path, 'models')
    env = os.path.join(root_path, '.env')
    dotenv.load_dotenv(dotenv_path=env)

    processador = ProcessaDadosContribuinte(data_path, models_path)
    processador.carregar_dados()
    processador.criar_variaveis_cluster()

    # Supondo que da_aberto_0, da_aberto_1 e base_notas_fiscais estejam definidos aqui
    processador.quantidades_por_id_pessoa_e_divida(da_aberto_0, base_notas_fiscais)
    processador.quantidades_por_id_pessoa_e_divida(da_aberto_1, base_notas_fiscais)
    processador.criar_pipeline_modelo()