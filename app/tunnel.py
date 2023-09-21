import os
import dotenv
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import psycopg2 as pg

# Cria conexão ao banco de dados e carrega os dados
def query_banco_recife(pathQuery):

    tunnel = SSHTunnelForwarder(
    (os.getenv('SSH_HOST'), int(os.getenv('SSH_PORT'))),
    ssh_username=os.getenv('SSH_US'),
    ssh_private_key=os.getenv('SSH_PKEY'),
    remote_bind_address=(os.getenv('DB_HOST'), int(os.getenv('PORT'))),
    local_bind_address=('localhost', 6543)
    )

    tunnel.start()

    query = """
    SELECT distinct
        uni_divida_ativa_imovel.termo_da_unico,
        uni_divida_ativa_imovel.chave_dim_imovel,
        dim_imovel_aquila_uni.tipo_propriedade_resp
    FROM dim_imovel_aquila_uni
    INNER JOIN uni_divida_ativa_imovel 
    ON uni_divida_ativa_imovel.chave_dim_imovel = dim_imovel_aquila_uni.chave_dim_imovel
    WHERE dim_imovel_aquila_uni.tipo_propriedade_resp IS NOT NULL
    """

    with pg.connect(
        database= os.getenv('PG_DB_NAME'),
        user=os.getenv('PG_US'),
        password = os.getenv('PG_DB_PW'),
        host=tunnel.local_bind_host,
        port=tunnel.local_bind_port,
    ) as conn:
        df = pd.read_sql_query(query, conn)

    tunnel.stop()

    return df

def cria_tabela_banco_recife(dataset, nome_tabela):

    tunnel = SSHTunnelForwarder(
    (os.getenv('SSH_HOST'), int(os.getenv('SSH_PORT'))),
    ssh_username=os.getenv('SSH_US'),
    ssh_private_key=os.getenv('SSH_PKEY'),
    remote_bind_address=(os.getenv('DB_HOST'), int(os.getenv('PORT'))),
    local_bind_address=('localhost', 6543)
    )

    tunnel.start()

    with pg.connect(
        database= os.getenv('PG_DB_NAME'),
        user=os.getenv('PG_US'),
        password = os.getenv('PG_DB_PW'),
        host=tunnel.local_bind_host,
        port=tunnel.local_bind_port,
    ) as conn:
        dataset.to_sql(nome_tabela, con=conn, if_exists='replace', index=False)

    tunnel.stop()

