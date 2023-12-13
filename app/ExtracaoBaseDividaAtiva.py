import pandas as pd
import os
import psycopg2 as pg
from sshtunnel import SSHTunnelForwarder
import boto3
from botocore.exceptions import NoCredentialsError
import io
import dotenv
import zipfile
import tempfile

class ExtracaoBaseDividaAtiva:

    def __init__(self):
        root_path = os.path.dirname(os.getcwd())
        env = os.path.join(root_path, '../.env')
        dotenv.load_dotenv(dotenv_path=env)

    def execute_sql_query(self, query, output_path):
        print("Iniciando conexão SSH")
        tunnel = SSHTunnelForwarder(
            ('3.82.169.227', int(22)),
            ssh_username='ubuntu',
            ssh_private_key=r'C:\Users\Consultor\.ssh\id_rsa',
            remote_bind_address=('192.207.206.134', int(55432)),
            local_bind_address=('localhost', 6543)
        )

        try:
            tunnel.start()
            print("Conexão SSH estabelecida com sucesso")

            print("Iniciando conexão ao banco de dados")
            with pg.connect(
                database='dbaquila',
                user='us_aquila',
                password='aquila#2023',
                host=tunnel.local_bind_host,
                port=tunnel.local_bind_port,
                connect_timeout=60
            ) as conn:

                if conn.closed == 0:  # Verifica se a conexão não está fechada
                    print("Conexão ao banco de dados estabelecida com sucesso")

                    print("Executando consulta SQL")
                    df = pd.read_sql_query(query, conn)
                    print("Consulta SQL concluída com sucesso")

                else:
                    print("A conexão com o banco de dados está fechada. Verifique o status.")

        except Exception as e:
            print(f"Erro durante a execução da consulta SQL: {e}")
            raise

        finally:
            tunnel.stop()
            print("Conexão SSH encerrada")

        # Convertendo DataFrame para CSV em formato de bytes
        csv_bytes = df.to_csv(index=False).encode()

        # Criando buffer de bytes
        csv_buffer = io.BytesIO(csv_bytes)

        return csv_buffer

    def upload_to_s3(self, csv_buffer, s3_bucket, s3_key):
        s3 = boto3.resource(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name='us-east-1'
        )

        try:
            bucket = s3.Bucket(s3_bucket)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(csv_buffer.getvalue())
                temp_file_path = temp_file.name
            # Carregando o arquivo local no S3
            bucket.upload_file(
                Filename=temp_file_path,
                Key=s3_key
            )
            print(f"Arquivo enviado com sucesso para o S3. Bucket: {s3_bucket}, Key: {s3_key}")
            return True
        except NoCredentialsError as e:
            print("As credenciais do AWS não estão configuradas corretamente, erro de acesso.")
            raise e
        finally:
            os.remove(temp_file_path)

    def extrair_e_enviar(self):
        print("Iniciando Primeira consulta da base imóvel e mercantil")
        # Consulta 1
        query_1 = """select * from public.vw_da_treino"""
        csv_buffer_1 = self.execute_sql_query(query_1, 'imovel_mercantil.csv')

        print("Iniciando Segunda consulta da base de notas")
        # Consulta 2
        query_2 = """select * from public.vw_py_nfse_2anos"""
        csv_buffer_2 = self.execute_sql_query(query_2, 'emissao_notas.csv')

        # Criação do arquivo ZIP diretamente no S3
        s3_key_zip = f'{os.getenv("S3_FOLDER_NAME")}/base_treino.zip'  # Incluindo o caminho da pasta
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                print("Adicionando csv imovel_mercantil ao ZIP")
                csv_content_1 = csv_buffer_1.getvalue().decode()
                zip_file.writestr('imovel_mercantil.csv', csv_content_1)

                print("Adicionando csv emissao_notas ao ZIP")
                csv_content_2 = csv_buffer_2.getvalue().decode()
                zip_file.writestr('emissao_notas.csv', csv_content_2)

            zip_buffer.seek(0)
            self.upload_to_s3(zip_buffer, os.getenv("S3_BUCKET_NAME"), s3_key_zip)

        print("Arquivo ZIP base_treino.zip gerado e enviado para o S3")

if __name__ == "__main__":
    extracao = ExtracaoBaseDividaAtiva()
    extracao.extrair_e_enviar()
