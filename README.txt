# INSTRÇÕES PARA PREPARAÇÃO DO AMBIENTE:
USO DO ANACONDA COMO BASE DO PYTHON

1. Utilize a versão 3.9.13 do Python como interpretador
2. Utilize o arquivo requirements.txt para instalar as biblitoecas necessárias (comando abaixo junto da criação do ambiente do projeto).
3. Crie um novo projeto com Ambiente Virtual Python (comando no terminal cmd: conda create --name <nome do env> --file requeriments.txt).

# ARQUIVO .env:
 Criar um arquivo com o nome .env casa necessário para armazenar as credenciais de acesso (bancos de dados, etc).

# OBTER OS MODELOS .pkl:
 Os modelos, por fins de armazenamento, ficam zipados na pasta models.

# ORGANIZAÇÃO DAS PASTAS DO DIRETORIO DO PROJETO

1. app> Local onde ficam as aplicações/scripts em py/endpoints que realizam ETL, previsões, etc.
2. data> Armazena os dados utilizados nos treinamentos e arquivos com os resultados dos testes e experimentos dos modelos.
3. doc> Toda documentação relacionada ao modelo, como aprensentações, dicionários de dados, rotas de api, etc.
4. models> Local com os modelos de ML serializados são salvos após gerados.
5. raiz do diretorio> Onde ficam armazenados os notebooks do projeto.