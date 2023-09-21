from flask import Flask, request
import divida_ativa as da
import pandas as pd

app = Flask(__name__)

@app.route("/")
def root():
    return "Rota raiz da API do produto de dados do Ciclo do Credito Tributario"

@app.route("/feature_store/<data>")
def constroi_feature_store(data):
    etl = da.etl()

    if data == 'contribuinte':
        df_feature_store_contribuinte = da.feature_store.build_fs_contribuinte()
        etl.up_s3_files(df_feature_store_contribuinte, 'feature_store_contribuinte.csv')
        m = "Processo finalizado - Feature store contribuinte salva"
    
    elif data == 'divida':
        df_divida_ativa = da.feature_store.build_fs_igr()
        etl.up_s3_files(df_divida_ativa, 'feature_store_igr.csv')
        m = "Processo finalizado - Feature store divida salva"
    
    else: m = "Opcao de data store invalida. Escola contribuinte ou divida"
    
    return m

@app.route("/train_model_igr/", defaults={'n_estimators':100})
@app.route("/train_model_igr/<n_estimators>")
def treina_modelo_igr(n_estimators):
    train = da.train_model()
    testes_prod = da.unitary_tests()

    n_estimators = int(n_estimators)

    dados_divida, dados_contribuinte = train.select_data()
    df_feature_store = train.filter_enginier_igr(dados_divida=dados_divida, dados_contribuinte=dados_contribuinte)
    df_norm = train.normalize_data(dataframe=df_feature_store)
    algoritimo_regressao, r2, mse, matriz_importancia = train.train_model_igr(
        dataframe = df_norm, 
        target_feature_name = 'percentual_pago_cda', 
        n_estimators = n_estimators)

    test_value = testes_prod.validation_model_igr(modelo_regressao=algoritimo_regressao, r2=r2, mse=mse)
    
    json_results = {
        'modelo': str(algoritimo_regressao),
        'r2': r2,
        'mse': mse,
        'matriz_importancia': matriz_importancia,
        'teste_producao': test_value
    }
    
    return json_results

@app.route("/train_model_rg_contribuinte/<num_cluster>")
def treina_modelo_rg_contribuinte(num_cluster):
    
    train = da.train_model()
    testes_prod = da.unitary_tests()

    try:
        k_test = int(num_cluster)
    except Exception as a:
        m = "Insira um valor número de clusteres que faça sentido (preencha um valor numérico inteiro)"

    else:
        k_max = int(num_cluster)

        dados_divida, dados_contribuinte = train.select_data()
        df_feature_store = train.filter_enginier_rg_contribuinte(dados_contribuinte=dados_contribuinte)
        valor_ideal_k, mean_inertia, df_centroide, model_predict_contribuinte = train.train_model_rg_contribuinte(df_pipe_cluster=df_feature_store, num_cluster=k_max)
        
        test_value = testes_prod.validation_model_rg_contribuinte(model_predict_contribuinte, k_cluster=valor_ideal_k, json_centroide=df_centroide)
        
        json_results = {
            'num_clusteres': valor_ideal_k,
            'inertia': mean_inertia,
            'centroides_clusteres': df_centroide,
            'teste_producao': test_value
            }

        return json_results
    
    if m == None:
        return json_results
    else: 
        return m

@app.route('/predict/batch')
def batch_predict():
    previsoes = da.predict()
    train_models = da.train_model()
    etl = da.etl()

    modelo_igr_prod = 'modeloDA-igr-divida-v1.pkl'

    dados_totais = previsoes.batch_select_data()
    df, df_feature_store = previsoes.batch_filter_enginer(dataframe=dados_totais)
    df_normalizado = train_models.normalize_data(dataframe=df_feature_store)
    df_predict = previsoes.batch_make_predict(prod_model=modelo_igr_prod, df=df, df_normalizado=df_normalizado)
    df_up = previsoes.make_rating_divida(df_predict)
        
    etl.up_s3_files(dataframe=df_up, file_name='previsoes_igr_estoqueDA.csv')
    m = f"Previsoes em lote realizadas e salvas no banco. Dados salvos no S3"
    
    return m

@app.route('/predict/solo', methods=['POST'])
def solo_predict():
    
    predict = da.predict()

    dic_dados = ['valor_total_da', 'anos_idade_da', 'quantidade_reparcelamentos', 
                 'situacao_ativa', 'edificacao', 'tem_cpf',
                 'frequencia_da_pessoa', 'historico_pagamento_em_qtd', 'historico_pagamento_em_valor']
    dados = request.get_json()
    
    dados_input = [[dados[col] for col in dic_dados]]
    df = pd.DataFrame(dados_input, columns=dic_dados)
    
    modelo_contribuinte_prod = 'classificador-contribuinte-v1.pkl'
    df_predicao = predict.solo_filter_enginer(dataframe=df, modelo_contribuinte_prod=modelo_contribuinte_prod)

    modelo_igr_prod = 'modeloDA-igr-divida-v1.pkl'
    json_predict = predict.solo_make_predict(dataframe=df_predicao, modelo_igr_prod=modelo_igr_prod, modelo_contribuinte_prod= modelo_contribuinte_prod)
    
    return json_predict

if __name__ == "__main__":
    app.run(debug=True)