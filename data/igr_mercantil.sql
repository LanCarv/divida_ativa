SELECT 
    concat('mercantil-', fda.termo_da_unico,fda.séerie_da_cda) as CDA,
    'mercantil' as tipo_divida,
     count(termo_da_unico) as DEB_TOTAIS,
   sum(CASE WHEN fda.chave_dim_situacao_processo in (2, 103)
    THEN 1  ELSE 0 END) AS DEB_PAGOS, 

    sum(CASE WHEN fda.chave_dim_tributo in (5, 8, 19) THEN fda.valor_total
    WHEN fda.chave_dim_tributo in (1, 2, 7, 11, 12) THEN fda.valor_total
    ELSE fda.valor_total END) AS VALOR_TOT,
	
	sum(CASE WHEN fda.chave_dim_tributo in (1, 2, 7, 11, 12) THEN fda.valor_total
	   else 0 end ) as VLR_TRIBUTO,
	sum(CASE WHEN fda.chave_dim_tributo NOT in (1, 2, 7, 11, 12) THEN fda.valor_total
	   else 0 end ) as VLR_TAXA,

    sum(CASE WHEN fda.chave_dim_situacao_processo in (2, 103)   THEN fda.valor_total
    ELSE 0 END) AS VALOR_PAGO,

    max(fda.chave_tempo_inscricao) AS inscricao_divida,
    max(dm.chave_dim_merct) AS id_pessoa,
    max(dm.a04_situacao_mercantil) AS SITUACAO,

max(CASE WHEN dm.cpf_unico is null THEN 0   ELSE 1 END) AS CPF_EXISTE,
max(CASE WHEN fda.protestada = 'N' THEN 0  ELSE 1 END) AS DIVIDA_PROTESTADA, 
max(CASE WHEN fda.chave_tempo_ajuizamento is null THEN 0 ELSE 1 END) AS DIVIDA_AJUIZADA

FROM
    uni_divida_ativa_mercantil fda
LEFT JOIN dim_situacao_debito dsd on fda.chave_dim_situacao_debito = dsd.chave_situacao_debito
LEFT JOIN dim_situacao_processo dsp on fda.chave_dim_situacao_processo = dsp.chave_situacao_processo
LEFT JOIN dim_infracao di on fda.chave_dim_infracao = di.chave_infracao
LEFT JOIN dim_tributo dt on fda.chave_dim_tributo = dt.chave_tributo
LEFT JOIN dim_mercantil_aquila_uni dm on fda.chave_dim_mercantil = dm.chave_dim_merct
group by fda.termo_da_unico,fda.séerie_da_cda