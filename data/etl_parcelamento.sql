SELECT 
    CASE WHEN adimovsequ IS NULL THEN concat('mercantil-', termo_da_unico, cterdiseri) ELSE concat('imovel-', termo_da_unico, cterdiseri) END as cda,
	CASE WHEN adimovsequ IS NULL THEN max(admercsequ) ELSE max(adimovsequ) END as id_pessoa,
    CASE WHEN adimovsequ IS NULL THEN 'mercantil' ELSE 'imovel' END as tipo_divida,
    sum(vl_lanc_principal) as valor_tot,
    sum(vl_arr_principal) as valor_pago,
    CASE WHEN fpropaprot = 'NAO' THEN 0 ELSE 1 END as divida_protestada,
    CASE WHEN dt_ajuizamento IS NULL THEN 0 ELSE 1 END as divida_ajuizada,
    max(extract(YEAR FROM ext_parcelamento_aquila_uni.dt_parcelamento)) as ano_inscricao_da,
    max(qtd_reparcelamentos) as quantidade_reparcelamentos
FROM ext_parcelamento_aquila_uni
WHERE termo_da_unico IS NOT NULL
GROUP BY termo_da_unico, cterdiseri, adimovsequ, divida_protestada, divida_ajuizada