select 
    dm.chave_dim_imovel as id_pessoa,
    COUNT(db.seq_imovel_unico) as total_creditos_lancados,
    SUM(db.vlr_lan_correcao + db.vlr_lan_juros + db.vlr_lan_juros + db.vlr_lan_multa + db.vlr_lan_principal) as valor_lancado_creditos,
    SUM(
        CASE WHEN db.dt_vencimento < '2023-06-29 00:00:00' 
        THEN 1 
        ELSE 0
        END) as quantidade_debitos_adm,
    SUM(
        CASE WHEN db.dt_vencimento < '2023-06-29 00:00:00' 
        THEN db.vlr_lan_correcao + db.vlr_lan_juros + db.vlr_lan_juros + db.vlr_lan_multa + db.vlr_lan_principal
        ELSE 0
        END) as valor_lancado_debitos,
    SUM(
        CASE WHEN dt_vencimento < '2023-06-29 00:00:00'
        THEN db.vlr_arr_correcao + db.vlr_arr_juros + db.vlr_arr_multa + db.vlr_arr_principal + db.vlr_arr_tsd
        ELSE 0
        END) as valor_arrecadado_debitos
from ext_deb_adm_aquila_uni db
inner join dim_imovel_aquila_uni dm on db.seq_imovel_unico = dm.seq_imovel_unico
where date_part('year', db.dt_fat_gerador) >= 2018 and db.inscr_mercantil_unico is Null
group by dm.chave_dim_imovel