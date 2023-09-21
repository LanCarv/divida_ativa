select
	dm.chave_dim_merct as id_pessoa,
	count(nf.inscr_merc_pres_uni) as qtd_notas_2anos
from ext_nfse_aquila_uni nf
inner join dim_mercantil_aquila_uni dm on dm.insc_merct_unico = nf.inscr_merc_pres_uni
where nf.ano > 2020
group by dm.chave_dim_merct