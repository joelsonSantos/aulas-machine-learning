# ==========================================
# CÉLULA [25] - Processamento Base Otimizado
# ==========================================

# 1. Poda de colunas: selecione APENAS o que vai usar antes de começar
df_base_enxuto = df_consolidada.select("dt_nascimento", "co_cpf_raiz_hash")

df_maiores = (
    df_base_enxuto
    .withColumn(
        "idade",
        F.floor(F.months_between(F.current_date(), F.col("dt_nascimento")) / 12)
    )
    .filter(F.col("idade") >= 18)
    .drop("idade")
    # Reduzimos o repartition para não fragmentar os dados em 2000 partes à toa
    .repartition(180, "co_cpf_raiz_hash") 
)

# Cruza apenas as colunas necessárias para a validação
df_join_maiores = (
    df_unhash.select("co_cpf_raiz_hash", "cpf_completo", "ts_atualizacao")
    .join(df_maiores, on="co_cpf_raiz_hash", how="inner")
    .dropDuplicates(["cpf_completo"])
    .orderBy(F.col("ts_atualizacao").asc())
    .limit(90_000_000)
    .select("cpf_completo") # <--- FOCO TOTAL NA VALIDAÇÃO
)

# ==========================================
# CÉLULA [19] - Preparação para Validação
# ==========================================

# 2. O PULO DO GATO PARA DESENVOLVIMENTO:
# Guardamos o resultado limpo de 90M na memória RAM dos nós.
# Isso garante que qualquer validação abaixo morda o dado direto da memória.
df_join_maiores = df_join_maiores.cache()

# Dispara o cálculo e armazena no cache (Esse count vai rodar uma única vez)
total_maiores = df_join_maiores.count()
print(f"Total Maiores: {total_maiores}")


# ==========================================
# CÉLULA SEGUINTE - Validação do Join com Magellan
# ==========================================

# Como df_join_maiores está cacheado, esse join vai rodar na velocidade da luz
df_magellan_valida = df_cpf_magellan.select("cpf_completo")

total_in_magellan = (
    df_magellan_valida
    .join(df_join_maiores, on="cpf_completo", how="inner")
    .count()
)

print(f"Total de batimentos com Magellan: {total_in_magellan}")

# Quando terminar suas validações na sessão, limpe a memória:
# df_join_maiores.unpersist()
