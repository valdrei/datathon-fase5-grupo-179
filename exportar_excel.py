"""
Script para exportar todas as abas do arquivo Excel para CSV.
Cada aba será salva como um arquivo CSV separado na pasta data/.
"""

import pandas as pd
import os
import sys


def exportar_abas_excel(caminho_excel: str, pasta_saida: str = "data") -> dict:
    """
    Lê todas as abas de um arquivo Excel e exporta cada uma como CSV.

    Args:
        caminho_excel: Caminho do arquivo .xlsx
        pasta_saida: Pasta onde os CSVs serão salvos

    Returns:
        Dicionário com nome da aba -> caminho do CSV gerado
    """
    if not os.path.exists(caminho_excel):
        print(f"ERRO: Arquivo não encontrado: {caminho_excel}")
        sys.exit(1)

    os.makedirs(pasta_saida, exist_ok=True)

    # Lê todas as abas do Excel
    print(f"Lendo arquivo: {caminho_excel}")
    abas = pd.read_excel(caminho_excel, sheet_name=None, engine="openpyxl")

    print(f"\nAbas encontradas: {len(abas)}")
    print("-" * 60)

    arquivos_gerados = {}

    for nome_aba, df in abas.items():
        # Limpa o nome da aba para usar como nome de arquivo
        nome_arquivo = (
            nome_aba.strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        caminho_csv = os.path.join(pasta_saida, f"{nome_arquivo}.csv")

        df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        arquivos_gerados[nome_aba] = caminho_csv

        print(f"  Aba: '{nome_aba}'")
        print(f"    -> Linhas: {len(df):,} | Colunas: {len(df.columns)}")
        print(f"    -> Salvo em: {caminho_csv}")
        print(f"    -> Colunas: {list(df.columns)}")
        print()

    print("-" * 60)
    print(f"Total: {len(arquivos_gerados)} arquivo(s) CSV exportado(s) para '{pasta_saida}/'")
    return arquivos_gerados


if __name__ == "__main__":
    ARQUIVO_EXCEL = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    exportar_abas_excel(ARQUIVO_EXCEL, pasta_saida="data")
