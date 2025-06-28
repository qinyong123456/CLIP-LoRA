import os, re, json
from pathlib import Path
from fnmatch import fnmatch    # importa o coringa estilo shell
import matplotlib.pyplot as plt
from collections import defaultdict

def coletar_acuracias(pasta_out: str, saida_json: str = "results/acuracias.json") -> None:
    padrao_final   = re.compile(r"Final test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
    padrao_zero    = re.compile(r"Zero-shot CLIP's test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")

    resultados = {}

    for caminho in Path(pasta_out).glob("*.out"):
        with caminho.open("r", encoding="utf-8", errors="ignore") as f:
            texto = f.read()

        resultados[caminho.name] = {}

        if fnmatch(caminho.name, "CLIP-LoRA_*_1shots.out"):
            if (m := padrao_zero.search(texto)):
                resultados[caminho.name]["zero_shot"] = float(m.group(1))

        if (m := padrao_final.search(texto)):
            resultados[caminho.name]["final"] = float(m.group(1))

        if not resultados[caminho.name]:
            resultados.pop(caminho.name)

    Path(saida_json).parent.mkdir(parents=True, exist_ok=True)
    with open(saida_json, "w", encoding="utf-8") as fp:
        json.dump(resultados, fp, ensure_ascii=False, indent=2)

    print(f"✅ JSON salvo em {saida_json} com {len(resultados)} arquivos.")

coletar_acuracias("logs_scripts")


def plotar_graficos_por_dataset(json_path: str, pasta_saida: str = "results") -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    # Ex: { "eurosat": [(1, 72.3, 42.25), (2, 85.1, None), ...] }
    datasets = defaultdict(list)
    padrao = re.compile(r"CLIP-LoRA_(.+)_(\d+)shots\.out")

    for nome_arquivo, resultados in dados.items():
        match = padrao.match(nome_arquivo)
        if not match:
            continue

        nome_dataset = match.group(1)
        num_shots = int(match.group(2))
        final = resultados.get("final")
        zero = resultados.get("zero_shot")

        datasets[nome_dataset].append((num_shots, final, zero))

    # Criar pasta se necessário
    os.makedirs(pasta_saida, exist_ok=True)

    # Gerar gráficos
    for dataset, valores in datasets.items():
        # ordenar pelos shots
        valores.sort(key=lambda x: x[0])
        shots = [v[0] for v in valores]
        finais = [v[1] for v in valores]
        zeros = [v[2] if v[2] is not None else None for v in valores]

        plt.figure(figsize=(8, 5))
        plt.plot(shots, finais, marker='o', label="Final Accuracy", color="blue")
        if any(zeros):
            plt.plot(shots, zeros, marker='x', linestyle='--', label="Zero-shot Accuracy", color="orange")

        plt.title(f"Acurácia por número de shots - {dataset}")
        plt.xlabel("Número de shots")
        plt.ylabel("Acurácia (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        caminho_saida = os.path.join(pasta_saida, f"{dataset}.png")
        plt.savefig(caminho_saida)
        plt.close()
        print(f"📊 Gráfico salvo: {caminho_saida}")

# Exemplo de uso:
# plotar_graficos_por_dataset("results/acuracias.json")