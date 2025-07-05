import os, re, json
from pathlib import Path
from fnmatch import fnmatch    # importa o coringa estilo shell
import matplotlib.pyplot as plt
from collections import defaultdict

def coletar_acuracias(pasta_out: str, saida_json: str = "results/acuracias.json") -> None:
    padrao_final = re.compile(r"Final test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
    padrao_zero  = re.compile(r"Zero-shot CLIP's test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")

    resultados = {}

    for caminho in Path(pasta_out).glob("*.out"):
        with caminho.open("r", encoding="utf-8", errors="ignore") as f:
            texto = f.read()

        match_nome = re.match(r".*_(\d+)shots_(\d+)experts\.out", caminho.name)
        if not match_nome:
            continue  # ignora arquivos com nome fora do padrão

        num_shots = int(match_nome.group(1))
        num_experts = int(match_nome.group(2))

        resultados[caminho.name] = {
            "shots": num_shots,
            "experts": num_experts,
        }

        if fnmatch(caminho.name, "CLIP-MoLE_*_1shots_*.out"):
            if (m := padrao_zero.search(texto)):
                resultados[caminho.name]["zero_shot"] = float(m.group(1))

        if (m := padrao_final.search(texto)):
            resultados[caminho.name]["final"] = float(m.group(1))

        if not ("final" in resultados[caminho.name] or "zero_shot" in resultados[caminho.name]):
            resultados.pop(caminho.name)

    Path(saida_json).parent.mkdir(parents=True, exist_ok=True)
    with open(saida_json, "w", encoding="utf-8") as fp:
        json.dump(resultados, fp, ensure_ascii=False, indent=2)

    print(f"✅ JSON salvo em {saida_json} com {len(resultados)} arquivos.")


coletar_acuracias("logs_scripts/mole")


def plotar_graficos_por_dataset(json_path: str, pasta_saida: str = "results") -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        dados = json.load(f)

    # Ex: { "oxford_pets": {2: [(1, 72.3, 42.25), ...], 4: [(1, 75.0, 44.1), ...] } }
    datasets = defaultdict(lambda: defaultdict(list))
    padrao = re.compile(r"CLIP-MoLE_(.+)_(\d+)shots_(\d+)experts\.out")

    for nome_arquivo, resultados in dados.items():
        match = padrao.match(nome_arquivo)
        if not match:
            continue

        nome_dataset = match.group(1)
        num_shots = int(match.group(2))
        num_experts = int(match.group(3))
        final = resultados.get("final")
        zero = resultados.get("zero_shot")

        datasets[nome_dataset][num_experts].append((num_shots, final, zero))

    os.makedirs(pasta_saida, exist_ok=True)

    for dataset, por_expert in datasets.items():
        plt.figure(figsize=(8, 5))

        for num_experts, valores in sorted(por_expert.items()):
            valores.sort(key=lambda x: x[0])
            shots = [v[0] for v in valores]
            finais = [v[1] for v in valores]
            plt.plot(shots, finais, marker='o', label=f"{num_experts} experts")

        plt.title(f"Acurácia por número de shots - {dataset}")
        plt.xlabel("Número de shots")
        plt.ylabel("Acurácia (%)")
        plt.grid(True)
        plt.legend(title="Experts")
        plt.tight_layout()

        caminho_saida = os.path.join(pasta_saida, f"{dataset}.png")
        plt.savefig(caminho_saida)
        plt.close()
        print(f"📊 Gráfico salvo: {caminho_saida}")


plotar_graficos_por_dataset("results/acuracias.json")
