import os, re, json
from pathlib import Path
from fnmatch import fnmatch    # importa o coringa estilo shell
import matplotlib.pyplot as plt
from collections import defaultdict

def coletar_acuracias_mole(pasta_out: str, saida_json: str = "acuracias.json") -> None:
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

def coletar_acuracias_lora(pasta_out: str, saida_json: str = "acuracias.json") -> None:
    padrao_final   = re.compile(r"Final test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
    padrao_zero    = re.compile(r"Zero-shot CLIP's test accuracy:\s*([0-9]+(?:\.[0-9]+)?)")

    resultados = {}

    for caminho in Path(pasta_out).glob("*.out"):
        with caminho.open("r", encoding="utf-8", errors="ignore") as f:
            texto = f.read()

        resultados[caminho.name] = {}

        if fnmatch(caminho.name, "CLIP-MoLE_*_1shots.out"):
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


coletar_acuracias_mole("logs_scripts/mole", "results/mole/acuracias.json")
coletar_acuracias_lora("logs_scripts/lora", "results/lora/acuracias.json")



def plotar_graficos_por_dataset_mole(json_path: str, pasta_saida: str = "results/mole") -> None:
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


def plotar_graficos_por_dataset_lora(json_path: str, pasta_saida: str = "results/lora") -> None:
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

plotar_graficos_por_dataset_lora("results/lora/acuracias.json")
plotar_graficos_por_dataset_mole("results/mole/acuracias.json")

def plotar_comparativo_mole_lora(
    json_mole: str = "results/mole/acuracias.json",
    json_lora: str = "results/lora/acuracias.json",
    pasta_saida: str = "results/molexlora"
) -> None:
    with open(json_mole, "r", encoding="utf-8") as f:
        dados_mole = json.load(f)
    with open(json_lora, "r", encoding="utf-8") as f:
        dados_lora = json.load(f)

    # Agrupar dados do MoLE: dataset -> experts -> lista de (shots, final)
    mole_datasets = defaultdict(lambda: defaultdict(list))
    padrao_mole = re.compile(r"CLIP-MoLE_(.+)_(\d+)shots_(\d+)experts\.out")

    for nome_arquivo, resultados in dados_mole.items():
        match = padrao_mole.match(nome_arquivo)
        if not match:
            continue
        dataset, shots, experts = match.group(1), int(match.group(2)), int(match.group(3))
        final = resultados.get("final")
        mole_datasets[dataset][experts].append((shots, final))

    # Agrupar dados do LoRA: dataset -> lista de (shots, final)
    lora_datasets = defaultdict(list)
    padrao_lora = re.compile(r"CLIP-LoRA_(.+)_(\d+)shots\.out")

    for nome_arquivo, resultados in dados_lora.items():
        match = padrao_lora.match(nome_arquivo)
        if not match:
            continue
        dataset, shots = match.group(1), int(match.group(2))
        final = resultados.get("final")
        lora_datasets[dataset].append((shots, final))

    os.makedirs(pasta_saida, exist_ok=True)
    datasets_em_comum = set(mole_datasets.keys()) & set(lora_datasets.keys())

    for dataset in sorted(datasets_em_comum):
        plt.figure(figsize=(8, 5))

        # Plota MoLE: uma linha por número de experts
        for experts, valores in sorted(mole_datasets[dataset].items()):
            valores.sort(key=lambda x: x[0])
            shots = [v[0] for v in valores]
            finais = [v[1] for v in valores]
            plt.plot(shots, finais, marker='o', label=f"MoLE - {experts} experts")

        # Plota LoRA: linha única
        valores_lora = sorted(lora_datasets[dataset], key=lambda x: x[0])
        shots_lora = [v[0] for v in valores_lora]
        finais_lora = [v[1] for v in valores_lora]
        plt.plot(shots_lora, finais_lora, marker='s', linestyle='--', color='black', label="LoRA")

        plt.title(f"Acurácia por número de shots - {dataset}")
        plt.xlabel("Número de shots")
        plt.ylabel("Acurácia (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        caminho_saida = os.path.join(pasta_saida, f"{dataset}.png")
        plt.savefig(caminho_saida)
        plt.close()
        print(f"📊 Gráfico comparativo salvo: {caminho_saida}")


plotar_comparativo_mole_lora()
