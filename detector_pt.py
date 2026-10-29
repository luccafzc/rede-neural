"""Experimento acadêmico humano x IA em português. Execute com --help."""
from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.svm import LinearSVC

REVISION = "69e5e82a5c0e2edd0b7765e04cb3a2c902962d8e"
DATA_URL = ("https://raw.githubusercontent.com/nxthang/"
    f"ProAID---Proficiency-Calibrated-AI-Detection/{REVISION}/"
    "PT-L2Detect%20Dataset/data/processed/")
HASHES = {
    "train": "6a1c9d00ddf9239df1b43dac76e52de478b327ba376b971f5eff92b7acb78eff",
    "val": "6541102035a594284d057afa03c06d7efd9f6e24148e70342fdeabd0b0ecd666",
    "test": "494e4642bbbad5d886675b2802188fc89bc68217c5f51361560adbb1eef37d97",
}
PROJECT_DIR = Path(__file__).resolve().parent


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def clean_text(text):
    """Preserva acentos, pontuação e caixa; retira marcação de transcrição."""
    if not isinstance(text, str):
        raise ValueError("Cada texto precisa ser uma string.")
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"</?(?:p|br|div)(?:\s[^>]*)?/?>", " ", text, flags=re.I)
    # Tags inline podem ocorrer DENTRO da palavra: ci<u>dade -> cidade.
    text = re.sub(r"</?(?:u|b|i|span|em|strong)(?:\s[^>]*)?/?>", "", text, flags=re.I)
    # Os corpora incluem trechos anotados entre <...>; preservamos o conteúdo.
    text = text.replace("<", "").replace(">", "")
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text):
    return hashlib.sha256(clean_text(text).casefold().encode("utf-8")).hexdigest()


def load_data(data_dir):
    """Baixa uma revisão fixa, verifica bytes e elimina sobreposição entre splits."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split, expected in HASHES.items():
        path = data_dir / f"{split}_full.jsonl"
        if not path.exists():
            with urlopen(DATA_URL + path.name, timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Arquivo remoto inesperado: {split}.")
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum inválido: {path}. Use a revisão documentada.")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(records)
        if not df.is_ai.map(lambda x: isinstance(x, bool)).all():
            raise ValueError("Rótulo is_ai deve ser booleano.")
        if df.essay_id.isna().any():
            raise ValueError("Registro sem identificador.")
        df["label"] = df.is_ai.astype(int)
        df["text"] = df.text.map(clean_text)
        if df.text.str.len().eq(0).any():
            raise ValueError("Texto vazio após limpeza.")
        df["text_hash"] = df.text.map(text_hash)
        df["n_words"] = df.text.str.split().str.len()
        frames[split] = df
    combined = pd.concat(frames.values(), ignore_index=True)
    for key in ("essay_id", "text_hash"):
        if combined.groupby(key).label.nunique().gt(1).any():
            raise ValueError(f"Rótulos conflitantes para {key}; revise a base.")
    audit = {"dataset_revision": REVISION, "sha256": HASHES,
             "original_counts": {k: len(v) for k, v in frames.items()}, "overlaps_before": {}}
    for a, b in itertools.combinations(frames, 2):
        audit["overlaps_before"][f"{a}/{b}"] = {
            key: len(set(frames[a][key]) & set(frames[b][key]))
            for key in ("essay_id", "text_hash")}
    # Preserva primeiro o teste, depois validação; elimina cópias do treino.
    seen_ids, seen_hashes = set(), set()
    cleaned = {}
    for split in ("test", "val", "train"):
        keep = []
        for idx, row in frames[split].iterrows():
            if row.essay_id in seen_ids or row.text_hash in seen_hashes:
                continue
            seen_ids.add(row.essay_id)
            seen_hashes.add(row.text_hash)
            keep.append(idx)
        cleaned[split] = frames[split].loc[keep].reset_index(drop=True)
        if set(cleaned[split].label.unique()) != {0, 1}:
            raise ValueError(f"O split {split} precisa conter ambas as classes.")
    audit["clean_counts"] = {s: {"total": len(df), "human": int(df.label.eq(0).sum()),
        "ai": int(df.label.eq(1).sum())} for s, df in cleaned.items()}
    audit["removed"] = {s: len(frames[s])-len(cleaned[s]) for s in frames}
    audit["human_topic_missing"] = int(combined.loc[combined.label.eq(0), "topic_num"].isna().sum())
    for a, b in itertools.combinations(cleaned, 2):
        for key in ("essay_id", "text_hash"):
            if set(cleaned[a][key]) & set(cleaned[b][key]):
                raise AssertionError("Sobreposição após limpeza.")
    return cleaned, audit


def metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else None,
        "precision_ai": float(precision_score(y, pred, zero_division=0)),
        "recall_ai": float(recall_score(y, pred, zero_division=0)),
        "f1_ai": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "false_positive_rate": float(fp/(tn+fp)) if tn+fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if score is not None and len(np.unique(y)) == 2:
        result.update(roc_auc=float(roc_auc_score(y, score)),
                      average_precision_ai=float(average_precision_score(y, score)))
    return result


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]
    return model.decision_function(X)


def candidates(seed):
    return {
        "regressao_logistica": [LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed) for c in (0.5, 1., 2.)],
        "svm_linear": [LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=5000) for c in (0.5, 1., 2.)],
        "random_forest": [RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed)],
        "dummy_majoritario": [DummyClassifier(strategy="most_frequent")],
    }


def paired_tests(predictions, y):
    comparisons = []
    for a, b in itertools.combinations(predictions, 2):
        ca, cb = predictions[a] == y, predictions[b] == y
        n_ab, n_ba = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        p = float(binomtest(n_ab, n_ab+n_ba, .5).pvalue) if n_ab+n_ba else 1.
        comparisons.append({"model_a": a, "model_b": b, "a_only_correct": n_ab,
                            "b_only_correct": n_ba, "p_raw": p})
    previous = 0.
    for rank, item in enumerate(sorted(comparisons, key=lambda x: x["p_raw"])):
        previous = max(previous, min(1., item["p_raw"]*(len(comparisons)-rank)))
        item["p_holm"] = previous
        item["significant_0_05"] = previous < .05
    return comparisons


def bootstrap_ci(y, pred, n_boot=1000, seed=42):
    """IC condicionado a este conjunto; não mede mudança de domínio."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append([accuracy_score(y[idx], pred[idx]), f1_score(y[idx], pred[idx], average="macro")])
    ci = np.quantile(values, [.025, .975], axis=0)
    return {name: {"low": float(ci[0, i]), "high": float(ci[1, i])}
            for i, name in enumerate(("accuracy", "f1_macro"))}


def run_baselines(data_dir="data", output_dir="resultados", seeds=(7, 42, 123)):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(data_dir)
    save_json(out / "auditoria_dados.json", audit)
    # Somente texto entra no classificador. Origem, modelo, nível e IDs ficam fora.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
        max_features=30000, sublinear_tf=True, lowercase=True, strip_accents=None)
    matrices = {"train": vectorizer.fit_transform(data["train"].text)}
    matrices.update({s: vectorizer.transform(data[s].text) for s in ("val", "test")})
    selected, validation = {}, []
    for seed in seeds:
        for name, options in candidates(seed).items():
            best_score, best_model = -1., None
            started = time.perf_counter()
            for model in options:
                model.fit(matrices["train"], data["train"].label)
                val_score = f1_score(data["val"].label, model.predict(matrices["val"]), average="macro")
                if val_score > best_score:
                    best_score, best_model = val_score, model
            selected[(name, seed)] = best_model
            validation.append({"model": name, "seed": seed, "validation_f1_macro": best_score,
                "selected_C": best_model.get_params().get("C"), "fit_search_seconds": time.perf_counter()-started})
    validation = pd.DataFrame(validation)
    validation.to_csv(out / "selecao_validacao.csv", index=False)
    # A escolha da família usa apenas validação; o teste entra a partir daqui.
    winner = validation.groupby("model").validation_f1_macro.mean().idxmax()
    primary_seed = 42 if 42 in seeds else seeds[0]
    results, predictions, primary_preds = [], [], {}
    y_test = data["test"].label.to_numpy()
    for (name, seed), model in selected.items():
        started = time.perf_counter()
        pred = model.predict(matrices["test"])
        elapsed = time.perf_counter()-started
        score = scores(model, matrices["test"])
        results.append({"model": name, "seed": seed,
            "predict_batch_seconds": elapsed, **metrics(y_test, pred, score)})
        for row, prediction, value in zip(data["test"].itertuples(), pred, score):
            predictions.append({"essay_id": row.essay_id, "text_hash": row.text_hash,
                "model": name, "seed": seed, "true_label": row.label, "prediction": int(prediction),
                "score_ai": float(value), "score_type": "probability_uncalibrated" if hasattr(model,"predict_proba") else "svm_margin",
                "cefr_level": row.cefr_level, "n_words": row.n_words})
        if seed == primary_seed:
            primary_preds[name] = pred
        joblib.dump({"vectorizer": vectorizer, "classifier": model, "model_name": name,
            "seed": seed, "labels": {0: "Humano", 1: "IA"}, "data_revision": REVISION}, out / f"{name}_seed{seed}.joblib")
    result_df = pd.DataFrame(results)
    result_df.to_csv(out / "metricas_teste.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predicoes_teste.csv", index=False)
    compare = {k: v for k, v in primary_preds.items() if k != "dummy_majoritario"}
    save_json(out / "mcnemar_holm.json", paired_tests(compare, y_test))
    summary = {"selected_by": "mean validation macro F1", "winner": winner,
        "primary_seed": primary_seed, "ci95_bootstrap": bootstrap_ci(y_test, primary_preds[winner]),
        "transformers_status": "not_trained", "warning": "Estudo exploratório de português L2; não comprova autoria."}
    save_json(out / "resumo.json", summary)
    joblib.dump(joblib.load(out / f"{winner}_seed{primary_seed}.joblib"), out / "modelo_selecionado.joblib")
    error_groups = []
    test = data["test"].copy()
    test["prediction"] = primary_preds[winner]
    test["length_bin"] = pd.cut(test.n_words, [0, 50, 100, 200, np.inf], labels=["1-50", "51-100", "101-200", "201+"])
    for column in ("cefr_level", "length_bin"):
        for group, frame in test.groupby(column, observed=True):
            error_groups.append({"group_type": column, "group": str(group), **metrics(frame.label, frame.prediction)})
    pd.DataFrame(error_groups).to_csv(out / "erros_por_grupo.csv", index=False)
    create_figures(result_df, primary_preds, y_test, out, primary_seed)
    print(json.dumps({"clean_counts": audit["clean_counts"], "winner": winner}, ensure_ascii=False, indent=2))
    print(result_df.loc[result_df.seed.eq(primary_seed), ["model", "accuracy", "f1_macro", "false_positive_rate"]].to_string(index=False))
    return data, result_df, summary


def create_figures(results, predictions, y, out, primary_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    subset = results[results.seed.eq(primary_seed)].sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(subset.model.str.replace("_", " "), subset.f1_macro, color="#21736b")
    for i, value in enumerate(subset.f1_macro):
        ax.text(value+.01, i, f"{value:.3f}", va="center")
    ax.set(xlim=(0, 1.1), xlabel="F1 macro no teste", title=f"Humano × IA — português L2 (seed {primary_seed})")
    fig.tight_layout()
    fig.savefig(out / "comparacao_modelos.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(predictions), figsize=(4*len(predictions), 4))
    for ax, (name, pred) in zip(np.atleast_1d(axes), predictions.items()):
        ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0,1], display_labels=["Humano", "IA"], ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(out / "matrizes_confusao.png", dpi=160)
    plt.close(fig)


def predict_text(model_path, text):
    text = clean_text(text)
    if not text:
        raise ValueError("Informe um texto não vazio.")
    bundle = joblib.load(model_path)  # Use apenas arquivos de modelo de origem confiável.
    if bundle.get("schema_version") == 2:
        from modelo_v2 import predict_bundle
        return predict_bundle(bundle, text)
    # Modelos antigos continuam legíveis, mas não devem decidir frases isoladas.
    if len(text.split()) < 50:
        return {"prediction": "Inconclusivo", "label": None, "score_ai": None,
                "reason": "Texto curto: esta versão exige pelo menos 50 palavras.",
                "model_version": "legado"}
    X = bundle["vectorizer"].transform([text])
    if X.nnz == 0:
        return {"prediction": "Inconclusivo", "reason": "Nenhum termo reconhecido no vocabulário."}
    model = bundle["classifier"]
    label = int(model.predict(X)[0])
    return {"prediction": bundle["labels"][label], "label": label,
        "score_ai": float(scores(model, X)[0]),
        "score_type": "probabilidade não calibrada" if hasattr(model,"predict_proba") else "margem SVM (não é probabilidade)",
        "scope": "Português L2; saída experimental, não comprovação de autoria."}


def explain_linear(model_path, text, top_n=12):
    bundle = joblib.load(model_path)
    model = bundle["classifier"]
    if not hasattr(model, "coef_"):
        raise ValueError("Use o modelo de regressão logística ou SVM linear.")
    vectorizer = bundle["vectorizer"]
    X = vectorizer.transform([clean_text(text)]).tocsr()
    contributions = X.data * model.coef_[0, X.indices]
    names = vectorizer.get_feature_names_out()[X.indices]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return pd.DataFrame({"termo": names[order], "contribuicao_para_IA": contributions[order]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Treinar a versão revisada com calibração e abstenção")
    train.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    train.add_argument("--output-dir", default=str(PROJECT_DIR / "resultados_v2"))
    baseline = sub.add_parser("train-baselines", help="Experimento clássico de referência")
    baseline.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    baseline.add_argument("--output-dir", default=str(PROJECT_DIR / "resultados_baselines_revisados"))
    pred = sub.add_parser("predict", help="Classificar um texto com o modelo já treinado")
    pred.add_argument("--model", default=str(PROJECT_DIR / "resultados_v2/modelo_selecionado.joblib"))
    source = pred.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", help="Arquivo .txt em UTF-8")
    evaluate = sub.add_parser("evaluate", help="Avaliar uma base externa rotulada, sem retreinar")
    evaluate.add_argument("--csv", required=True, help="CSV com colunas text,label (0 humano; 1 IA)")
    evaluate.add_argument("--model", default=str(PROJECT_DIR / "resultados_v2/modelo_selecionado.joblib"))
    evaluate.add_argument("--output-dir", default=str(PROJECT_DIR / "avaliacao_externa"))
    args = parser.parse_args()
    try:
        if args.command == "train":
            from modelo_v2 import run_improved
            run_improved(args.data_dir, args.output_dir)
        elif args.command == "train-baselines":
            run_baselines(args.data_dir, args.output_dir)
        elif args.command == "evaluate":
            from modelo_v2 import evaluate_external
            evaluate_external(args.model, args.csv, args.output_dir)
        else:
            text = Path(args.text_file).read_text(encoding="utf-8-sig") if args.text_file else args.text
            print(json.dumps(predict_text(args.model, text), ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
