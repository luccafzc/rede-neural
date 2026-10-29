"""Seleção em treino, calibração independente e previsão com abstenção."""
from pathlib import Path
import json
import math
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from detector_pt import clean_text, load_data, metrics, save_json, text_hash

SCOPE = "Experimental: redações de português L2. Não validado para autoria geral em português nativo."


def prune_near_duplicates(frames, threshold=.95):
    """Filtro lexical conservador; não identifica paráfrases ou autores."""
    joined = pd.concat([frames[s].assign(split=s) for s in ("test", "val", "train")], ignore_index=True)
    # Esta representação serve somente à auditoria; não entra no classificador.
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=30000).fit_transform(joined.text)
    parent = list(range(len(joined)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    pairs = []
    for offset in range(0, len(joined), 128):
        similarity = (X[offset:offset+128] @ X.T).tocoo()
        keep = (similarity.data >= threshold) & (similarity.row+offset < similarity.col)
        for r, c, value in zip(similarity.row[keep]+offset, similarity.col[keep], similarity.data[keep]):
            a, b = find(int(r)), find(int(c))
            parent[max(a,b)] = min(a,b)
            pairs.append({"id_a": joined.iloc[r].essay_id, "id_b": joined.iloc[c].essay_id,
                "split_a": joined.iloc[r].split, "split_b": joined.iloc[c].split,
                "similarity": float(value), "same_label": bool(joined.iloc[r].label == joined.iloc[c].label)})
    components = {}
    for i in range(len(joined)):
        components.setdefault(find(i), []).append(i)
    retained, conflicts = [], 0
    for members in components.values():
        if joined.iloc[members].label.nunique() > 1:
            conflicts += len(members)  # Ambiguidade: excluir todo o grupo.
        else:
            retained.append(min(members))
    result = joined.iloc[sorted(retained)]
    cleaned = {s: result[result.split.eq(s)].drop(columns="split").reset_index(drop=True) for s in frames}
    report = {"cosine_threshold": threshold, "near_pairs": pairs,
        "removed_by_split": {s: len(frames[s])-len(cleaned[s]) for s in frames},
        "conflicting_rows_removed": conflicts,
        "note": "Filtro lexical de auditoria, ajustado sem rótulos; não garante ausência de similaridade semântica."}
    return cleaned, report


def feature_candidates():
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
        sublinear_tf=True, lowercase=True, strip_accents=None)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=50000, sublinear_tf=True)
    hybrid = FeatureUnion([("words", clone(words)), ("chars", clone(chars))],
        transformer_weights={"words": 1., "chars": 1.})
    lr = LogisticRegression(C=1., max_iter=2000, class_weight="balanced", random_state=42)
    svm = LinearSVC(C=1., class_weight="balanced", max_iter=5000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced", random_state=42, n_jobs=-1)
    return {name: Pipeline([("features", clone(features)), ("classifier", clone(model))])
        for name, features, model in [
            ("palavras_logistica", words, lr), ("palavras_svm", words, svm),
            ("palavras_random_forest", words, rf), ("caracteres_logistica", chars, lr),
            ("hibrido_logistica", hybrid, lr), ("hibrido_svm", hybrid, svm)]}


def word_tokens(text):
    return re.findall(r"(?u)\b\w\w+\b", text.casefold())


def vocabulary_coverage(text, vocabulary):
    tokens = set(word_tokens(text))
    return sum(t in vocabulary for t in tokens)/len(tokens) if tokens else 0.


def support_check(bundle, text):
    policy = bundle["policy"]
    count = len(text.split())
    tokens = word_tokens(text)
    coverage = vocabulary_coverage(text, bundle["vocabulary"])
    reasons = []
    if count < policy["min_words"]:
        reasons.append(f"Texto curto: mínimo operacional de {policy['min_words']} palavras; recebido {count}.")
    if count > policy["max_words"]:
        reasons.append(f"Texto além do comprimento estudado: máximo {policy['max_words']} palavras; recebido {count}.")
    if coverage < policy["min_vocabulary_coverage"]:
        reasons.append("Pouca cobertura do vocabulário de treinamento.")
    if not tokens or len(set(tokens))/len(tokens) < .15:
        reasons.append("Texto vazio de palavras ou excessivamente repetitivo.")
    return {"n_words": count, "vocabulary_coverage": float(coverage), "reasons": reasons}


def conformal_pvalues(reference, probabilities):
    """p-valores de compatibilidade por classe; NÃO são probabilidades de autoria."""
    values = []
    for label in (0, 1):
        scores = np.asarray(reference[str(label)], dtype=float)
        if not len(scores):
            raise ValueError("Falta referência de uma das classes para abstenção.")
        candidate_score = 1-float(probabilities[label])
        values.append(float((1+np.sum(scores >= candidate_score))/(len(scores)+1)))
    return values


def decide(bundle, text, probabilities=None):
    support = support_check(bundle, text)
    result = {"prediction": "Inconclusivo", "label": None, "score_ai": None,
        "score_type": "não calculado fora das condições operacionais",
        "model_version": 2, "scope": SCOPE, **support}
    if support["reasons"]:
        return result
    if probabilities is None:
        probabilities = bundle["calibrated_model"].predict_proba([text])[0]
    pvalues = conformal_pvalues(bundle["conformal_reference"], probabilities)
    compatible = [k for k, p in enumerate(pvalues) if p > bundle["policy"]["alpha"]]
    result.update(score_ai=float(probabilities[1]), score_type="probabilidade calibrada no domínio L2; não comprova autoria",
        compatible_labels=[bundle["labels"][k] for k in compatible],
        compatibility_pvalues={"Humano": pvalues[0], "IA": pvalues[1]})
    if len(compatible) == 1:
        result.update(prediction=bundle["labels"][compatible[0]], label=compatible[0])
    else:
        result["reasons"] = ["Ambas as classes são compatíveis com a referência." if compatible
                             else "Nenhuma classe tem compatibilidade suficiente com a referência."]
    return result


def predict_bundle(bundle, text):
    if bundle.get("schema_version") != 2:
        raise ValueError("Use um modelo da versão 2.")
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError("Informe uma string de até 100.000 caracteres.")
    text = clean_text(text)
    if not text:
        raise ValueError("Informe um texto não vazio.")
    return decide(bundle, text)


def explain_improved(model_path, text, top_n=15):
    """Contribuições exatas para a margem linear antes da calibração."""
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 2:
        raise ValueError("Use um modelo v2.")
    base = bundle["base_model"]
    classifier = base.named_steps["classifier"]
    if not hasattr(classifier,"coef_"):
        raise ValueError("O modelo selecionado não é linear; use uma explicação apropriada a árvores.")
    features = base.named_steps["features"]
    X = features.transform([clean_text(text)]).tocsr()
    contributions = X.data * classifier.coef_[0,X.indices]
    names = features.get_feature_names_out()[X.indices]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return {"note": "Margem antes da calibração; contribuição não é evidência de autoria.",
        "intercept": float(classifier.intercept_[0]), "all_features_sum": float(contributions.sum()),
        "raw_margin": float(classifier.decision_function(X)[0]),
        "top_features": [{"feature": str(names[i]), "contribution": float(contributions[i])} for i in order]}


def selection_metrics(labels, decisions):
    y = np.asarray(labels, dtype=int)
    predicted = np.array([-1 if d["label"] is None else d["label"] for d in decisions])
    accepted = predicted >= 0
    return {"n_total": len(y), "n_decided": int(accepted.sum()), "n_inconclusive": int((~accepted).sum()),
        "decision_coverage": float(accepted.mean()),
        "human_decision_coverage": float(accepted[y==0].mean()) if np.any(y==0) else None,
        "ai_decision_coverage": float(accepted[y==1].mean()) if np.any(y==1) else None,
        "false_ai_among_all_humans": float(np.mean(predicted[y==0] == 1)) if np.any(y==0) else None,
        "false_human_among_all_ai": float(np.mean(predicted[y==1] == 0)) if np.any(y==1) else None,
        "conditional_metrics_decided_only": metrics(y[accepted], predicted[accepted]) if accepted.any() else None}


def probability_metrics(y, p):
    return {"brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.column_stack([1-p,p]), labels=[0,1]))}


def run_improved(data_dir="data", output_dir="resultados_v2"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, exact_audit = load_data(data_dir)
    known_hashes = set().union(*(set(f.text_hash) for f in frames.values()))
    frames, near_audit = prune_near_duplicates(frames)
    save_json(out/"auditoria_dados.json", {"exact": exact_audit, "near": near_audit,
        "effective_counts": {s: {"total": len(f), "human": int(f.label.eq(0).sum()), "ai": int(f.label.eq(1).sum())} for s,f in frames.items()}})
    train = frames["train"]
    # Seleção de família SOMENTE por CV do treino, com vetorizador dentro do pipeline.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    candidate_models = feature_candidates()
    rows = []
    for name, model in candidate_models.items():
        start = time.perf_counter()
        res = cross_validate(model, train.text.tolist(), train.label.to_numpy(), cv=cv,
            scoring={"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
            error_score="raise", n_jobs=1)
        rows.append({"model": name, "cv_f1_macro": float(res["test_f1_macro"].mean()),
            "cv_f1_sd": float(res["test_f1_macro"].std(ddof=1)),
            "cv_balanced_accuracy": float(res["test_balanced_accuracy"].mean()),
            "seconds": time.perf_counter()-start})
        print(f"CV {name}: {rows[-1]['cv_f1_macro']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out/"comparacao_cv_treino.csv", index=False)
    winner = max(rows, key=lambda row: row["cv_f1_macro"])["model"]
    base = clone(candidate_models[winner]).fit(train.text.tolist(), train.label.to_numpy())
    probability_cal, uncertainty_cal = train_test_split(frames["val"], test_size=.5,
        random_state=42, stratify=frames["val"].label)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(probability_cal.text.tolist(), probability_cal.label.to_numpy())
    probs_reference = calibrated.predict_proba(uncertainty_cal.text.tolist())
    y_reference = uncertainty_cal.label.to_numpy()
    reference = {str(k): (1-probs_reference[y_reference==k,k]).tolist() for k in (0,1)}
    vocabulary = set(TfidfVectorizer(min_df=2).fit(train.text.tolist()).vocabulary_)
    train_coverage = [vocabulary_coverage(t,vocabulary) for t in train.text]
    bundle = {"schema_version": 2, "base_model": base, "calibrated_model": calibrated,
        "model_name": winner, "labels": {0: "Humano", 1: "IA"},
        "policy": {"min_words": 50, "max_words": int(train.n_words.max()),
            "min_vocabulary_coverage": float(max(.5,min(.9,np.quantile(train_coverage,.01)-.05))), "alpha": .05},
        "vocabulary": vocabulary, "conformal_reference": reference, "known_data_hashes": known_hashes,
        "scope": SCOPE}
    joblib.dump(bundle, out/"modelo_selecionado.joblib", compress=3)
    for name, frame in {"train": train, "calibration_probability": probability_cal,
                         "calibration_uncertainty": uncertainty_cal, "test": frames["test"]}.items():
        frame[["essay_id","text_hash","label"]].to_csv(out/f"ids_{name}.csv",index=False)
    # O teste histórico já foi visto na v1: esta avaliação é diagnóstica, não um novo holdout.
    test = frames["test"]
    p = calibrated.predict_proba(test.text.tolist())
    decisions = [decide(bundle, t, prob) for t,prob in zip(test.text,p)]
    report = {"winner_selected_on_training_cv": winner, "selection": "3-fold stratified CV macro F1",
        "test_status": "Diagnóstico no teste histórico; validação externa independente ainda necessária.",
        "partitions": {"train": len(train), "probability_calibration": len(probability_cal),
                       "uncertainty_calibration": len(uncertainty_cal), "test": len(test)},
        "binary_default_0_5_all_test": metrics(test.label, p[:,1]>=.5, p[:,1]),
        "calibrated_probability_metrics": probability_metrics(test.label,p[:,1]),
        "selective": selection_metrics(test.label, decisions), "policy": bundle["policy"],
        "reference_per_class": {k:len(v) for k,v in reference.items()},
        "conformal_note": "Compatibilidade condicionada à classe requer dados intercambiáveis com a referência. Sem garantia para português nativo ou textos fora do domínio; abstenção não é acerto.",
        "transformer_status": "not_trained"}
    if hasattr(base,"predict_proba"):
        report["uncalibrated_probability_metrics"] = probability_metrics(test.label,base.predict_proba(test.text.tolist())[:,1])
    save_json(out/"relatorio_v2.json", report)
    pd.DataFrame([{"essay_id": row.essay_id, "text_hash": row.text_hash, "true_label": row.label,
        "n_words": row.n_words, "probability_ai": float(prob[1]), "prediction": decision["prediction"],
        "predicted_label": decision["label"], "reasons": " | ".join(decision["reasons"])}
        for row,prob,decision in zip(test.itertuples(),p,decisions)]).to_csv(out/"predicoes_teste.csv",index=False)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def evaluate_external(model_path, csv_path, output_dir):
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 2:
        raise ValueError("A avaliação externa requer um modelo v2.")
    df = pd.read_csv(csv_path)
    if not {"text","label"}.issubset(df.columns) or df.empty:
        raise ValueError("CSV deve conter linhas e colunas text,label.")
    if df.text.isna().any() or not df.label.isin([0,1]).all():
        raise ValueError("Textos obrigatórios e rótulos 0/1 são necessários.")
    df["text"] = df.text.map(clean_text)
    if df.text.str.len().eq(0).any():
        raise ValueError("Há texto vazio no CSV.")
    df["hash"] = df.text.map(text_hash)
    if df.hash.duplicated().any():
        raise ValueError("Há textos repetidos na base externa; revise antes de avaliar.")
    if set(df.hash) & bundle["known_data_hashes"]:
        raise ValueError("A base externa contém textos já presentes no experimento original.")
    decisions = [predict_bundle(bundle, t) for t in df.text]
    report = selection_metrics(df.label, decisions)
    report["scope"] = "Base fornecida; conferir origem, grupos, temas, autores e geradores manualmente. Não foi usada para ajuste."
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/"relatorio_externo.json",report)
    pd.DataFrame([{"row": i, "true_label": int(y), "text_hash": h, **d}
        for i,(y,h,d) in enumerate(zip(df.label,df.hash,decisions))]).to_json(out/"predicoes_externas.json",orient="records",force_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report
