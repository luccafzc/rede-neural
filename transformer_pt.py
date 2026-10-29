"""Extensão opcional: fine-tuning, calibração e limiar, sem usar teste para ajuste."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_recall_curve
from sklearn.model_selection import train_test_split

from detector_pt import load_data, metrics, save_json, clean_text

MODELS = {
    "bertimbau": ("neuralmind/bert-base-portuguese-cased", "94d69c95f98f7d5b2a8700c420230ae10def0baa"),
    "xlm_roberta": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"),
}


def ece_binary(y, p, bins=10):
    """ECE da probabilidade da classe IA, em intervalos fixos, incluindo 0 e 1."""
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0 or y.shape != p.shape or bins < 1 or not np.isin(y, [0,1]).all():
        raise ValueError("Rótulos/probabilidades inválidos para ECE.")
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Probabilidades devem estar entre 0 e 1.")
    assignments = np.minimum((p * bins).astype(int), bins-1)
    return float(sum(np.mean(assignments == k) * abs(p[assignments == k].mean()-y[assignments == k].mean())
                     for k in range(bins) if np.any(assignments == k)))


def calibration_metrics(y, p):
    return {"brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
            "ece_binary_10_bins": ece_binary(y, p)}


def fit_temperature(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.shape != (len(labels), 2) or set(labels) != {0,1} or not np.isfinite(logits).all():
        raise ValueError("Calibração exige logits finitos (n,2) e ambas as classes.")
    result = minimize_scalar(lambda log_t: log_loss(labels, softmax(logits / np.exp(log_t), axis=1), labels=[0, 1]),
                             bounds=(math.log(.05), math.log(20)), method="bounded")
    if not result.success:
        raise RuntimeError("A calibração de temperatura não convergiu.")
    return float(np.exp(result.x))


def choose_threshold(y, p, target_precision=.90, minimum_positives=10):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not len(y) or not 0 < target_precision <= 1 or minimum_positives < 1:
        raise ValueError("Parâmetros inválidos para ajuste de limiar.")
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Rótulos/probabilidades inválidos.")
    if len(np.unique(y)) < 2:
        return {"threshold": None, "status": "Não é possível validar o limiar sem ambas as classes."}
    precision, recall, thresholds = precision_recall_curve(y, p)
    valid = [i for i, t in enumerate(thresholds)
             if precision[i] >= target_precision and np.sum(p >= t) >= minimum_positives]
    if not valid:
        return {"threshold": None, "status": "Meta não atingida no subconjunto de ajuste."}
    i = max(valid, key=lambda i: (recall[i], precision[i], thresholds[i]))
    return {"threshold": float(thresholds[i]), "tuning_precision": float(precision[i]),
            "tuning_recall": float(recall[i]), "status": "Meta observada no ajuste; não é garantia no teste."}


def train_transformer(name="bertimbau", data_dir="data", output_dir="resultados_transformer", seed=42, epochs=3, allow_cpu=False):
    if name not in MODELS or epochs < 1:
        raise ValueError("Modelo ou número de épocas inválido.")
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
        DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("Ative uma GPU no Colab/Kaggle. CPU pode ser usada com allow_cpu=True, mas será lenta.")
    set_seed(seed)
    out = Path(output_dir) / f"{name}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(data_dir)
    from modelo_v2 import prune_near_duplicates
    data, near_audit = prune_near_duplicates(data)
    audit["near_duplicate_audit"] = near_audit
    # Três subconjuntos distintos: checkpoint, temperatura e limiar.
    checkpoint_val, remainder = train_test_split(data["val"], test_size=.5, random_state=seed, stratify=data["val"].label)
    calibration, threshold_data = train_test_split(remainder, test_size=.5, random_state=seed, stratify=remainder.label)
    partitions = {"train": data["train"], "checkpoint": checkpoint_val,
                  "calibration": calibration, "threshold": threshold_data, "test": data["test"]}
    model_id, revision = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    truncation_audit = {}
    for key, frame in partitions.items():
        lengths = [len(ids) for ids in tokenizer(frame.text.tolist(), truncation=False)["input_ids"]]
        truncation_audit[key] = {"n": len(lengths), "over_256_tokens": int(np.sum(np.asarray(lengths)>256))}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
        num_labels=2, id2label={0: "Humano", 1: "IA"}, label2id={"Humano": 0, "IA": 1})
    def encode(frame):
        dataset = Dataset.from_dict({"text": frame.text.tolist(), "labels": frame.label.tolist()})
        return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=256), batched=True, remove_columns=["text"])
    encoded = {k: encode(v) for k, v in partitions.items()}
    args = TrainingArguments(output_dir=str(out / "checkpoints"), learning_rate=3e-5,
        per_device_train_batch_size=8, gradient_accumulation_steps=4,
        per_device_eval_batch_size=16, num_train_epochs=epochs,
        weight_decay=.01, warmup_ratio=.1, max_grad_norm=1., optim="adamw_torch",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        fp16=torch.cuda.is_available(), seed=seed, data_seed=seed, report_to="none",
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["checkpoint"],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: {"f1_macro": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="macro")})
    trainer.train()
    logits_cal = trainer.predict(encoded["calibration"]).predictions
    temperature = fit_temperature(logits_cal, calibration.label.to_numpy())
    logits_threshold = trainer.predict(encoded["threshold"]).predictions
    threshold_info = choose_threshold(threshold_data.label.to_numpy(), softmax(logits_threshold/temperature, axis=1)[:,1])
    # Teste consultado só após congelar checkpoint, temperatura e limiar.
    logits_test = trainer.predict(encoded["test"]).predictions
    raw = softmax(logits_test, axis=1)[:, 1]
    calibrated = softmax(logits_test/temperature, axis=1)[:, 1]
    y = data["test"].label.to_numpy()
    report = {"model": name, "model_id": model_id, "revision": revision, "seed": seed,
        "epochs": epochs, "max_length": 256, "effective_batch_size_single_gpu": 32,
        "temperature": temperature, "threshold_tuning": threshold_info,
        "partition_sizes": {k: len(v) for k, v in partitions.items()},
        "truncation_audit": truncation_audit,
        "test_status": "Teste histórico já examinado; requer validação externa independente.",
        "default_threshold_0_5": metrics(y, calibrated >= .5, calibrated),
        "calibration_before": calibration_metrics(y, raw),
        "calibration_after": calibration_metrics(y, calibrated)}
    threshold = threshold_info["threshold"]
    if threshold is not None:
        report["tuned_threshold_metrics"] = metrics(y, calibrated >= threshold, calibrated)
    trainer.save_model(str(out / "model"))
    tokenizer.save_pretrained(out / "model")
    save_json(out / "relatorio.json", report)
    save_json(out / "auditoria_dados.json", audit)
    for key, frame in partitions.items():
        frame[["essay_id", "text_hash", "label"]].to_csv(out / f"ids_{key}.csv", index=False)
    pd.DataFrame({"essay_id": data["test"].essay_id, "text_hash": data["test"].text_hash,
        "true_label": y, "probability_raw": raw, "probability_calibrated": calibrated,
        "prediction_default": (calibrated >= .5).astype(int)}).to_csv(out / "predicoes_teste.csv", index=False)
    return report


def load_predictor(model_dir, temperature=None):
    # Sem temperatura explícita, usar a do treinamento; nunca voltar silenciosamente a 1.
    if temperature is None:
        report_path = Path(model_dir).parent / "relatorio.json"
        if not report_path.exists():
            raise ValueError("Relatório de calibração ausente; informe temperatura explicitamente.")
        temperature = json.loads(report_path.read_text(encoding="utf-8"))["temperature"]
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperatura precisa ser positiva e finita.")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    def predict_proba(texts):
        if isinstance(texts, str):
            raise ValueError("Passe uma lista de textos, não uma string isolada.")
        output = []
        for start in range(0, len(texts), 16):
            batch = [clean_text(str(t)) for t in texts[start:start+16]]
            inputs = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                output.append(torch.softmax(model(**inputs).logits/temperature, dim=-1).cpu().numpy())
        return np.concatenate(output) if output else np.empty((0,2))
    return tokenizer, predict_proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="bertimbau")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="resultados_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--allow-cpu", action="store_true")
    a = parser.parse_args()
    report = train_transformer(a.model, a.data_dir, a.output_dir, a.seed, a.epochs, a.allow_cpu)
    print(report)
