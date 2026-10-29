"""Diagnóstico reproduzível da tendência, usando apenas o teste histórico conhecido."""
import argparse
import importlib.metadata
from collections import Counter
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector_pt import load_data, save_json
from modelo_v2 import analyze_bundle, selection_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    frames, _ = load_data(args.data_dir)
    manifest = pd.read_csv(ROOT/'resultados_v2/ids_test.csv')
    test = frames['test']
    if set(test.text_hash) != set(manifest.text_hash):
        raise ValueError('O teste baixado não corresponde ao manifesto histórico.')
    bundle = joblib.load(ROOT/'resultados_v2/modelo_selecionado.joblib')
    results = [analyze_bundle(bundle, text) for text in test.text]
    displayed = [{'label': r['label'] if r['label'] is not None else
                  r['estimate']['label'] if r['estimate'] else None} for r in results]
    additional = [(int(y), r['estimate']['label']) for y, r in zip(test.label, results)
                  if r['label'] is None and r['estimate'] and r['estimate']['label'] is not None]
    report = {
        'status': 'Diagnóstico histórico L2; não é avaliação independente de português nativo.',
        'model_retrained': False,
        'runtime_versions': {name: importlib.metadata.version(name)
                             for name in ('scikit-learn', 'numpy', 'pandas', 'joblib')},
        'decision_rule': 'Decisão seletiva quando disponível; caso contrário, tendência pelo score em 0,5. Janelas ponderadas por palavras.',
        'strict': selection_metrics(test.label, results),
        'strict_plus_estimates': selection_metrics(test.label, displayed),
        'additional_estimates': len(additional),
        'errors_in_additional_estimates': sum(y != p for y, p in additional),
        'without_estimate': dict(Counter(w for r in results if not r['estimate'] for w in r['analysis_warnings'])),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'relatorio_analise.json', report)
    save_json(out/'predicoes_analise.json', [
        {'text_hash': h, 'true_label': int(y), 'strict_label': r['label'],
         'estimate': r['estimate'], 'analysis_warnings': r['analysis_warnings']}
        for h, y, r in zip(test.text_hash, test.label, results)])
    print(report)


if __name__ == '__main__':
    main()
