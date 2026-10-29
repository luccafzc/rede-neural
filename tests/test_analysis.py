"""Estimativas exploratórias não substituem a evidência seletiva."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelo_v2 import analyze_bundle, analysis_windows, predict_bundle


TEXT = ('A organização da rotina pode contribuir para uma distribuição mais equilibrada das '
        'atividades profissionais e pessoais. O planejamento de tarefas, deslocamentos e períodos '
        'de descanso ajuda a identificar prioridades e reduzir conflitos de horário. Para isso, '
        'uma estratégia consiste em preparar os materiais necessários no dia anterior e estabelecer '
        'objetivos compatíveis com o tempo disponível. Também é necessário reservar espaço para '
        'imprevistos, evitando uma programação excessivamente rígida.')


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT/'resultados_v2/modelo_selecionado.joblib')

    def test_vocabulary_warning_does_not_prevent_model_estimate(self):
        bundle = dict(self.bundle, vocabulary=set())
        strict = predict_bundle(bundle, TEXT)
        actual = analyze_bundle(bundle, TEXT)
        self.assertEqual({k: actual[k] for k in strict}, strict)
        self.assertIsNone(actual['label'])
        self.assertIsNone(actual['score_ai'])
        expected = bundle['calibrated_model'].predict_proba([TEXT])[0, 1]
        self.assertAlmostEqual(actual['estimate']['score_ai'], expected)
        self.assertEqual(actual['estimate']['label'], int(expected > .5))
        self.assertTrue(any('vocabulário' in w for w in actual['analysis_warnings']))

    def test_empty_conformal_set_keeps_estimate_explicitly_separate(self):
        model = Mock()
        model.predict_proba.return_value = np.array([[.3, .7]])
        bundle = dict(self.bundle, calibrated_model=model,
                      vocabulary=set(TEXT.casefold().replace('.', '').replace(',', '').split()),
                      conformal_reference={'0': [.01]*100, '1': [.01]*100})
        result = analyze_bundle(bundle, TEXT)
        self.assertEqual(result['compatible_labels'], [])
        self.assertIsNone(result['label'])
        self.assertEqual(result['estimate']['label'], 1)
        self.assertAlmostEqual(result['estimate']['score_ai'], .7)

    def test_windows_preserve_every_word_once_and_avoid_short_tail(self):
        for count in (50, 535, 536, 1071):
            text = ' '.join(f'word{i}' for i in range(count))
            windows = analysis_windows(text, 535)
            self.assertEqual(' '.join(windows), text)
            self.assertTrue(all(50 <= len(w.split()) <= 535 for w in windows))

    def test_long_text_uses_all_windows_and_reports_disagreement(self):
        text = ' '.join(f'biblioteca{i}' for i in range(537))
        model = Mock()
        model.predict_proba.return_value = np.array([[.8, .2], [.1, .9]])
        bundle = dict(self.bundle, calibrated_model=model)
        result = analyze_bundle(bundle, text)
        estimate = result['estimate']
        self.assertIsNone(result['label'])
        self.assertEqual(estimate['n_windows'], 2)
        self.assertEqual(estimate['n_words_analyzed'], 537)
        self.assertAlmostEqual(estimate['score_ai'], (.2*269 + .9*268)/537)
        self.assertEqual(estimate['method'], 'word_weighted_windows')
        self.assertTrue(any('tendências diferentes' in w for w in result['analysis_warnings']))
        sent = model.predict_proba.call_args.args[0]
        self.assertEqual(' '.join(sent), text)

    def test_tie_does_not_arbitrarily_select_a_class(self):
        model = Mock()
        model.predict_proba.return_value = np.array([[.5, .5]])
        result = analyze_bundle(dict(self.bundle, vocabulary=set(), calibrated_model=model), TEXT)
        self.assertIsNone(result['estimate']['label'])
        self.assertEqual(result['estimate']['prediction'], 'Equilibrado')

    def test_insufficient_text_does_not_receive_fabricated_estimate(self):
        texts = ['Hoje fui ao mercado.', 'casa '*60, '!!! '*60,
                 ' '.join(chr(0x4E00+i)*3 for i in range(60))]
        for text in texts:
            with self.subTest(text=text[:30]):
                result = analyze_bundle(self.bundle, text)
                self.assertIsNone(result['estimate'])
                self.assertTrue(result['analysis_warnings'])
        for text in (' ', '<b></b>', 'a'*100001):
            with self.assertRaises(ValueError):
                analyze_bundle(self.bundle, text)


if __name__ == '__main__':
    unittest.main()
