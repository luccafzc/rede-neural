"""Regressões de inferência, calibração, deduplicação e integridade dos resultados."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detector_pt import clean_text, metrics, predict_text
from modelo_v2 import conformal_pvalues, decide, predict_bundle, prune_near_duplicates, selection_metrics, evaluate_external, explain_improved
from transformer_pt import choose_threshold, ece_binary, fit_temperature, load_predictor


class CoreTests(unittest.TestCase):
    def test_inline_markup_and_unicode(self):
        self.assertEqual(clean_text('  ci<u>dade</u> e ac\u0327a\u0303o  '), 'cidade e ação')
        self.assertEqual(clean_text('um<br>dois<p>três</p>'), 'um dois três')
        text=clean_text('a  <b>casa</b>\n azul')
        self.assertEqual(clean_text(text),text)

    def test_conformal_finite_sample_and_ties(self):
        ref={'0':[.1,.2,.3], '1':[.2,.4,.6]}
        self.assertEqual(conformal_pvalues(ref,[.8,.2]),[.75,.25])
        self.assertEqual(conformal_pvalues({'0':[.5], '1':[.5]},[.5,.5]),[1.,1.])
        with self.assertRaises(ValueError):
            conformal_pvalues({'0':[], '1':[.1]},[.5,.5])

    def test_inconclusive_is_not_counted_as_correct(self):
        report=selection_metrics([0,1,1],[{'label':None},{'label':1},{'label':0}])
        self.assertEqual(report['n_decided'],2)
        self.assertEqual(report['conditional_metrics_decided_only']['accuracy'],.5)
        self.assertEqual(report['false_human_among_all_ai'],.5)

    def test_temperature_validity(self):
        with self.assertRaises(ValueError):
            fit_temperature(np.zeros((2,2)),[0,0])
        with self.assertRaises(ValueError):
            ece_binary([0,1],[-.1,1.1])
        self.assertEqual(ece_binary([0,1],[0.,1.]),0.)
        self.assertIsNone(choose_threshold([0,0],[.1,.2])['threshold'])
        self.assertIsNone(choose_threshold([0,1],[.9,.1],minimum_positives=1)['threshold'])
        self.assertEqual(choose_threshold([0,1],[.1,.9],minimum_positives=1)['threshold'],.9)

    def test_missing_calibration_report(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_predictor(Path(folder)/'model')

    def test_near_duplicate_priority(self):
        common='A cidade possui uma biblioteca pública com muitos livros e uma escola perto da praça central.'
        other='As plantas precisam de água e luz para crescer, enquanto os animais procuram alimento e abrigo.'
        def frame(ident,text,label):
            return pd.DataFrame([{'essay_id':ident,'text':text,'label':label}])
        frames={'test':frame('test',common,0),'val':frame('val',other,1),'train':frame('train',common,0)}
        result,audit=prune_near_duplicates(frames)
        self.assertEqual(len(result['train']),0)
        self.assertEqual(result['test'].essay_id.tolist(),['test'])
        self.assertEqual(audit['removed_by_split']['train'],1)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=ROOT/'resultados_v2'
        cls.path=cls.folder/'modelo_selecionado.joblib'
        cls.bundle=joblib.load(cls.path)

    def test_user_examples_abstain(self):
        for text in ['Hoje fui ao mercado comprar pão e frutas para o café da manhã.',
                     'Amanhã pretendo ir na academia durante o período matutino, posteriormente vou almoçar e me arrumar para o trabalho']:
            result=predict_text(self.path,text)
            self.assertEqual(result['prediction'],'Inconclusivo')
            self.assertIsNone(result['label'])
            self.assertIsNone(result['score_ai'])

    def test_empty_oov_repetition_and_long_text(self):
        with self.assertRaises(ValueError): predict_bundle(self.bundle,' ')
        for text in ['zzzxxyyqqqq '*60, 'casa '*60, 'casa escola cidade '*250]:
            self.assertIsNone(predict_bundle(self.bundle,text)['label'])

    def test_disjoint_partitions(self):
        files=list(self.folder.glob('ids_*.csv'))
        self.assertEqual(len(files),4)
        ids,hashes=set(),set()
        for file in files:
            df=pd.read_csv(file)
            self.assertFalse(ids & set(df.essay_id))
            self.assertFalse(hashes & set(df.text_hash))
            ids.update(df.essay_id); hashes.update(df.text_hash)

    def test_saved_report_recomputed(self):
        df=pd.read_csv(self.folder/'predicoes_teste.csv')
        report=json.loads((self.folder/'relatorio_v2.json').read_text(encoding='utf-8'))
        binary=metrics(df.true_label,df.probability_ai.ge(.5),df.probability_ai)
        self.assertAlmostEqual(binary['f1_macro'],report['binary_default_0_5_all_test']['f1_macro'])
        decisions=[{'label':None if pd.isna(x) else int(x)} for x in df.predicted_label]
        computed=selection_metrics(df.true_label,decisions)
        self.assertEqual(computed,report['selective'])

    def test_cli_runs_outside_project_and_reads_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'texto.txt'
            path.write_text('Hoje fui ao mercado.',encoding='utf-8')
            run=subprocess.run([sys.executable,str(ROOT/'detector_pt.py'),'predict','--text-file',str(path)],
                cwd=folder,capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout)['prediction'],'Inconclusivo')

    def test_external_data_rejects_training_overlap(self):
        from detector_pt import text_hash
        with tempfile.TemporaryDirectory() as folder:
            csv=Path(folder)/'data.csv'
            pd.DataFrame({'text':['mesmo texto','mesmo texto'],'label':[0,1]}).to_csv(csv,index=False)
            with self.assertRaises(ValueError):
                evaluate_external(self.path,csv,Path(folder)/'out')
            pd.DataFrame({'text':['texto conhecido','texto novo'],'label':[0,1]}).to_csv(csv,index=False)
            bundle=dict(self.bundle,known_data_hashes={text_hash('texto conhecido')})
            path=Path(folder)/'fixture.joblib'
            joblib.dump(bundle,path)
            with self.assertRaisesRegex(ValueError,'já presentes'):
                evaluate_external(path,csv,Path(folder)/'out')

    def test_linear_explanation_reconstructs_margin(self):
        result=explain_improved(self.path,'Amanhã vou estudar na biblioteca e depois trabalhar.')
        self.assertAlmostEqual(result['intercept']+result['all_features_sum'],result['raw_margin'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
