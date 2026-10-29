"""Verifica a interface HTTP contra o mesmo modelo usado no terminal."""
import http.client
import json
import sys
import threading
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from interface import make_server
from detector_pt import predict_text


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=make_server(0)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=5)

    def request(self,path,method='GET',body=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=15)
        try:
            conn.request(method,path,body=body,headers=headers or {})
            response=conn.getresponse()
            return response.status,response.read(),dict(response.getheaders())
        finally:conn.close()

    def analyze(self,text,extra=None):
        headers={'Content-Type':'application/json','X-Local-UI':'1'}
        headers.update(extra or {})
        return self.request('/api/analyze','POST',json.dumps({'text':text}),headers)

    def test_static_and_config(self):
        for path in ['/','/app.css','/app.js','/api/config']:
            status,body,headers=self.request(path)
            self.assertEqual(status,200)
            self.assertTrue(body)
            self.assertEqual(headers['Cache-Control'],'no-store')
        config=json.loads(self.request('/api/config')[1])
        self.assertEqual(config['policy']['min_words'],50)

    def test_predict_matches_cli_model(self):
        text='A organização da rotina pode contribuir para uma distribuição mais equilibrada das atividades profissionais e pessoais. O planejamento de tarefas, deslocamentos e períodos de descanso ajuda a identificar prioridades e reduzir conflitos de horário. Para isso, uma estratégia consiste em preparar os materiais necessários no dia anterior e estabelecer objetivos compatíveis com o tempo disponível. Também é necessário reservar espaço para imprevistos, evitando uma programação excessivamente rígida.'
        status,payload,_=self.analyze(text)
        self.assertEqual(status,200)
        result=json.loads(payload)
        strict=predict_text(ROOT/'resultados_v2/modelo_selecionado.joblib',text)
        self.assertEqual({key:result[key] for key in strict},strict)
        self.assertIsNotNone(result['estimate'])

    def test_vercel_and_local_return_same_analysis(self):
        from app import app
        text=' '.join(f'biblioteca{i}' for i in range(600))
        local_status,local,_=self.analyze(text)
        with app.test_client() as client:
            remote=client.post('/api/analyze',json={'text':text})
        self.assertEqual(local_status,200)
        self.assertEqual(remote.status_code,200)
        self.assertEqual(remote.get_json(),json.loads(local))
        self.assertEqual(remote.get_json()['estimate']['n_words_analyzed'],600)

    def test_vercel_rejects_text_empty_after_cleaning(self):
        from app import app
        with app.test_client() as client:
            result=client.post('/api/analyze',json={'text':'<b></b>'})
        self.assertEqual(result.status_code,400)

    def test_short_text_preserves_abstention(self):
        status,payload,_=self.analyze('Hoje fui ao mercado.')
        self.assertEqual(status,200)
        result=json.loads(payload)
        self.assertEqual(result['prediction'],'Inconclusivo')
        self.assertIsNone(result['score_ai'])

    def test_invalid_and_empty_text(self):
        for text in ['',None,['não é texto'],'a'*100001]:
            self.assertEqual(self.analyze(text)[0],400)
        self.assertEqual(self.request('/api/analyze','POST','bad json',{'Content-Type':'application/json','X-Local-UI':'1'})[0],400)

    def test_only_local_requests(self):
        self.assertEqual(self.analyze('texto',{'Origin':'https://example.com'})[0],403)
        self.assertEqual(self.request('/api/analyze','POST','{}',{'Content-Type':'application/json'})[0],403)
        self.assertEqual(self.request('/api/config',headers={'Host':'example.com'})[0],403)

    def test_no_project_file_exposure(self):
        for path in ['/detector_pt.py','/resultados_v2/modelo_selecionado.joblib','/../LEIA-ME.md']:
            self.assertEqual(self.request(path)[0],404)


if __name__=='__main__':unittest.main(verbosity=2)
