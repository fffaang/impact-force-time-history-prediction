"""CPU prediction and local-HTTP checks; no training or published outputs are modified."""
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib import request, error

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from engine import Predictor, InputError
from app import Server, Handler


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.predictor=Predictor()

    def test_complete_example(self):
        case=self.predictor.examples[0]
        self.assertEqual(len(case['inputs']),17)
        self.assertTrue(all(v is not None for v in case['inputs'].values()))
        self.assertEqual(self.predictor.predict(case)['warnings'],[])

    def test_integral_and_batch(self):
        cases=self.predictor.examples;batch=self.predictor.predict_many(cases)
        for case,result in zip(cases,batch):
            other=self.predictor.predict(case)
            np.testing.assert_allclose(result['curve']['force_kN'],other['curve']['force_kN'],atol=1e-9)
            a=result['anchors'];t=np.asarray(result['curve']['time_ms']);f=np.asarray(result['curve']['force_kN'])
            self.assertTrue((f>=0).all() and (np.diff(t)>0).all())
            self.assertAlmostEqual(a['Astar'],a['I']/(a['Fmax']*a['Td']),places=12)
            np.testing.assert_allclose(np.trapezoid(f,t),a['I'],rtol=1e-10)

    def test_input_boundary(self):
        good=self.predictor.examples[0]['inputs']
        for key,value in [('v',0),('m',-1),('h',True),('b','nan'),('corrosion_pct',101),('Fmax',100)]:
            with self.assertRaises(InputError):self.predictor.validate(good|{key:value})

    def test_batch_csv(self):
        result=self.predictor.batch_csv(self.predictor.template_csv())
        self.assertEqual(len(result['results']),4)
        lines=self.predictor.template_csv().splitlines()
        with self.assertRaises(InputError):self.predictor.batch_csv('\n'.join([lines[0],lines[1],lines[1]]))

    def test_local_http(self):
        server=Server(('127.0.0.1',0),Handler);server.predictor=self.predictor;server.token='test-session'
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        url='http://127.0.0.1:'+str(server.server_port)
        try:
            self.assertEqual(json.load(request.urlopen(url+'/api/health'))['status'],'ready')
            body=json.dumps(self.predictor.examples[0]).encode()
            with self.assertRaises(error.HTTPError) as caught:request.urlopen(request.Request(url+'/api/predict',data=body))
            self.assertEqual(caught.exception.code,403)
            req=request.Request(url+'/api/predict',data=body,headers={'X-Session-Token':server.token,'Content-Type':'application/json'})
            result=json.load(request.urlopen(req));self.assertGreater(result['anchors']['Fmax'],0)
            req=request.Request(url+'/api/predict',data=body,headers={'X-Session-Token':server.token,'Origin':'https://example.com'})
            with self.assertRaises(error.HTTPError) as caught:request.urlopen(req)
            self.assertEqual(caught.exception.code,403)
            with self.assertRaises(error.HTTPError) as caught:request.urlopen(url+'/models/C_final.joblib')
            self.assertEqual(caught.exception.code,404)
        finally:server.shutdown();server.server_close();worker.join()


if __name__=='__main__':unittest.main()
