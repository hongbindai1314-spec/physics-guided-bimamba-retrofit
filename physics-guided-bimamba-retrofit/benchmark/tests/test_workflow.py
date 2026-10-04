import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pipeline.core import *

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.weather=weather(periods=960)
        self.X,self.y,self.info=simulate(self.weather)
    def test_seed_reproduces_data_and_physical_balance(self):
        X,y,info=simulate(weather(periods=960))
        np.testing.assert_array_equal(self.X,X);np.testing.assert_array_equal(self.y,y)
        self.assertTrue(np.all(self.info['reference_hvac_kwh']>=0))
        self.assertEqual(self.X.shape,(960,47))
        # Past load channels at t use t-1, t-4 and t-96, never y_t.
        np.testing.assert_allclose(self.X[96:,14],self.y[:-96])
        np.testing.assert_allclose(self.X[1:,12],self.y[:-1])
    def test_training_only_features_and_save_load(self):
        a=Features().fit(self.X[:600]);mean=a.mean.copy()
        future=self.X[600:].copy();future[:,0]+=100
        transformed=a.transform(future)
        np.testing.assert_array_equal(a.mean,mean)
        self.assertEqual(transformed.shape,(360,55))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'encoder.npz';a.save(p);loaded=Features.load(p)
            np.testing.assert_allclose(a.transform(self.X),loaded.transform(self.X))
    def test_loss_weight_changes_actual_training(self):
        z=Features().fit(self.X).transform(self.X)
        # Deliberately distinct physical target makes omitted physics loss detectable.
        p=self.y+3
        a=PhysicsMLP(lambda_phys=0);b=PhysicsMLP(lambda_phys=5)
        a.fit(z,self.y,p,epochs=4);b.fit(z,self.y,p,epochs=4)
        self.assertLess(np.mean((b.predict(z)-p)**2),np.mean((a.predict(z)-p)**2))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'weights.npz';b.save(path);q=PhysicsMLP.load(path)
            np.testing.assert_array_equal(b.predict(z),q.predict(z))
    def test_block_bootstrap_identical_prediction_delta(self):
        y=np.arange(1000,dtype=float);p=y+1
        r=block_bootstrap(y,p,b=500,length=96,baseline=p)
        self.assertEqual(r['RMSE_ci95'],[1.,1.])
        self.assertEqual(r['centered_null_two_sided_p'],1.)
        self.assertEqual(r['mean_squared_error_gain_vs_persistence'],0.)
    def test_reference_direction_selection_and_optimization_bounds(self):
        F=np.array([[1,2,0,1],[2,1,0,1],[3,3,1,2],[4,4,2,3]],float)
        self.assertEqual(set(fronts(F)[0]),{0,1})
        selected=select(F,2,np.random.default_rng(4));self.assertEqual(set(selected),{0,1})
        C,R,H=optimize(self.weather.iloc[:192],pop=8,generations=3)
        self.assertTrue(np.all(C>=LOW));self.assertTrue(np.all(C<=HIGH))
        self.assertTrue(np.all(np.isin(C[:,10:],[0,1])))
        self.assertEqual(len(fronts(R)[0]),len(R))
        self.assertEqual(len(H),3)

if __name__=='__main__':unittest.main()
