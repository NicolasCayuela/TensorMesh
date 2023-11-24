import sys 
sys.path.append('../..')
import torch 

from torch_fem.ode import MidPointLinearEuler 

def test_implicit_euler():
    class MyMidPointLinearEuler(MidPointLinearEuler):
        """
            du/dt = u
        """
        pass
    u0 = torch.rand(4)
    dt = 0.1 
    ut_gt = ((dt+2)/(2-dt)) * u0
    ut_my = MyMidPointLinearEuler().step(0, u0, dt)
    
    assert torch.allclose(ut_gt, ut_my), f"expected {ut_gt}, got {ut_my}"

