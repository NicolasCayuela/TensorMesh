import sys 
sys.path.append('../..')
import torch 

from torch_fem.ode import ImplicitLinearEuler 

def test_implicit_euler():
    class MyImplicitLinearEuler(ImplicitLinearEuler):
        """
            du/dt = u
        """
        pass
    u0 = torch.rand(4).double()
    dt = 0.1 
    ut_gt = (1/(1-dt)) * u0
    ut_my = MyImplicitLinearEuler().step(0, u0, dt)
    
    assert torch.allclose(ut_gt, ut_my), f"expected {ut_gt}, got {ut_my}"

