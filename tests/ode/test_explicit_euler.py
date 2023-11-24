import sys 
sys.path.append('../..')
import torch 

from torch_fem.ode import ExplicitEuler 

def test_explicit_euler():
    class MyExplicitEuler(ExplicitEuler):
        """
            du/dt = u
        """
        def forward(self, t, u):
            return u
        
    u0 = torch.rand(4)
    dt = 0.1 
    ut_gt = u0 + dt * u0 
    ut_my = MyExplicitEuler().step(0, u0, dt)
    
    assert torch.allclose(ut_gt, ut_my)

