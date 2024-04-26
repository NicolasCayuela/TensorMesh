import torch 
from typing import Callable
from .elasticity import strain, isotropic_stress, deviatoric_stress, deviatoric_stress_norm
from .assemble_operator import eye,trace, sqrt, divide


def update_plastic_stress(gradu:torch.Tensor, 
                          strain:torch.Tensor, 
                          stress:torch.Tensor,
                          E:float|torch.Tensor = 70.0,
                          yield_stress:float|torch.Tensor = 250.0,
                          strain_fn:Callable[[torch.Tensor],torch.Tensor] = strain,
                          stress_fn:Callable[[torch.Tensor,torch.Tensor|float],torch.Tensor] = isotropic_stress,
                         )->torch.Tensor:
    """
    Parameters
    ----------
    gradu: torch.Tensor
        ND Tensor of shape [..., dim]
        gradient of displacement
    strain: torch.Tensor
        ND Tensor of shape [..., dim, dim]
        old strain tensor
    stress: torch.Tensor
        ND Tensor of shape [..., dim, dim]
        old stress tensor
    E: float|torch.Tensor
        if tensor, ND Tensor of shape [...]
        Young's modulus
    yield_stress: float|torch.Tensor
        if tensor, ND Tensor of shape [...]
        yield stress
    Returns
    -------
    torch.Tensor
        ND Tensor of shape [..., dim, dim]
    """
    # assertion
    if isinstance(E, torch.Tensor):
        assert E.shape == gradu.shape[:-1]
    if isinstance(yield_stress, torch.Tensor):
        assert yield_stress.shape == gradu.shape[:-1]
    
    # get stress trial
    delta_strain = strain_fn(gradu) - strain 
    stress_trial = stress_fn(delta_strain, E) + stress # [..., dim, dim]

    # yield function
    stress_devia = deviatoric_stress(stress_trial)
    stress_devia_norm = deviatoric_stress_norm(stress_trial)
    f_yield = stress_devia_norm - yield_stress
    f_yield = torch.clamp_min(f_yield, 0.)

    # update stress
    stress = stress_trial - divide(f_yield*stress_devia, stress_devia_norm)

    return stress
    
