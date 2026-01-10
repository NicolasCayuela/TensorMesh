from dataclasses import dataclass

@dataclass
class IsotropicMaterial:
    name: str
    E: float  # Young's Modulus (Pa)
    nu: float # Poisson's Ratio
    rho: float # Density (kg/m^3)
    sigma_y: float = None # Yield Stress (Pa)
    H: float = 0.0 # Hardening Modulus (Pa)

    @property
    def lame_params(self):
        mu = self.E / (2 * (1 + self.nu))
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu, lam

Steel = IsotropicMaterial("Steel", E=210e9, nu=0.3, rho=7850, sigma_y=250e6)
Aluminum = IsotropicMaterial("Aluminum", E=70e9, nu=0.33, rho=2700, sigma_y=100e6, H=700e6) # Example H
Rubber = IsotropicMaterial("Rubber", E=10e6, nu=0.48, rho=1100)
Glass = IsotropicMaterial("Glass", E=70e9, nu=0.2, rho=2500)

