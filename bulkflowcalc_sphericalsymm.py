import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
import camb
import numpy.typing as npt
from testcodefft import theory_tophat3d
from pydantic import Field
from pydantic_settings import BaseSettings
from rich.console import Console


class Settings(BaseSettings):
    c: float = Field(default=299792.458, description="Speed of light in km/s")
    H0: float = Field(default=67.74, description="Hubble constant in km/s/Mpc")
    radius_spherical_survey: float = Field(
        default=5.0, description="Radius of spherical survey in Mpc"
    )
    Omega_m: float = Field(default=0.3089, description="Matter density parameter", gt=0)

    @property
    def Omega_l(self) -> float:
        return 1.0 - self.Omega_m


# read in power spectrum
def read_in_powerspectrum(cosmo: tuple, redshift: float):
    omega_b, omega_cdm, H0, As, m_nu = cosmo

    omega_b = omega_b * (H0 / 100.0) ** 2
    omega_cdm = omega_cdm * (H0 / 100.0) ** 2

    M = camb.set_params(
        H0=H0, ombh2=omega_b, omch2=omega_cdm, As=As, ns=0.9667, tau=0.0561
    )
    M.set_matter_power(redshifts=[redshift], kmax=10.0)

    results = camb.get_results(M)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints=1000)

    k = (
        kh 
    )  
    pk = pk[0] 
    return k, pk


# function used to integrate to compute proper distance from Friedmann Equation
def E_z_inverse(z: float, settings: Settings):
    """
    Compute the inverse of the E(z) function (from the first Friedmann Equation).
    """
    return 1.0 / (
        np.sqrt((settings.Omega_m * (1.0 + z) ** 3) + (1.0 - settings.Omega_l))
    )


# function that computes the proper distance as a function of redshift (dimensionless)
def rz(red: npt.NDArray, settings: Settings):
    """
    Calculates the proper radial distance to an object at redshift z for the given cosmological model.
    """

    distances = np.zeros(len(red))
    for i, z in enumerate(red):
        distances[i] = (settings.c / settings.H0) * quad(
            E_z_inverse, 0.0, z, epsabs=5e-5, args=(settings,)
        )[0]
    return distances


def theory_3dgauss(R: float, k: npt.NDArray):
    res = (
        3.0 * (np.sin(k * R) - (k * R) * np.cos(k * R)) / ((R * k) ** 3)
    )  # / (np.pi/2)
    res = np.where(k == 0, 1.0, res)
    res = res * np.exp(-9.0 * (np.pi * k) ** 2 / ((R) ** 2))
    return res


if __name__ == "__main__":
    # setting up a redshift distance interpolator
    settings = Settings()
    zeds = np.linspace(0.0, 2.0, 200)
    rzs = rz(zeds, settings)
    interp_z_dz = CubicSpline(rzs, zeds)

    ks, Pk = read_in_powerspectrum(
        [0.0486, settings.Omega_m - 0.0486, settings.H0, 2.07e-9, 0.0], 0.0
    )
    pksline = CubicSpline(ks, Pk)  # setting up spline

    radius = settings.radius_spherical_survey / settings.H0 * 100.0  # in Mpc/h
    prefactor = ((settings.H0 * settings.Omega_m ** (0.55)) ** 2) / (
        2.0 * (np.pi**2)
    )  # constant
    Wk_squared = theory_tophat3d(radius, ks) ** 2

    func = prefactor * Pk * Wk_squared
    sigmavsquared = simpson(func, ks)
    V_mostprob = np.sqrt((2 / 3) * sigmavsquared)
    sigma = np.sqrt(sigmavsquared * (1.0 - 8.0 / (3.0 * np.pi)))

    console = Console()
    console.log(
        "Calculating most probable velocity and 1sigma standard dev. for a spherical top hat survey at z=0"
    )
    console.print(f"Radius of spherical survey: {radius:.2f} Mpc", style="bold blue")
    console.print(
        f"V_mostprob: {V_mostprob:.2f} Mpc/h, sigma: {sigma:.2f} Mpc/h",
        style="bold green",
    )
