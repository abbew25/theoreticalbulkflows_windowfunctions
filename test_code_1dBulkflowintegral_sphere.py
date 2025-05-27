from classy import Class
import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings
from rich.console import Console
from testcodefft import theory_tophat3d


# constants
class Settings(BaseSettings):
    c: float = Field(default=299792.458, description="Speed of light in km/s")
    omega_m: float = Field(default=0.3089, description="Matter density parameter")
    H0: float = Field(default=67.74, description="Hubble constant in km/s/Mpc")
    # zarr: npt.NDArray = Field(default=np.linspace(0.0, 2.0, 200), description="Redshift array (to get comoving distance for survey radius)")
    radius: npt.NDArray = Field(
        default=np.array(([150])) / H0 * 100.0, description="Radius of sphere in Mpc/h"
    )
    z_survey: float = Field(
        default=0.0, description="Redshift for matter power spectrum"
    )

    model_config = SettingsConfigDict(cli_parse_args=True)

    @property
    def omega_l(self):
        return 1.0 - self.omega_m


# read in power spectrum
def read_in_powerspectrum(cosmo: tuple, redshift: float):
    omega_b, omega_cdm, H0, As, m_nu = cosmo
    omega_b = omega_b * (H0 / 100.0) ** 2
    omega_cdm = omega_cdm * (H0 / 100.0) ** 2

    # Set the CLASS parameters
    M = Class()
    m1, m2, m3 = m_nu / 3, m_nu / 3, m_nu / 3
    mass_input_string = str(m1) + "," + str(m2) + "," + str(m3)

    M.set(
        {
            "omega_b": omega_b,
            "omega_cdm": omega_cdm,
            "H0": H0,
            "A_s": As,
            "N_ur": 0.00641,
            "N_ncdm": 3,
            "m_ncdm": mass_input_string,
            "tau_reio": 0.0561,
            "n_s": 0.9667,
        }
    )

    M.set({"output": "mPk, mTk", "P_k_max_1/Mpc": 100.0, "z_max_pk": 0.0})
    M.compute()

    # Get the power spectra
    # k = np.logspace(np.log10(0.7*1e-3), np.log10(3.0), 2000, base = 10.0)
    # k = np.logspace(-6, 2, 3000, base=10)*(0.6774)
    k = np.linspace(0.0, 20.0, 5000)
    Pk = np.array([M.pk_cb_lin(ki, redshift) for ki in k])
    # transfer = M.get_transfer(redshift, output_format='class')
    # print(M.sigma8())
    return k, Pk


# function used to integrate to compute proper distance from Friedmann Equation
# def E_z_inverse(z: float, Omega_m: float, Omega_L: float):
#     """
#     Compute the inverse of the E(z) function (from the first Friedmann Equation).
#     """
#     return 1.0 / (np.sqrt((Omega_m * (1.0 + z) ** 3) + (1.0 - Omega_L)))

# function that computes the proper distance as a function of redshift (dimensionless)
# def rz(red: npt.ArrayLike, c: float, H0: float, Omega_m: float, Omega_L: float):
#     """
#     Calculates the proper radial distance to an object at redshift z for the given cosmological model.
#     """
#     # try:
#     #     d_com = (c / H0) * quad(E_z_inverse, 0.0, red, epsabs=5e-5)[0]
#     #     return d_com
#     # except ValueError:
#     distances = np.zeros(len(red))
#     for i, z in enumerate(red):
#         distances[i] = (c / H0) * quad(E_z_inverse, 0.0, z, epsabs=5e-5, args=(Omega_m, Omega_L))[0]
#     return distances


# def theory_tophat3d(R: float, k: npt.NDarray):
#     res = (
#         3.0 * (np.sin(k * R) - (k * R) * np.cos(k * R)) / ((R * k) ** 3)
#     )  # / (np.pi/2)
#     res = np.where(k == 0, 1.0, res)
#     return res


if __name__ == "__main__":
    console = Console()
    console.log(
        "Computing expected bulk flow amplitude and standard deviation for spherical top hat survey geometries of different radii"
    )
    console.log("Using settings: ")
    settings = Settings()
    console.print_json(settings.model_dump_json(indent=2))

    # setting up a redshift distance interpolator
    # zeds = settings.zarr
    # rzs = rz(zeds)
    # interp_z_dz = CubicSpline(rzs, zeds)
    ks, Pk = read_in_powerspectrum(
        [0.0486, settings.omega_m - 0.0486, settings.H0, 2.07e-9, 0.0],
        settings.z_survey,
    )
    pksline = CubicSpline(ks, Pk)  # setting up spline
    radii = settings.radius
    prefactor = ((settings.H0 * settings.omega_m ** (0.55)) ** 2) / (
        2.0 * (np.pi**2)
    )  # constant

    for i, r in enumerate(radii):
        console.log(f"Computing for radius {r} Mpc/h")
        Wk_squared = theory_tophat3d(r, ks) ** 2
        func = prefactor * Pk * Wk_squared
        sigmavsquared = simpson(func, ks)
        V_mostprob = np.sqrt((2 / 3) * sigmavsquared)
        sigma = np.sqrt(sigmavsquared * (1.0 - 8.0 / (3.0 * np.pi)))
        console.log(
            f"Expected bulk flow amplitude: {V_mostprob} km/s and standard deviation: {sigma} km/s"
        )
