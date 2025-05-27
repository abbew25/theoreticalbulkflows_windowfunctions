import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simpson
from scipy.fft import fftfreq, fftshift, rfftn
from enum import StrEnum
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
from loguru import logger
from rich.progress import track
from rich.console import Console

Distance = StrEnum("Distance", "UNIFORM GAUSSIAN R_POWER_MINUS_2 R_POWER_1")


class Settings(BaseSettings):
    # c: float = Field(default=299792.458, description="Speed of light in km/s")
    # omega_m: float = Field(default=0.3089, description="Matter density parameter")
    # H0: float = Field(default=67.74, description="Hubble constant in km/s/Mpc")
    distance: Distance = Field(
        default=Distance.UNIFORM, description="Distance function"
    )
    angles: list[float] = Field(
        default=[22.5, 45.0, 90.0, 180.0], description="Angles in degrees"
    )
    radius: float = Field(default=150.0, description="Radius in Mpc", gt=0)
    grid_size: int = Field(default=250, description="Grid size. Must be even.", gt=0)

    model_config = SettingsConfigDict(cli_parse_args=True)

    @property
    def box_size(self) -> float:
        return self.radius * 15

    @property
    def omega_l(self) -> float:
        return 1.0 - self.omega_m

    @field_validator("grid_size")
    @classmethod
    def check_grid_size(cls, v):
        if v % 2 != 0:
            raise ValueError("Grid size must be even.")
        return v


def number_per_vol_gaussian(x: npt.NDArray, sigma: float):
    res = np.exp(-(x**2) / ((sigma / 3) ** 2))
    return res


def number_per_vol_rpowmin2(x: npt.NDArray):
    res = 1.0 / (x**2)
    return res


def number_per_vol_rpow1(x: npt.NDArray):
    res = x
    return res


def surveywindowfunction(
    x: npt.NDArray,
    y: npt.NDArray,
    z: npt.NDArray,
    maxdist: float,
    dist: Distance,
    phicutoff: float,
):
    if dist == Distance.UNIFORM:
        R = np.sqrt(x**2 + y**2 + z**2)
        phidata = np.arccos(z / R)
        # print(np.max(phidata), np.min(phidata))
        prob = np.ones(R.shape)
        prob[phidata > phicutoff] = 0.0
        # dat = np.linspace(-rz(0.3), rz(0.3), 350)
        # prob = prob/simpson(simpson(simpson(prob,dat),dat),dat)
        prob[R > maxdist] = 0.0
        # prob = prob/np.sum(prob)
        return prob

    elif dist == Distance.GAUSSIAN:
        R = np.sqrt(x**2 + y**2 + z**2)
        phidata = np.arccos(z / R)
        prob = number_per_vol_gaussian(R, maxdist)
        prob[phidata > phicutoff] = 0.0
        # dat = np.linspace(-rz(0.3), rz(0.3), 350)
        # prob = prob/simpson(simpson(simpson(prob,dat),dat),dat)
        prob[R > maxdist] = 0.0
        # prob = prob/np.max(prob)
        # prob = prob/np.sum(prob)

        return prob

    elif dist == Distance.R_POWER_MINUS_2:
        R = np.sqrt(x**2 + y**2 + z**2)
        phidata = np.arccos(z / R)
        prob = number_per_vol_rpowmin2(R)
        prob[phidata > phicutoff] = 0.0
        prob[R > maxdist] = 0.0
        # prob = prob/np.max(prob)
        # prob = prob/np.sum(prob)

        return prob

    elif dist == Distance.R_POWER_1:
        R = np.sqrt(x**2 + y**2 + z**2)
        phidata = np.arccos(z / R)
        prob = number_per_vol_rpow1(R)
        prob[phidata > phicutoff] = 0.0
        prob[R > maxdist] = 0.0
        # prob = prob/np.max(prob)
        # prob = prob/np.sum(prob)

        return prob

    else:
        return 0


def theory_tophat3d(R, k):
    res = 3.0 * (np.sin(k * R) - (k * R) * np.cos(k * R)) / ((R * k) ** 3)
    res = np.where(k == 0, 1.0, res)
    return res


def plot_power_spectra_varygeometries(settings: Settings, console: Console):
    f, ax = plt.subplots(1, 1, figsize=(10, 6))

    # setting up window function data
    xdata = np.linspace(-settings.box_size, settings.box_size, settings.grid_size)
    ydata = xdata
    zdata = xdata
    # rdata = np.sqrt(xdata**2 + ydata**2 + zdata**2)

    # set up 3D coordinate data to get W in real space in 3D numerically
    X, Y, Z = np.array(np.meshgrid(xdata, ydata, zdata))

    # numpy fft routine
    rangex = np.max(xdata) - np.min(xdata)
    rangey = np.max(ydata) - np.min(ydata)
    rangez = np.max(zdata) - np.min(zdata)

    # set up k-space frequencies
    ksx = fftshift((fftfreq(settings.grid_size + 1, rangex / len(xdata)))) * (
        2.0 * np.pi
    )  # get k modes - factor of 2pi is so that these
    # are consistent with k = 2pi factor in normalization for k modes in matter power spectrum
    ksy = fftshift((fftfreq(settings.grid_size + 1, rangey / len(ydata)))) * (
        2.0 * np.pi
    )
    ksz = fftshift((fftfreq(settings.grid_size + 1, rangez / len(zdata)))) * (
        2.0 * np.pi
    )

    # set up the radius of the equivalent spherical volume
    volstandard = (
        4.0 * np.pi * (settings.radius**3) / 3.0
    )  # volume of sphere with this r = maxdistgal

    for i, theta in enumerate(
        track(
            settings.angles, console=console, description="Calculating power spectrum"
        )
    ):
        console.log(f"Calculating power spectrum for angle {theta}")

        maxdistgal = settings.radius

        if theta < 180.0:
            angle = np.pi * theta / 180.0
            # adjust the grid maxdistgal so the volume is constant for cones with opening angle theta < 180
            maxdistgal = np.power(
                volstandard * 3.0 / (2.0 * np.pi * (1.0 - np.cos(angle))), 1.0 / 3.0
            )

            if maxdistgal > settings.box_size:
                # print(maxdistgal, settings.box_size)
                # raise Exception(
                #     "Size of space volume is too small to fit the survey inside. "
                # )
                msg = f"Size of space volume is too small to fit the survey inside: {maxdistgal=} > {settings.box_size=}"
                logger.error(msg)
                raise (ValueError)

        wrealspace = surveywindowfunction(
            X, Y, Z, maxdistgal, settings.distance, phicutoff=theta * np.pi / 180.0
        )

        # now get the window function in Fourier space numerically
        int_over_wrealsp = simpson(simpson(simpson(wrealspace, zdata), ydata), xdata)

        # calculate 3D fourier transform - fast fourier transform - requires different normalization with (2pi)^(3/2) for using ifft() inverse
        wfourierspace = (
            rfftn(
                wrealspace,
                (
                    settings.grid_size + 1,
                    settings.grid_size + 1,
                    settings.grid_size + 1,
                ),
            )
            * (np.max(xdata) - np.min(xdata)) ** 3
            / (len(xdata) ** 3 * int_over_wrealsp)
        )

        # wfourierspace = np.concatenate(( np.conj(np.flip(wfourierspace, axis=2)), wfourierspace[:,:,1:] ), axis=2)
        wfourierspace = np.concatenate(
            (wfourierspace[:, :, :], np.conj(np.flip(wfourierspace[:, :, 1:], axis=2))),
            axis=2,
        )
        wfourierspace = fftshift(wfourierspace)

        ksv = np.logspace(-3, 2, 200, base=10)
        # ksv = np.insert(ksv, 0, 0)
        ks_cn = (ksv[1:] + ksv[0:-1]) / 2.0
        KSX, KSY, KSZ = np.meshgrid(ksx, ksy, ksz)
        kabs_reshape = np.sqrt(KSX**2 + KSY**2 + KSZ**2).reshape(-1)
        fftreshape = abs(wfourierspace.reshape(-1))

        index_arr = np.zeros((len(kabs_reshape)))
        for i, kk in enumerate(ks_cn):
            index_arr = np.where(
                np.logical_and(kabs_reshape >= ksv[i], kabs_reshape < ksv[i + 1]),
                i,
                index_arr,
            )

        binned_dat_power = (
            pd.DataFrame({"fftval": fftreshape, "index": index_arr})
            .groupby("index", as_index=False)
            .agg(sum=("fftval", "sum"), counts=("fftval", "count"))
            .reset_index()
            .assign(power=lambda x: x["sum"] / x["counts"])
            .assign(power2=lambda x: x["power"].abs() ** 2)
            .fillna(0.0)
        )

        # print(binned_dat_power['index'])
        # print(binned_dat_power)
        # print(ks_cn)

        # binned_dat_power = pd.DataFrame({"fftval": fftreshape, "index": index_arr})
        # binned_dat_power_counts = binned_dat_power.groupby(
        #   binned_dat_power["index"], as_index=False
        # ).count()
        # binned_dat_power_counts = binned_dat_power_counts.rename(
        #     columns={"fftval": "counts"}
        # )

        # binned_dat_power_sum = binned_dat_power.groupby(
        #     binned_dat_power["index"], as_index=False
        # ).sum()
        # binned_dat_power_sum = binned_dat_power_sum.rename(columns={"fftval": "sum"})

        # binned_dat_pow_grouped = binned_dat_power_sum.merge(
        #     binned_dat_power_counts, on="index", how="outer"
        # )

        # del binned_dat_power, binned_dat_power_counts, binned_dat_power_sum
        # binned_dat_pow_grouped["power"] = (
        #     binned_dat_pow_grouped["sum"] / binned_dat_pow_grouped["counts"]
        # )

        # binned_dat_pow_grouped["power"] = np.where(
        #     np.isnan(binned_dat_pow_grouped["power"].to_numpy()),
        #     0.0,
        #     binned_dat_pow_grouped["power"].to_numpy(),
        # )

        # print(binned_dat_pow_grouped)

        # print(ks_cn[binned_dat_power["index"]], len(ks_cn[binned_dat_power["index"]]))
        # print(binned_dat_power["power2"], len(binned_dat_power["power2"]))

        ax.semilogx(
            ks_cn[binned_dat_power["index"].to_numpy(dtype=int)],
            binned_dat_power["power2"],
            label=r"$\theta$ = %.2f" % theta,
        )

    # plt.semilogx(ks_cn, abs(theory_tophat3d(radius, ks_cn))**2, label='theory', linestyle='-.')
    ax.legend()
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P(k)$")
    ax.set_title(r"Power spectrum for geometry %s" % settings.distance)
    plt.tight_layout()
    console.log("Plotting power spectrum for geometry %s" % settings.distance)
    return f


if __name__ == "__main__":
    settings = Settings()
    console = Console()
    console.log("Running with settings:")
    console.print_json(settings.model_dump_json(indent=2))

    figure = plot_power_spectra_varygeometries(settings, console)

    outputpath = (
        "output/plot_powerspectrum_forsurveygeometry_%s.png" % settings.distance
    )
    figure.savefig(outputpath)
    console.log(f"[bold green]Saved plot to {outputpath}[/bold green]")
