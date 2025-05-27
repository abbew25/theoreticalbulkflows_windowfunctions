import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.fft import (
    fftfreq,
    fftshift,
    rfftn,
)  # using rfftn for speed up / decrease memory in fast fourier transform of real func.
from rich.console import Console
from bulkflowcalc_sphericalsymm import read_in_powerspectrum, rz
from plot_powerspectrum_forsurveygeometry import surveywindowfunction, Distance
import numpy.typing as npt
from pydantic import field_validator, Field
from pydantic_settings import SettingsConfigDict, BaseSettings


mathnames = {
    Distance.UNIFORM: r"$n(r) = $ constant",
    Distance.GAUSSIAN: r"$n(r) \propto e^{-r^2 / (R/3)^2}$",
    Distance.R_POWER_MINUS_2: r"$n(r) \propto r^{-2}$",
    Distance.R_POWER_1: r"$n(r) \propto r$",
}


class Settings(BaseSettings):
    c: float = Field(default=299792.458, description="Speed of light in km/s")
    Omega_m: float = Field(default=0.3089, description="Matter density parameter")
    H0: float = Field(default=67.74, description="Hubble constant in km/s/Mpc")
    distance: Distance = Field(
        default=Distance.UNIFORM, description="Distance function"
    )
    angles: list[float] = Field(
        default=[22.5, 45.0, 90.0, 180.0], description="Angles in degrees"
    )
    radius: list[float] = Field(default=[150.0], description="Radius in Mpc")
    grid_size: int = Field(default=250, description="Grid size. Must be even.")

    model_config = SettingsConfigDict(cli_parse_args=True)

    subplots_by_angle: bool = Field(
        default=True, description="Subplots by angle or radius"
    )
    calculate: bool = Field(default=False, description="Calculate or use saved data")

    @property
    def Omega_l(self) -> float:
        return 1.0 - self.Omega_m

    @field_validator("grid_size")
    @classmethod
    def check_grid_size(cls, v):
        if v % 2 != 0:
            raise ValueError("Grid size must be even.")
        return v


def calculate_Vp_sigma(
    ksx: npt.NDArray,
    ksy: npt.NDArray,
    ksz: npt.NDArray,
    wk: npt.NDArray,
    settings: Settings,
):
    prefactor = ((settings.H0 * settings.Omega_m ** (0.55)) ** 2) / (
        (2.0 * np.pi) ** 3
    )  # constant

    dkx = abs(ksx[1:] - ksx[0:-1])  # + deltax
    dky = abs(ksy[1:] - ksy[0:-1])  # + deltay
    dkz = abs(ksz[1:] - ksz[0:-1])  # + deltaz

    ksx = ksx[0:-1]
    ksy = ksy[0:-1]
    ksz = ksz[0:-1]

    sigma_v_2 = None

    for kkxx in np.arange(len(ksx)):
        vals_ky = np.zeros((len(ksy)))

        for kkyy in np.arange(len(ksy)):
            vals_kz = np.zeros((len(ksz)))

            k_abs = np.sqrt(ksx[kkxx] ** 2 + ksy[kkyy] ** 2 + ksz[:] ** 2)

            Pkmm = pksline(k_abs)

            wkval = wk[kkxx][kkyy][:]

            kabsnotzero = np.where(k_abs != 0.0)
            vals_kz[kabsnotzero] = (
                prefactor
                * (abs(wkval[kabsnotzero]) ** 2)
                * Pkmm[kabsnotzero]
                / (k_abs[kabsnotzero] ** 2)
            ) * dkz[kabsnotzero]

            vals_kz = vals_kz * dky[kkyy]

            vals_ky[kkyy] = np.sum(vals_kz)

        vals_ky = vals_ky * dkx[kkxx]

        sigma_v_2 += np.sum(vals_ky)

    # get V
    V_mostprob = np.sqrt((2 / 3) * sigma_v_2)
    sigma = np.sqrt(sigma_v_2 * (1.0 - 8.0 / (3.0 * np.pi)))
    return V_mostprob, sigma


def calculate_Vp_sigma_v2(
    ksx: npt.NDArray,
    ksy: npt.NDArray,
    ksz: npt.NDArray,
    wk: npt.NDArray,
    settings: Settings,
    pkspline: CubicSpline,
):
    prefactor = ((settings.H0 * settings.Omega_m ** (0.55)) ** 2) / (
        (2.0 * np.pi) ** 3
    )  # constant

    KSX, KSY, KSZ = np.meshgrid(ksx, ksy, ksz)

    abs_K = np.sqrt(KSX**2 + KSY**2 + KSZ**2)

    Pkmm = pkspline(abs_K)

    integrand = prefactor * (abs(wk) ** 2) * Pkmm * (1 / abs_K**2)

    integrand[abs_K == 0] = 0.0

    sigma_v_squared = simpson(simpson(simpson(integrand, ksz), ksy), ksx)

    V_mostprob = np.sqrt((2 / 3) * sigma_v_squared)
    sigma = np.sqrt(sigma_v_squared * (1.0 - 8.0 / (3.0 * np.pi)))
    return V_mostprob, sigma


if __name__ == "__main__":
    # setting up a redsift distance interpolator
    settings = Settings(
        angles=[25, 50, 100, 180.0], radius=[150.0], grid_size=400, calculate=True
    )
    console = Console()
    zeds = np.linspace(0.0, 2.0, 200)
    rzs = rz(zeds, settings)
    interp_z_dz = CubicSpline(rzs, zeds)

    grid_size_orig = settings.grid_size
    rad_distributions = [Distance.UNIFORM, Distance.GAUSSIAN]
    rad_mathname = [mathnames[rad] for rad in rad_distributions]

    Vp_to_plot = np.zeros(
        (len(settings.angles), len(settings.radius), len(rad_distributions))
    )
    sigma = np.zeros(
        (len(settings.angles), len(settings.radius), len(rad_distributions))
    )

    # -----------------------------------------------------------------------------------------------------#
    # main script below
    # -----------------------------------------------------------------------------------------------------#

    USE_SAVED = not (settings.calculate)
    filename_note = "_grid" + str(settings.grid_size)

    global ks, Pk, pksline
    if settings.calculate:
        ks, Pk = read_in_powerspectrum(
            [0.0486, settings.Omega_m - 0.0486, settings.H0, 2.07e-9, 0.0], 0.0
        )
        pksline = CubicSpline(ks, Pk, extrapolate=False)  # setting up spline

        for i, theta in enumerate(settings.angles):
            for j, R in enumerate(settings.radius):  # ([radius[0]]):
                boxsizehalf = np.max(
                    [settings.radius[j] * 10.0, np.max(settings.radius) * 2]
                )
                if np.max(settings.radius) > settings.radius[j] * 10.0:
                    grid_size = 700
                    console.print("set grid size = 700")
                else:
                    grid_size = grid_size_orig

                # setting up window function data
                xdata = np.linspace(-boxsizehalf, boxsizehalf, grid_size)
                ydata = xdata
                zdata = xdata
                rdata = np.sqrt(xdata**2 + ydata**2 + zdata**2)

                # set up 3D coordinate data to get W in real space in 3D numerically
                X, Y, Z = np.array(np.meshgrid(xdata, ydata, zdata))

                # numpy fft routine
                rangex = np.max(xdata) - np.min(xdata)
                rangey = np.max(ydata) - np.min(ydata)
                rangez = np.max(zdata) - np.min(zdata)

                # set up k-space frequencies
                ksx = fftshift((fftfreq(grid_size + 1, rangex / len(xdata)))) * (
                    2.0 * np.pi
                )  # get k modes - factor of 2pi is so that these
                # are consistent with k = 2pi factor in normalization for k modes in matter power spectrum
                ksy = fftshift((fftfreq(grid_size + 1, rangey / len(ydata)))) * (
                    2.0 * np.pi
                )
                ksz = fftshift((fftfreq(grid_size + 1, rangez / len(zdata)))) * (
                    2.0 * np.pi
                )

                for k, n_ame in enumerate(rad_distributions):
                    # set up the radius of the equivalent spherical volume
                    maxdistgal = settings.radius[j]  # Mpc
                    volstandard = (
                        4.0 * np.pi * (maxdistgal**3) / 3.0
                    )  # volume of sphere with this r = maxdistgal

                    if theta < 180.0:
                        angle = np.pi * theta / 180.0
                        # adjust the grid maxdistgal so the volume is constant for cones with opening angle theta < 180
                        maxdistgal = np.power(
                            volstandard * 3.0 / (2.0 * np.pi * (1.0 - np.cos(angle))),
                            1.0 / 3.0,
                        )

                        if maxdistgal > boxsizehalf:
                            raise Exception(
                                "Size of space volume is too small to fit the survey inside. "
                            )

                    wrealspace = surveywindowfunction(
                        X, Y, Z, maxdistgal, n_ame, phicutoff=theta * np.pi / 180.0
                    )

                    # now get the window function in Fourier space numerically
                    int_over_wrealsp = simpson(
                        simpson(simpson(wrealspace, zdata), ydata), xdata
                    )

                    # calculate 3D fourier transform - fast fourier transform - requires different normalization with (2pi)^(3/2) for using ifft() inverse
                    wfourierspace = (
                        rfftn(wrealspace, (grid_size + 1, grid_size + 1, grid_size + 1))
                        * (np.max(xdata) - np.min(xdata)) ** 3
                        / (len(xdata) ** 3 * int_over_wrealsp)
                    )

                    wfourierspace = np.concatenate(
                        (
                            wfourierspace[:, :, :],
                            np.conj(np.flip(wfourierspace[:, :, 1:], axis=2)),
                        ),
                        axis=2,
                    )
                    wfourierspace = fftshift(wfourierspace)

                    # now we have everything needed to compute the Bulk Flow amplitude and error
                    Vp_to_plot[i, j, k], sigma[i, j, k] = calculate_Vp_sigma_v2(
                        ksx, ksy, ksz, wfourierspace, settings, pksline
                    )

                    console.print(
                        "Angle index:",
                        i,
                        "Radius index:",
                        j,
                        "Distribution index:",
                        k,
                        "Vp:",
                        Vp_to_plot[i, j, k],
                        "Sigma:",
                        sigma[i, j, k],
                    )

        np.save("output/Vp_to_plot" + filename_note + ".npy", Vp_to_plot)
        np.save("output/sigma" + filename_note + ".npy", sigma)

    if USE_SAVED:
        Vp_to_plot = np.load("output/Vp_to_plot" + filename_note + ".npy")
        sigma = np.load("output/sigma" + filename_note + ".npy")

    cmap = [
        "crimson",
        "fuchsia",
        "purple",
        "violet",
        "indigo",
        "darkblue",
        "blue",
        "cyan",
        "green",
        "lightgreen",
        "olive",
        "yellow",
        "orange",
        "red",
        "darkred",
    ]
    cmap = ["purple", "blue"]
    lines = ["--", "-", "-.", ":"]

    plt.rcParams["font.size"] = "15"

    radius = np.array(settings.radius) * (settings.H0 / 100.0)

    if settings.subplots_by_angle:
        f, axes = plt.subplots(1, len(settings.angles), sharey=False)
        f.set_size_inches(12, 6)
        for kk, k in enumerate(rad_distributions):
            if len(settings.angles) > 1:
                for t, theta in enumerate(settings.angles):
                    axes[len(settings.angles) - 1 - t].plot(
                        radius,
                        Vp_to_plot[t, :, kk],
                        label=rad_mathname[kk],
                        color=cmap[kk],
                        linestyle=lines[kk],
                    )
                    axes[len(settings.angles) - 1 - t].scatter(
                        radius, Vp_to_plot[t, :, kk], color=cmap[kk]
                    )

                    axes[len(settings.angles) - 1 - t].fill_between(
                        radius,
                        Vp_to_plot[t, :, kk] - sigma[t, :, kk],
                        Vp_to_plot[t, :, kk] + sigma[t, :, kk],
                        alpha=0.15,
                        color=cmap[kk],
                    )

                    axes[len(settings.angles) - 1 - t].set_xlabel(
                        r"$R$ Mpc $h^{-1}$", fontsize=15
                    )
                    axes[len(settings.angles) - 1 - t].set_title(
                        r"$\theta = %.2f$" % theta, fontsize=15
                    )

            else:
                axes.plot(
                    radius,
                    Vp_to_plot[0, :, kk],
                    label=rad_mathname[kk],
                    color=cmap[kk],
                    linestyle=lines[kk],
                )
                axes.scatter(radius, Vp_to_plot[0, :, kk], color=cmap[kk])
                axes.fill_between(
                    radius,
                    Vp_to_plot[0, :, kk] - sigma[0, :, kk],
                    Vp_to_plot[0, :, kk] + sigma[0, :, kk],
                    alpha=0.15,
                    color=cmap[kk],
                )

                axes.set_xlabel(r"$R$ Mpc $h^{-1}$", fontsize=15)
                axes.set_title(r"$\theta = %.2f$" % settings.angles[0], fontsize=15)

        if len(settings.angles) > 1:
            axes[0].set_ylabel(r"$V_p$ $\mathrm{km s}^{-1}$", fontsize=15)
            axes[0].legend(fontsize=15)
            axes[0].set_xticks([10, 20, 40, 80, 160, 320])
            axes[0].set_xticklabels(["10", "20", "40", "80", "160", "320"])
            axes[0].set_xscale("log", base=2)
            plt.savefig(
                "output/anglesubplots_bulk_flow_vs_R" + filename_note + ".png",
                bbox_inches="tight",
                dpi=200,
            )
            plt.show()

        else:
            axes.legend(fontsize=15)
            axes.set_ylabel(r"$V_p$ $\mathrm{km s}^{-1}$", fontsize=15)
            axes.set_xscale("log", base=2)
            axes.set_xticks([10, 20, 40, 80, 160, 320])
            axes.set_xticklabels(["10", "20", "40", "80", "160", "320"])
            plt.savefig(
                "output/anglesubplots_bulk_flow_vs_R" + filename_note + ".png",
                bbox_inches="tight",
                dpi=200,
            )
            plt.show()

    else:
        f, axes = plt.subplots(1, len(radius), sharey=True)
        f.set_size_inches(12, 6)
        for kk, k in enumerate(rad_distributions):
            if len(radius) > 1:
                for t, R in enumerate(radius):
                    axes[t].plot(
                        settings.angles,
                        Vp_to_plot[:, t, kk],
                        label=rad_mathname[kk],
                        color=cmap[kk],
                        linestyle=lines[kk],
                    )
                    axes[t].scatter(
                        settings.angles, Vp_to_plot[:, t, kk], color=cmap[kk]
                    )
                    axes[t].fill_between(
                        settings.angles,
                        Vp_to_plot[:, t, kk] - sigma[:, t, kk],
                        Vp_to_plot[:, t, kk] + sigma[:, t, kk],
                        alpha=0.15,
                        color=cmap[kk],
                    )

                    axes[t].set_xlabel(r"$\theta$", fontsize=15)
                    axes[t].set_title(r"$R = %.2f$ Mpc $h^{-1}$" % R, fontsize=15)

            else:
                axes.plot(
                    settings.angles,
                    Vp_to_plot[:, 0, kk],
                    label=rad_mathname[kk],
                    color=cmap[kk],
                    linestyle=lines[kk],
                )
                axes.scatter(settings.angles, Vp_to_plot[:, 0, kk], color=cmap[kk])
                axes.fill_between(
                    settings.angles,
                    Vp_to_plot[:, 0, kk] - sigma[:, 0, kk],
                    Vp_to_plot[:, 0, kk] + sigma[:, 0, kk],
                    alpha=0.15,
                    color=cmap[kk],
                )

                axes.set_xlabel(r"$\theta$", fontsize=15)
                axes.set_title(r"$R = %.2f$ Mpc $h^{-1}$" % radius[0], fontsize=15)

        if len(settings.angles) > 1:
            # axes[1].set_yticks([])
            axes[0].set_xscale("log", base=2)
            axes[0].set_xticks([10, 20, 40, 80, 160, 320])
            axes[0].set_xticklabels(["10", "20", "40", "80", "160", "320"])
            axes[0].set_ylabel(r"$V_p$ $\mathrm{km s}^{-1}$", fontsize=15)
            axes[0].legend(fontsize=15)
            # plt.subplots_adjust(wspace = 0.0, hspace=0.0)
            plt.savefig(
                "output/anglesubplots_bulk_flow_vs_R" + filename_note + ".png",
                bbox_inches="tight",
                dpi=200,
            )
            plt.show()

        else:
            axes.legend(fontsize=15)
            axes.set_xscale("log", base=2)
            # plt.subplots_adjust(wspace = 0.0, hspace=0.0)
            axes.set_xticks([10, 20, 40, 80, 160, 320])
            axes.set_xticklabels(["10", "20", "40", "80", "160", "320"])
            axes.set_ylabel(r"$V_p$ $\mathrm{km s}^{-1}$", fontsize=15)
            plt.savefig(
                "output/anglesubplots_bulk_flow_vs_R" + filename_note + ".png",
                bbox_inches="tight",
                dpi=200,
            )
            plt.show()


##################################################################################################################################
# tophat_halfwidth = 5.0

# def fouriertophat(kx): # integral of 1 * exp(-i2pix) dx from -a/2 to a/2 multiplied by 2pi
#     res = tophat_halfwidth*2.0*(np.sin(kx*2*np.pi*tophat_halfwidth))/((kx*2*np.pi*tophat_halfwidth)) # * (2.0*np.pi)
#     return res

# def fouriergaussian(kx): # integral of exp(-x^2)* exp(-i2pik) dx from xmin to xmax multiplied by 2pi
#     res = np.sqrt(np.pi)*np.exp(-1.0*(np.pi**2)*((kx)**2))
#     return res

# x = np.linspace(-50, 50, 10000)
# ytophat = np.where(abs(x) <= tophat_halfwidth, 1.0, 0.0)
# ygaussian = np.exp(-(x**2))
# kx = fftshift(fftfreq( len(x), (np.max(x)-np.min(x))/len(x)  ))

# fftyxtophat = abs(fftshift(fftn(ytophat))) * (np.max(x)-np.min(x))/len(x)
# fftyxgaussian = abs(fftshift(fftn(ygaussian))) * (np.max(x)-np.min(x))/len(x)

# expectationtophat = abs(fouriertophat(kx))
# expectationgaussian = abs(fouriergaussian(kx))


# plt.plot(kx, fftyxtophat, alpha=1, label='fft')
# plt.plot(kx, expectationtophat, alpha=1, label='expectation')
# plt.legend()
# plt.show()

# plt.plot(kx, fftyxgaussian, alpha=1, label='fft')
# plt.plot(kx, expectationgaussian, alpha=1, label='expectation')
# plt.legend()
# plt.show()

##################################################################################################################################


# x = np.linspace(-50, 50, 300)
# y = x
# z = x

# radius = 1

# X, Y, Z = np.meshgrid(x,y,z)
# Z = np.where(np.sqrt(X**2 + Y**2 + Z**2) < radius, 1.0, 0.0)
# Z = np.where(np.sqrt(X**2 + Y**2 + Z**2) < radius, np.exp(-np.sqrt(X**2 + Y**2 + Z**2)), 0.0)

# plt.imshow(Z[int(len(x)/2),:,:])

# plt.show()

# kx = fftshift(fftfreq( len(x), (np.max(x)-np.min(x))/len(x)  ))

# kx = fftshift(fftfreq( len(x), (np.max(x)-np.min(x))/len(x)  ))


# KX, KY, KZ = np.meshgrid(kx, kx, kx)

# kabs = np.sqrt(KX**2 + KY**2 + KZ**2)

# #vol = 4.0*np.pi*(np.max(radius)**3)/3

# vol = simpson(simpson(simpson(Z, z),y),x)

# #ffttophat = (fftshift(fftn(Z))) * ((np.max(x) - np.min(x))**3 / (len(x))**3 ) * (1.0/vol)

# ffttophat = (fftshift(fftn(Z))) * ((np.max(x) - np.min(x))**3 / (len(x))**3 ) * (1.0/vol)

# print(ffttophat.shape)

# theorytophat = (theory_tophat3d(radius, (2.0*np.pi*kabs)))

# ax1 = plt.subplot(131)
# ax2 = plt.subplot(132)
# ax3 = plt.subplot(133)


# ax1.imshow(ffttophat[int(len(x)/2),:,:])

# ax2.imshow(theorytophat[int(len(x)/2),:,:])

# ax3.imshow(ffttophat[int(len(x)/2),:,:]/theorytophat[int(len(x)/2),:,:])

# plt.legend()
# plt.show()

# print(ffttophat[int(len(x)/2),int(len(x)/2),int(len(x)/2)],theorytophat[int(len(x)/2),int(len(x)/2),int(len(x)/2)])

# # print(ffttophat[int(len(x)/2),int(len(x)/2),int(len(x)/2)]/theorytophat[int(len(x)/2),int(len(x)/2),int(len(x)/2)])

# print(abs(np.mean(ffttophat[:,:,:]/theorytophat[:,:,:])))

# ksv = np.logspace(-3, 1, 100, base=10)
# ks_cn = (ksv[1:]+ksv[:-1])/2.0

# kabs_reshape = kabs.reshape(-1)
# fftreshape = np.abs(ffttophat.reshape(-1))**2

# binned_dat_count = np.array((pd.cut((kabs_reshape), ksv, right=True).value_counts()))

# binned_dat_index = np.argmin( abs((kabs_reshape)[:,None]-ks_cn), axis=1 )

# binned_dat_power = pd.DataFrame({"fftval": fftreshape, "index": binned_dat_index})


# # print(binned_dat_power.groupby("index").sum().to_numpy()[0])

# binned_dat_power = binned_dat_power.groupby("index").sum().to_numpy()[0]/binned_dat_count
# print(binned_dat_power)
# for kindex in np.arange(len(ks_cn)):

#   binned_dat_power[kindex] = np.sum(np.where( binned_dat_index==kindex, fftreshape, 0.0 ))/binned_dat_count[kindex]


# plt.semilogx(ks_cn, binned_dat_power)
# plt.show()


# exit()
##################################################################################################################################

# exit()

# test
# -----------------------------------------------------------------------------------------------------#
# functions
# -----------------------------------------------------------------------------------------------------#


# rarr = np.linspace(-rz(0.3), rz(0.3), 50)
# print(rarr[1]-rarr[0])
# print(rz(0.041))
# exit()


# # Function that computes the survey window function for differen distributions with distance
# def surveywindowfunction(x, y, z, maxdist, dist, phicutoff):
#     R = np.sqrt(x**2 + y**2 + z**2)
#     phidata = np.arccos(z / R)

#     if dist == "uniform":
#         prob = np.ones(R.shape)

#     elif dist == "gaussian":
#         # sigma=70.0
#         prob = np.exp(
#             -(R**2) / ((maxdist / 3.0) ** 2)
#         )  # change here to look at BF for gaussian spheres with different sigmas

#     elif dist == "r power minus 2":
#         prob = 1.0 / (R**2)

#     elif dist == "r power 1":
#         prob = R

#     else:
#         return 0

#     prob[phidata > phicutoff] = 0.0
#     prob[R > maxdist] = 0.0
#     # prob = prob / np.max(prob) # - this line was not doing anything
#     return prob


# def calculate_Vp_sigma(ksx, ksy, ksz, wk):

#     # calculate 3D k-space integral to get sigma_V^2 for the survey
#     # (absolute value of the variance of most probable Bulk Flow vector)
#     sigma_v_2 = 0.0

#     prefactor = ((H0*Omega_m**(0.55))**2)/( (2.0*np.pi)**3 ) # constant

#     deltax = (ksx[1]-ksx[0])/2
#     deltay = (ksy[1]-ksy[0])/2
#     deltaz = (ksz[1]-ksz[0])/2

#     dkx = abs(ksx[1:] - ksx[0:-1]) + deltax
#     dky = abs(ksy[1:] - ksy[0:-1]) + deltay
#     dkz = abs(ksz[1:] - ksz[0:-1]) + deltaz

#     dkx = np.concatenate(( dkx, np.array(([dkx[-1]])) ))
#     dky = np.concatenate(( dky, np.array(([dky[-1]])) ))
#     dkz = np.concatenate(( dkz, np.array(([dkz[-1]])) ))

#     for kkxx in np.arange(len(ksx)):

#         vals_ky = np.zeros((len(ksy)))

#         for kkyy in np.arange(len(ksy)):

#             vals_kz = np.zeros((len(ksz)))

#             k_abs = np.sqrt(ksx[kkxx] ** 2 + ksy[kkyy] ** 2 + ksz[:] ** 2)

#             Pkmm = pksline(k_abs)

#             wkval = wk[kkxx][kkyy][:]

#             kabsnotzero = np.where(k_abs != 0.0)
#             vals_kz[kabsnotzero] = prefactor * abs(wkval[kabsnotzero]) ** 2 * Pkmm[kabsnotzero] / (k_abs[kabsnotzero] ** 2) * dkz[kabsnotzero]

#             vals_kz = vals_kz*dky[kkyy]

#             vals_ky[kkyy] = np.sum(vals_kz)

#         vals_ky = vals_ky*dkx[kkxx]

#         sigma_v_2 += np.sum(vals_ky)

#     # get V
#     V_mostprob = np.sqrt( 1.5 * sigma_v_2 )
#     sigma = np.sqrt( sigma_v_2*(1.0 - 8.0/(3.0*np.pi))  )
#     return V_mostprob, sigma
