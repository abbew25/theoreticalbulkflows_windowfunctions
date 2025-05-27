import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.fft import fftfreq, fftshift, rfftn
from calculate_bulkflowvector_varyinggeometries import (
    read_in_powerspectrum,
    CubicSpline,
    surveywindowfunction,
    calculate_Vp_sigma_v2,
)
from scipy.integrate import simpson

# plotting code for data for paper
data = pd.DataFrame(
    {
        "dataset_name": [
            "CF4",
            "CF4",
            "SDSSv",
            "CF4TF",
            "CF3",
            "2MTF+6dFGSv",
            "6dFGSv",
            "CF2",
            "2MTF",
            "SFI++",
            "COMPOSITE",
            "CMB x-ray clusters",
            "COMPOSITE",
            "COMPOSITE",
            "CF4",
            "CF4",
            "CF2",
            "SFI++",
        ],
        "authors": [
            "Watkins et al 2023",
            "Watkins et al 2023",
            "Howlett et al 2022",
            "Qin et al 2021",
            "Peery et al 2018",
            "Qin et al 2018",
            "Scrimgeour et al 2016",
            "Watkins and Feldman, 2015",
            "Hong et al 2014",
            "Ma and Pan 2014",
            "Ma, Gordon and Feldman 2011",
            "Kashlinsky et al 2008",
            "Feldman, Watkins, Hudson 2010",
            "Watkins, Feldman, Hudson 2009",
            "this work",
            "this work",
            "Hoffman, Courtois and Tully 2015",
            "Nusser and Davis 2011",
        ],
        "methods": [
            "MVE-2018",
            "MVE-2018",
            "Kaiser MLE",
            r"$\eta$MLE",
            "MVE-2018",
            r"$\eta$MLE",
            "kaiser MVE-2009",
            "MVE-2009",
            r"$\chi^2$ minimization",
            "ML method",
            "ML method",
            "SZ effect",
            "MVE-2009",
            "MVE-2009",
            "MVE-2018",
            "Kaiser MLE",
            "WF/CR",
            "ASCE",
        ],
        "BF_sgx": [
            -360.8,
            -391.0,
            -345,
            -353.7,
            None,
            -250.4,
            -186.1,
            -251,
            -271.4,
            -231.2,
            -283.9,
            None,
            -337.4,
            -347.7,
            -391,
            -382,
            -196,
            -198,
        ],
        "BF_sgy": [
            -25.5,
            -43,
            -88,
            -25.7,
            None,
            113,
            155,
            60,
            44.3,
            58.9,
            73.2,
            None,
            69.6,
            78.7,
            -119,
            48,
            59,
            61,
        ],
        "BF_sgz": [
            -137.6,
            -143,
            -75,
            -129.9,
            None,
            -86,
            19.2,
            -136.3,
            -99.2,
            -164.8,
            -172.2,
            None,
            -233.2,
            -196.3,
            -126,
            -135,
            -123,
            -151,
        ],
        "BF_amplitude": [
            387,
            419,
            381,
            376,
            282,
            288,
            243,
            292,
            292.3,
            290.0,
            340,
            800,
            416,
            407,
            428,
            408,
            239,
            257,
        ],
        "amplitude_err1": [
            np.sqrt(28**2 + 64.0**2),
            np.sqrt(36**2 + 52**2),
            np.sqrt(79**2 + 43**2),
            np.sqrt(183**2 + 23**2),
            64.0,
            np.sqrt(24**2 + 126**2),
            np.sqrt(58**2 + 101**2),
            np.sqrt(61**2 + 115**2),
            np.sqrt(28**2 + 125**2),
            np.sqrt(30**2 + 108**2),
            130,
            np.sqrt(200**2 + 38**2),
            78,
            np.sqrt(81**2 + 115**2),
            165,
            108,
            38,
            44,
        ],
        "x_err1": [
            14.2,
            7.0,
            np.sqrt((66 / 2.0 + 101 / 2.0) ** 2 + 119**2),
            45.2,
            None,
            33.7,
            4.1,
            44.4,
            25.6,
            44.2,
            6.1,
            None,
            5.7,
            29.5,
            165,
            104,
            43,
            34,
        ],
        "y_err1": [
            13.1,
            22,
            np.sqrt(34.5**2 + 131**2),
            7.6,
            None,
            40.6,
            67.2,
            27.6,
            34,
            19.6,
            87.7,
            None,
            31.6,
            30.4,
            149,
            93,
            27,
            11,
        ],
        "z_err1": [
            76.8,
            75,
            np.sqrt(119.5**2 + 107**2),
            180.8,
            None,
            134.1,
            169.6,
            130.2,
            138.1,
            110.5,
            192.2,
            None,
            115.1,
            140.5,
            154,
            122,
            41,
            26,
        ],
        "depth": [
            150,
            200,
            139,
            35,
            150,
            40,
            70,
            50,
            40,
            58,
            33,
            300,
            100,
            50,
            173,
            49,
            150,
            100,
        ],
        "depth_string": [
            "150",
            "200",
            "139",
            "35",
            "150",
            "40",
            "70",
            "50",
            "40",
            "58",
            "33",
            "300",
            "100",
            "50",
            "173",
            "49",
            "150",
            "100",
        ],
    }
)

# 'uncertainty type': [ r'$^* $', r'$^* ^{\dagger}$', r'$^* ^{\dagger}$', r'$^* $', r'$^* ^{\dagger}$',
#                      r'$^* ^{\dagger}$', r'$^* $', r'$^* $', r'$^* ^{\dagger}$', r'$^* $',
#                      r'$^* $', r'$^* $', r'$^* $', r'$^* ^{\dagger}$', r'$^* ^{\dagger}$']


plt.rcParams["font.size"] = 16

# constants
c = 299792.458  # km/s


radius = np.logspace(np.log2(10.0), np.log2(600.0), 20, base=2)
grid_size = 400
grid_size_orig = grid_size
Vp_to_plot = np.zeros((len(radius)))
sigma = np.zeros((len(radius)))

Omega_m = 0.3089
Omega_L = 1.0 - Omega_m
H0 = 67.74  # km/s/Mpc
CALCULATE = False

cosmologies = [
    [0.0486, Omega_m - 0.0486, H0, 2.07e-9, 0.0],
    # [0.0486, Omega_m-0.0486, 73.2, 2.07e-9, 0.0], # varying H0
    # [0.0486, Omega_m-0.0486, H0, 2.07e-9, 0.0]  # varying sigma8
]

if CALCULATE:
    for c in np.arange(len(cosmologies)):
        global ks, Pk, pksline
        ks, Pk = read_in_powerspectrum(cosmologies[c], 0.0)
        pksline = CubicSpline(ks, Pk, extrapolate=False)  # setting up spline

        for j, R in enumerate(radius):  # ([radius[0]]):
            boxsizehalf = np.max([radius[j] * 10.0, np.max(radius) * 2])
            if np.max(radius) > radius[j] * 10.0:
                grid_size = 700
                print("grid size = 700")
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

            #  N = 151 # THIS MUST BE AN UNEVEN NUMBER TO GET FOURIER TRANSFORM USING RFFTN()

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

            maxdistgal = radius[j]
            # for SF, n_ame in enumerate(['uniform', 'gaussian', 'r power minus 2']):
            n_ame = "uniform"
            wrealspace = surveywindowfunction(
                X, Y, Z, maxdistgal, n_ame, phicutoff=180.0 * np.pi / 180.0
            )

            # now get the window function in Fourier space numerically
            int_over_wrealsp = simpson(
                simpson(simpson(wrealspace, zdata), ydata), xdata
            )

            # print(int_over_wrealsp, np.sum(wrealspace)*(volstandard))

            # calculate 3D fourier transform - fast fourier transform - requires different normalization with (2pi)^(3/2) for using ifft() inverse
            wfourierspace = (
                rfftn(wrealspace, (grid_size + 1, grid_size + 1, grid_size + 1))
                * (np.max(xdata) - np.min(xdata)) ** 3
                / (len(xdata) ** 3 * int_over_wrealsp)
            )

            # wfourierspace = np.concatenate(( np.conj(np.flip(wfourierspace, axis=2)), wfourierspace[:,:,1:] ), axis=2)
            wfourierspace = np.concatenate(
                (
                    wfourierspace[:, :, :],
                    np.conj(np.flip(wfourierspace[:, :, 1:], axis=2)),
                ),
                axis=2,
            )
            wfourierspace = fftshift(wfourierspace)

            Vp_to_plot[j], sigma[j] = calculate_Vp_sigma_v2(
                ksx, ksy, ksz, wfourierspace, pksline
            )

    if CALCULATE:
        np.save("data_literature_comparison_vp.npy", Vp_to_plot)
        np.save("data_literature_comparison_sigma.npy", sigma)

else:
    Vp_to_plot = np.load("data_literature_comparison_vp.npy")
    sigma = np.load("data_literature_comparison_sigma.npy")

    # radius = radius*(H0/100.0)

plt.plot(
    radius * (H0 / 100.0),
    Vp_to_plot,
    label=r"$\Lambda$CDM (standard model prediction)",
    color="lightblue",
    linestyle="--",
)  # , label=r'$\Lambda$CDM expectation')
plt.fill_between(
    radius * (H0 / 100.0),
    Vp_to_plot - sigma,
    Vp_to_plot + sigma,
    color="lightblue",
    alpha=0.6,
)


data = data.sort_values(by="depth")

markers = [
    "s",
    "^",
    "v",
    "*",
    "*",
    "<",
    ">",
    "8",
    "p",
    "P",
    "h",
    "H",
    "D",
    "d",
    "x",
    "s",
    "s",
    "^",
]


# cmap = cm.get_cmap("plasma", 15)
cmap = [
    "crimson",
    "fuchsia",
    "purple",
    "violet",
    "k",
    "indigo",
    "darkblue",
    "blue",
    "cyan",
    "green",
    "lightgreen",
    "olive",
    "yellow",
    "orange",
    "k",
    "red",
    "darkred",
    "pink",
    "skyblue",
]

cmap = cmap[::-1]
check = True
for i in np.arange(len(data["depth"])):
    if "this work" in data["authors"].to_numpy()[i]:
        plt.errorbar(
            data["depth"].to_numpy()[i],
            data["BF_amplitude"].to_numpy()[i],
            data["amplitude_err1"].to_numpy()[i],
            fmt="o",
            label="%s, %s, %s"
            % (
                data["authors"].to_numpy()[i],
                data["dataset_name"].to_numpy()[i],
                data["methods"].to_numpy()[i],
            ),
            marker=markers[i],
            color="red",
            alpha=0.8,
            ecolor="red",
        )
    # color=cmap[i], alpha=0.8, ecolor=cmap[i])

    else:
        if check:
            plt.errorbar(
                data["depth"].to_numpy()[i],
                data["BF_amplitude"].to_numpy()[i],
                data["amplitude_err1"].to_numpy()[i],
                fmt="o",
                # label = '%s, %s, %s' % (data['authors'].to_numpy()[i], data['dataset_name'].to_numpy()[i], data['methods'].to_numpy()[i]),
                # marker=markers[i],
                label="Other works",
                color="blue",
                alpha=0.8,
                ecolor="blue",
            )
            check = False
        else:
            plt.errorbar(
                data["depth"].to_numpy()[i],
                data["BF_amplitude"].to_numpy()[i],
                data["amplitude_err1"].to_numpy()[i],
                fmt="o",
                # label = '%s, %s, %s' % (data['authors'].to_numpy()[i], data['dataset_name'].to_numpy()[i], data['methods'].to_numpy()[i]),
                # marker=markers[i], #label ='Other works',
                color="blue",
                alpha=0.8,
                ecolor="blue",
            )

plt.xscale("log", base=2)
plt.xticks(ticks=[20, 40, 80, 160, 320], labels=["20", "40", "80", "160", "320"])
plt.rcParams["font.size"] = 16
# plt.xlabel(r'$ d_e$ $\mathrm{Mpc} h^{-1}$')
plt.xlabel(r"distance $\mathrm{Mpc} h^{-1}$")

# plt.ylabel(r'$|B|$ $\mathrm{km s}^{-1}$')
plt.ylabel(r"bulk flow $\mathrm{km s}^{-1}$")

plt.legend(ncol=1, loc="upper left", fontsize=13)
# plt.legend(ncol=2, loc='upper left', fontsize=10)
plt.show()
