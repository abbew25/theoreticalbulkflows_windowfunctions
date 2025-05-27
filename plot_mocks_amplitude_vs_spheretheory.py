import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

##########################################################################################################################
# plotting code for my paper for specific data
# theory

H0 = 67.74  # km/s/Mpc
radius = np.logspace(np.log2(10.0), np.log2(600.0), 20, base=2)
c = 2.99792458e5  # km/s
Om = 0.31
q0 = -0.55
j0 = 1.0
H0 = 69.0

Vp_to_plot = np.load("data_literature_comparison_vp.npy")
sigma = np.load("data_literature_comparison_sigma.npy")

plt.plot(
    radius * (H0 / 100.0),
    Vp_to_plot,
    color="lightblue",
    linestyle="--",
    label=r"$\Lambda$CDM expectation for a spherical volume",
)
plt.fill_between(
    radius * (H0 / 100.0),
    Vp_to_plot - sigma,
    Vp_to_plot + sigma,
    color="lightblue",
    alpha=0.6,
)


##########################################################################################################################

# kaiser mocks


# function used to integrate to compute proper distance from Friedmann Equation
def E_z_inverse(z):
    """
    Compute the inverse of the E(z) function (from the first Friedmann Equation).
    """
    return 1.0 / (np.sqrt((Om * (1.0 + z) ** 3) + (1.0 - Om)))


# function that computes the proper distance as a function of redshift (dimensionless)
def rz(red):
    """
    Calculates the proper radial distance to an object at redshift z for the given cosmological model.
    """
    try:
        d_com = (c / H0) * quad(E_z_inverse, 0.0, red, epsabs=5e-5)[0]
        return d_com
    except TypeError:
        distances = np.zeros(len(red))
        for i, z in enumerate(red):
            distances[i] = (c / H0) * quad(E_z_inverse, 0.0, z, epsabs=5e-5)[0]
        return distances


# get zmod
def zed_mod(z):
    zmod = z * (
        1.0
        + 0.5 * (1.0 - q0) * z
        - (1.0 / 6.0) * (j0 - q0 - 3.0 * (pow(q0, 2)) + 1.0) * (pow(z, 2))
    )

    return zmod


depth_spheres = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]

# read in mocks and get average distributions on different scales
num_sims = 256
num_subsims = 2
truth_BF_mocks_diffscales_r = np.zeros((len(depth_spheres), num_sims * num_subsims))

truth_BF_mocks_effectivescale = np.zeros((len(depth_spheres)))


for sphere in np.arange(9):
    # read on one mock to estimate effective depth

    mockdat = pd.read_csv(
        "/Users/s4638026/Desktop/MVE BF project/real_data_BF_measurement/stacked_calibrated_mocks_perfectcal_wnbar/MOCK_HAMHOD_R19000.0",
        index_col=False,
        header=0,
    )
    mockdat["r"] = rz(mockdat["z_obs"].to_numpy())
    mockdat = mockdat[mockdat["r"] < depth_spheres[sphere]]

    zmod = zed_mod(mockdat["z_obs"].to_numpy())
    sigma_vel = c * zmod / (1.0 + zmod) * np.log(10) * mockdat["logdist_err"].to_numpy()
    weights = 1.0 / (sigma_vel**2 + 300.0**2)
    truth_BF_mocks_effectivescale[sphere] = np.sum(
        weights * mockdat["r"].to_numpy()
    ) / np.sum(weights)

    counter = 0
    for i in np.arange(19000, 19000 + num_sims):
        for j in np.arange(0, num_subsims):
            try:
                with open(
                    "/Users/s4638026/Desktop/MVE BF project/real_data_BF_measurement/kaiser_mocks_shells_sphere_res/MOCK_RES_HAMHOD_"
                    + str(i)
                    + "."
                    + str(j)
                    + "_sphere_"
                    + str(sphere)
                    + ".csv",
                    "r",
                ) as f:
                    dat = f.readlines()

                    bfmodes = dat[0]
                    bfmodes = bfmodes.strip()
                    bfmodes = bfmodes.split()

                    vec = np.array(
                        ([float(bfmodes[0]), float(bfmodes[1]), float(bfmodes[2])])
                    )

                    truth_BF_mocks_diffscales_r[sphere, counter] = np.sqrt(
                        vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
                    )

            except FileNotFoundError:
                continue

            counter += 1

averages_ofcut = np.array(([np.mean(bf) for bf in truth_BF_mocks_diffscales_r[:]]))
stdev_ofcut = np.array(([np.std(bf) for bf in truth_BF_mocks_diffscales_r[:]]))

h = H0 / 100.0

plt.scatter(
    truth_BF_mocks_effectivescale * h,
    averages_ofcut,
    color="black",
    label="Average of mocks + 1 st. dev. \n (Kaiser mocks)",
)
plt.fill_between(
    truth_BF_mocks_effectivescale * h,
    averages_ofcut - stdev_ofcut,
    averages_ofcut + stdev_ofcut,
    color="black",
    alpha=0.2,
)


##########################################################################################################################

# Peery MVE mocks


depths_mocks = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
depths_mocks = np.array((depths_mocks), dtype=np.float)

# read in mocks and get average distributions on different scales
num_sims = 256
num_subsims = 2
truth_BF_mocks_diffscales_r = np.zeros((10, num_sims * num_subsims))

folder_path = "/Users/s4638026/Desktop/MVE BF project/real_data_BF_measurement/stacked_mocks_results_peery3modes_truevels_version2/"

for sphere in np.arange(10):
    counter = 0
    for i in np.arange(19000, 19000 + num_sims):
        for j in np.arange(0, num_subsims):
            try:
                with open(
                    folder_path
                    + "/MOCK_RES_HAMHOD_"
                    + str(i)
                    + "."
                    + str(j)
                    + "_scale_"
                    + str(int(depths_mocks[sphere]))
                    + ".000000.txt",
                    "r",
                ) as f:
                    dat = f.readlines()
                    # print(dat)
                    bfmodes = dat[0]
                    bfmodes = bfmodes.strip()
                    bfmodes = bfmodes.split()

                    vec = np.array(
                        ([float(bfmodes[0]), float(bfmodes[1]), float(bfmodes[2])])
                    )

                    truth_BF_mocks_diffscales_r[sphere, counter] = np.sqrt(
                        vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
                    )

            except FileNotFoundError:
                print(i, j)
                continue

            counter += 1


averages_ofcut = np.mean(truth_BF_mocks_diffscales_r, axis=1)
stdev_ofcut = np.array(([np.std(bf) for bf in truth_BF_mocks_diffscales_r[:]]))

plt.scatter(
    depths_mocks * h,
    averages_ofcut,
    color="purple",
    label="Average mocks + 1 st. dev. \n (Peery MVE mocks)",
)
plt.fill_between(
    depths_mocks * h,
    averages_ofcut - stdev_ofcut,
    averages_ofcut + stdev_ofcut,
    color="purple",
    alpha=0.2,
)

##########################################################################################################################

# now do the plot

plt.xscale("log", base=2)
plt.xticks(ticks=[20, 40, 80, 160, 320], labels=["20", "40", "80", "160", "320"])
plt.rcParams["font.size"] = 16
plt.xlabel(r"$ d_e$ $\mathrm{Mpc} h^{-1}$")
plt.ylabel(r"$|B|$ $\mathrm{km s}^{-1}$")
plt.legend(ncol=1, loc="upper right", fontsize=10)
plt.show()
