import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fftshift, fftn
from rich.console import Console

# code for comparing fft vs analytic fourier transform of basic functions for galaxy survey geometry

tophat_halfwidth = 5.0


def fouriertophat_1d(kx: npt.NDArray):
    res = (
        tophat_halfwidth
        * 2.0
        * (np.sin(kx * 2 * np.pi * tophat_halfwidth))
        / (kx * 2 * np.pi * tophat_halfwidth)
    )
    return res


def fouriergaussian_1d(kx: npt.NDArray):
    res = np.sqrt(np.pi) * np.exp(-1.0 * (np.pi**2) * ((kx) ** 2))
    return res


def theory_tophat3d(R: float, k: npt.NDArray):
    res = 3.0 * (np.sin(k * R) - (k * R) * np.cos(k * R)) / ((R * k) ** 3)
    res = np.where(k == 0, 1.0, res)
    return res


def compare_fourierfft_to_analyticfourier_tophat_1d(tophat_halfwidth: float):
    x = np.linspace(-50, 50, 10000)
    ytophat = np.where(abs(x) <= tophat_halfwidth, 1.0, 0.0)
    # ygaussian = np.exp(-(x**2))

    kx = fftshift(fftfreq(len(x), (np.max(x) - np.min(x)) / len(x)))  # fourier k modes
    fftyxtophat = abs(fftshift(fftn(ytophat))) * (np.max(x) - np.min(x)) / len(x)
    # fftyxgaussian = abs(fftshift(fftn(ygaussian))) * (np.max(x) - np.min(x)) / len(x)

    expectationtophat = abs(fouriertophat_1d(kx))
    # expectationgaussian = abs(fouriergaussian(kx))

    plt.rcParams["font.size"] = 14
    plt.plot(kx, fftyxtophat, alpha=1, label="fft")
    plt.plot(kx, expectationtophat, alpha=1, label="analytic")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$F(k)$")
    plt.title("Fourier Transform of Top hat function")
    plt.legend()
    plt.show()


def compare_fourierfft_to_analyticfourier_gaussian_1d():
    x = np.linspace(-50, 50, 10000)
    # ytophat = np.where(abs(x) <= tophat_halfwidth, 1.0, 0.0)
    ygaussian = np.exp(-(x**2))

    kx = fftshift(fftfreq(len(x), (np.max(x) - np.min(x)) / len(x)))  # fourier k modes
    # fftyxtophat = abs(fftshift(fftn(ytophat))) * (np.max(x) - np.min(x)) / len(x)
    fftyxgaussian = abs(fftshift(fftn(ygaussian))) * (np.max(x) - np.min(x)) / len(x)

    # expectationtophat = abs(fouriertophat(kx))
    expectationgaussian = abs(fouriergaussian_1d(kx))

    plt.rcParams["font.size"] = 14
    plt.plot(kx, fftyxgaussian, alpha=1, label="fft")
    plt.plot(kx, expectationgaussian, alpha=1, label="analytic")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$F(k)$")
    plt.title("Fourier Transform of gaussian function")
    plt.legend()
    plt.show()


def compare_fourierfft_to_analyticfourier_tophat_3d():
    x = np.linspace(-30, 30, 500)
    y = x
    z = x

    radius = 3

    X, Y, Z = np.meshgrid(x, y, z)
    Z = np.where(np.sqrt(X**2 + Y**2 + Z**2) < radius, 1.0, 0.0)

    # plt.imshow(Z[int(len(x) / 2), :, :])
    # plt.show()

    kx = fftshift(fftfreq(len(x), (np.max(x) - np.min(x)) / len(x)))

    KX, KY, KZ = np.meshgrid(kx, kx, kx)

    kabs = np.sqrt(KX**2 + KY**2 + KZ**2)

    vol = 4.0 * np.pi * (np.max(radius) ** 3) / 3
    ffttophat = (
        abs(fftshift(fftn(Z)))
        * ((np.max(x) - np.min(x)) ** 3 / (len(x)) ** 3)
        * (1.0 / vol)
    )

    theorytophat = abs(theory_tophat3d(radius, (2.0 * np.pi * kabs)))

    plt.rcParams["font.size"] = 14
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    # ax3 = plt.subplot(133)

    ax1.imshow(ffttophat[int(len(x) / 2), :, :])
    ax1.set_title("FFT")
    ax2.imshow(theorytophat[int(len(x) / 2), :, :])
    ax2.set_title("analytic")
    plt.tight_layout()
    # ax3.imshow(
    #     np.log(ffttophat[int(len(x) / 2), :, :] / theorytophat[int(len(x) / 2), :, :])
    # )

    ax1.set_xlabel(r"$k$")
    ax2.set_xlabel(r"$k$")
    ax1.set_ylabel(r"$k$")
    ax2.set_ylabel(r"$k$")

    plt.show()

    # print(
    #     ffttophat[int(len(x) / 2), int(len(x) / 2), int(len(x) / 2)],
    #     theorytophat[int(len(x) / 2), int(len(x) / 2), int(len(x) / 2)],
    # )

    # print(
    #     ffttophat[int(len(x) / 2), int(len(x) / 2), int(len(x) / 2)]
    #     / theorytophat[int(len(x) / 2), int(len(x) / 2), int(len(x) / 2)]
    # )


if __name__ == "__main__":
    console = Console()
    console.log("Running test code for fft vs analytic (tophat)")
    compare_fourierfft_to_analyticfourier_tophat_1d(tophat_halfwidth)
    console.log("Running test code for fft vs analytic (gaussian)")
    compare_fourierfft_to_analyticfourier_gaussian_1d()
    console.log(
        "Running test code for fft vs analytic (3d top hat) - plotting 1D slice through the centre"
    )
    compare_fourierfft_to_analyticfourier_tophat_3d()
