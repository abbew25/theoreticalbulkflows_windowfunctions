import numpy as np

# import matplotlib.pyplot as plt
from classy import Class


def run_class(cosmo, redshift):
    omega_b, omega_cdm, H0, As, m_nu = cosmo

    omega_b = omega_b * (H0 / 100.0) ** 2
    omega_cdm = omega_cdm * (H0 / 100.0) ** 2
    h = H0 / 100.0
    omega_m = (omega_b + omega_cdm + m_nu / 93.14) / (h**2)

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
            "n_s": 0.96,
        }
    )

    M.set({"output": "mPk, mTk", "P_k_max_1/Mpc": 10.0, "z_max_pk": 1.0})
    M.compute()

    # Get the power spectra
    # k = np.logspace(np.log10(0.7*1e-3), np.log10(3.0), 2000, base = 10.0)
    k = np.linspace(1e-5, 3.0, 3000)

    Pk = np.array([M.pk_cb_lin(ki, redshift) for ki in k])

    transfer = M.get_transfer(redshift, output_format="class")

    print(M.sigma8(), omega_m)

    return k, Pk, transfer


# k, Pk = run_class([0.04, 0.26, 67.32, 2.1005e-9, 0.0], 0.0)

# Ocdm_frac = 0.26/0.3
# Ob_frac = 0.04/0.3

# Om = 0.3121

# Ocdm = Om*Ocdm_frac
# Ob = Om*Ob_frac

# write the data to a file
# Pk0 = pd.DataFrame(run_class([Ob, Ocdm, 67.32, 2.1005e-9, 0.0], 0.0))
# Pk025 = pd.DataFrame(run_class([0.04, 0.26, 67.32, 2.1005e-9, 0.0], 0.025))
# Pk050 = pd.DataFrame(run_class([0.04, 0.26, 67.32, 2.1005e-9, 0.0], 0.05))

# write the data to a csv


# Pk0.to_csv('powerspectrum_SDSS_z=0.csv', header = False, index = False )
# Pk025.to_csv('powerspectrum_z=025.csv', header = False, index = False )
# Pk050.to_csv('powerspectrum_z=050.csv', header = False, index = False )


# k, Pk_0 = run_class([0.04, 0.26, 67.32, 2.1005e-9, 0.0], 0.0)

# k, Pk_05 = run_class([0.04, 0.26, 67.32, 2.1005e-9, 0.0], 0.05)

# k, Pk_10 = run_class([0.04, 0.26, 67.32, 2.1005e-9, 0.0], 0.10)


# plt.loglog(k, Pk_0, label = 'z = 0')
# plt.loglog(k, Pk_05, label = 'z = 0.05')
# plt.loglog(k, Pk_10, label = 'z = 0.1')
# plt.legend()
# plt.show()

# plt.semilogx(k, Pk_0/Pk_0, label = 'z = 0')
# plt.semilogx(k, Pk_05/Pk_0, label = 'z = 0.05')
# plt.semilogx(k, Pk_10/Pk_0, label = 'z = 0.1')
# plt.legend()
# plt.show()


# power spectrum for simple sdss mocks (using L-PICOLA)
h = 0.69
Pkktransfer = run_class([0.048, 0.31 - 0.048, 69.0, 2.1005e-9, 0.0], 0.00)[0:2]


# Pkktransfer = run_class([0.022169/(h**2), 0.11896/(h**2), 67.74, 2.124e-9, 0.057], 0.00)[:-1]

# exit()

Pkktransfer = np.array(Pkktransfer[0:2])
Pkktransfer[0] = Pkktransfer[0] / h
Pkktransfer[1] = Pkktransfer[1] * (h**2)

Pkktransfer = np.array((Pkktransfer))

np.savetxt("powerspectrum_SDSS_simple_mocks.csv", Pkktransfer)

exit()

print(Pkktransfer[0])
print(Pkktransfer[0] / 0.69)

Pk_k_file = {"# k/h: ": Pkktransfer[0] / 0.69, "Pk: ": Pkktransfer[1] * (0.69**3)}

# Pk_k_file = pd.DataFrame(Pk_k_file)

# Pk_k_file = {

#     "# k: ": Pkktransfer[0],
#     "Pk: ": Pkktransfer[1],

#              }

# Pk_k_file = pd.DataFrame(Pk_k_file)


# print(Pkktransfer[2])

# transfer_file = {
#     "# k/h: ": Pkktransfer[2]['k (h/Mpc)'],
#     "Tk: ": Pkktransfer[2]['d_cdm']
#     }

# transfer_file = pd.DataFrame(transfer_file)


Pk_k_file.to_csv(
    "~/Desktop/MVE BF project/real_data_BF_measurement/powerspectrum_SDSS_simple_mocks.csv",
    header=False,
    index=False,
    sep=" ",
)

# Pk_k_file.to_csv('~/Desktop/MVE BF project/two_powerspectrum_SDSS_simplemock_z=0_2.csv',
# header = False, index = False, sep=' ')

# transfer_file.to_csv('~/Desktop/CullanHowlett-l-picola-345ff44/files_abbe_simple_sdss_mocks/transferfunc_SDSS_simplemock_z=0.csv',
# header = False, index = False, sep=' ')


# -----------------------------------------------------------------------


# def j_0(x):

#     try:
#         if (x != 0):
#             return np.sin(x)/x
#         else:
#             return 1.0
#     except:
#         if (0 not in x):
#             return np.sin(x)/x
#         else:
#             return 1.0


# def j_2(x):

#     try:
#         if (x != 0):
#             return (3.0/(x**2) - 1.0 )*j_0(x) - 3.0*np.cos(x)/(x**2)
#         else:
#             return 0.0
#     except:
#         if (0 not in x):
#             return (3.0/(x**2) - 1.0 )*j_0(x) - 3.0*np.cos(x)/(x**2)
#         else:
#             return 0.0


# def f_mn(k, alpha, rm, rn):

#     A = np.sqrt( rm**2 + rn**2 - 2*rm*rn*np.cos(alpha) )

#     if alpha == 0 and rm == rn:
#         return 1.0/3.0
#     else:
#         res = (1.0/3.0)*np.cos(alpha)*( j_0(k*A) - 2.0*j_2(k*A) )
#         res += (1.0/(A**2))*j_2(k*A)*rm*rn*(np.sin(alpha)**2)
#         return res


# k, Pk = run_class([0.04, 0.26, 67.32, 2.1005e-9, 0.0], 0.0)

# alpha1 = 0.5
# rm1 = 2000.0
# rn1 = 2000.0

# fmn_k1 = f_mn(k, alpha1, rm1, rn1)
# fmn_k2 = f_mn(k, 1.0, rm1, rn1)
# fmn_k3 = f_mn(k, 0.0, rm1, rn1)
# fmn_k4 = f_mn(k, 0.0, rm1/2.0, rn1)
# fmn_k5 = f_mn(k, 0.0, rm1/2.0, rn1/4.0)
# fmn_k6 = f_mn(k, 1.0, rm1/10.0, rn1/10.0)

# func1 = Pk*fmn_k1
# func2 = Pk*fmn_k2
# func3 = Pk*fmn_k3
# func4 = Pk*fmn_k4
# func5 = Pk*fmn_k5
# func6 = Pk*fmn_k6

# plt.semilogx(k, func1, label = '0.5')
# plt.semilogx(k, func2, label = '1.0')
# plt.semilogx(k, func3, label = '0.0')
# plt.semilogx(k, func4, label = '0.0')
# plt.semilogx(k, func5, label = '0.0')
# plt.semilogx(k, func6, label = '1.0')
# plt.semilogx(k, Pk, label = 'power spectrum')
# plt.hlines(0.0, np.min(k), np.max(k))
# plt.legend()
# plt.ylim([-10,10])
# plt.show()
