"""
@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit

from Funz9 import accumulation, accumulation_open, animation_Ising
from Funz9 import T_variation, c_as_derivative





# -----------------------------------------------------------------------------
# STUDY OF THE EQUILIBRATION TIME
# -----------------------------------------------------------------------------

Beta = 1
neqs = 0
nmcs = 10**6
n1 = n2 = 30
N = n1*n2


results = accumulation(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)

fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.plot(Nsteps_lst, results[0]/N, marker='o' )
ax_m.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_m.set_ylabel(r'$ M / N $', fontsize=15)
ax_m.grid(True)
plt.show()

fig_e, ax_e = plt.subplots(figsize=(6.2, 4.5))
ax_e.plot(Nsteps_lst, results[2]/N, marker='o' )
ax_e.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_e.set_ylabel(r'$ E / N $', fontsize=15)
ax_e.grid(True)
plt.show()



# (Eye-made estimatimation!)

beta_lst = np.asarray([0.25,0.35, 0.4, 0.425, 0.45, 0.5, 0.55, 0.65, 0.75,0.85, 0.95])

# For a 30x30 square lattice:
n_eq30 = np.asarray([15*900, 50*900, 35*900, 200*900, 400*900, 250*900, 200*900, 150*900, 180*900, 130*900, 185*900])
# For a 20x20 square lattice:
n_eq20 = np.asarray([15*400, 20*400, 75*400, 160*400, 600*400, 130*400, 100*400, 50*400, 75*400, 45*400, 35*400])
# For a 4x4 square lattice:
n_eq4 = np.asarray([1*16, 3*16, 14*16, 13*16, 13*16, 13*16, 3*16, 5*16, 5*16, 5*16, 5*16])

fig_eq, ax_eq = plt.subplots(figsize=(6.2, 4.5))
ax_eq.plot(1/beta_lst, n_eq30, marker='o', label='Square lattice 30x30' )
ax_eq.plot(1/beta_lst, n_eq20, marker='o', label='Square lattice 20x20' )
ax_eq.plot(1/beta_lst, n_eq4, marker='o', label='Square lattice 4x4' )
ax_eq.set_xlabel('T [K]', fontsize=15)
ax_eq.set_ylabel('n° equil. steps', fontsize=15)
ax_eq.legend()
ax_eq.grid(True)
plt.show()





# -----------------------------------------------------------------------------
# CHESSBOARD STARTING CONFIGURATION: STUDY OF THE EQUILIBRATION TIME
# -----------------------------------------------------------------------------

Beta = 1
neqs = 0
nmcs = 10**6
n1 = n2 = 30
N= n1*n2

# Remember to change te settings in the "accumulation" function in Funz9 !
results = accumulation(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)

fig_m, ax_m = plt.subplots(figsize=(6.2, 4.5))
ax_m.plot(Nsteps_lst, results[0]/N, marker='o' )
ax_m.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_m.set_ylabel(r'$ M / N $', fontsize=15)
ax_m.grid(True)
plt.show()

fig_e, ax_e = plt.subplots(figsize=(6.2, 4.5))
ax_e.plot(Nsteps_lst, results[2]/N, marker='o' )
ax_e.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_e.set_ylabel(r'$ E / N $', fontsize=15)
ax_e.grid(True)
plt.show()



# (Eye-made estimatimation!)

beta_lst = np.asarray([0.25,0.35, 0.4, 0.425, 0.45, 0.5, 0.55, 0.65, 0.75,0.85, 0.95])

# For a 30x30 square lattice: 
n_eq30 = np.asarray([10*900, 20*900, 30*900, 50*900, 270*900, 200*900, 200*900, 150*900, 120*900, 90*900, 150*900])
# For a 20x20 square lattice: 0
n_eq20 = np.asarray([2*400, 40*400, 40*400, 70*400, 80*400, 100*400, 110*400, 150*400, 70*400, 40*400, 65*400])
# For a 20x20 square lattice: 1
n_eq20 = np.asarray([5*400, 15*400, 20*400, 25*400, 50*400, 100*400, 50*400, 70*400, 430*400, 50*400, 30*400])
# For a 4x4 square lattice:
n_eq4 = np.asarray([4*16, 7*16, 11*16, 3*16, 3*16, 3*16, 3*16, 6*16, 7*16, 13*16, 8*16])

fig_eq, ax_eq = plt.subplots(figsize=(6.2, 4.5))
ax_eq.plot(1/beta_lst, n_eq30, marker='o', label='Square lattice 30x30' )
ax_eq.plot(1/beta_lst, n_eq20, marker='o', label='Square lattice 20x20' )
ax_eq.plot(1/beta_lst, n_eq4, marker='o', label='Square lattice 4x4' )
ax_eq.set_xlabel('T [K]', fontsize=15)
ax_eq.set_ylabel('n° equil. steps', fontsize=15)
ax_eq.legend()
ax_eq.grid(True)
plt.show()


#  The estimation is UNSATISFACTORY, because with different runs the fluctuations
# are quite large. We cannot say that for all L= {30, 20, 4} the trend is
# similar to the one with random starting configuration.. However, it seems
# the equilibration time here is in general lower.






# -----------------------------------------------------------------------------
# OPEN BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------

Beta = 1.5
neqs = 0
nmcs = 10**7
n1 = n2 = 30
N = n1*n2


results_open = accumulation_open(n1, n2, Beta, neqs, nmcs)
Nsteps_lst = np.arange(int(nmcs/(n1*n2))+1)

fig_mo, ax_mo = plt.subplots(figsize=(6.2, 4.5))
ax_mo.plot(Nsteps_lst, results_open[0]/N, marker='o', label='Open Boundary Conditions')
ax_mo.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_mo.set_ylabel(r'$ M / N $', fontsize=15)
ax_mo.legend()
ax_mo.grid(True)
plt.show()

fig_eo, ax_eo = plt.subplots(figsize=(6.2, 4.5))
ax_eo.plot(Nsteps_lst, results_open[2]/N, marker='o', label='Open Boundary Conditions')
ax_eo.set_xlabel(r'$ MCsteps/{} $'.format(n1*n2), fontsize=15)
ax_eo.set_ylabel(r'$ E / N $', fontsize=15)
ax_eo.legend()
ax_eo.grid(True)
plt.show()






# -----------------------------------------------------------------------------
# PLOTTING PHYSICAL QUANTITIES: <|M|>/N, <E>/N, X/N, Cv/N
# -----------------------------------------------------------------------------

resultsT = T_variation(4, 4, 1, 4, 0.25, 10**3, 10**5, 100)
Nsteps_lst =  np.arange(1, 4, 0.25)

fig_mT, ax_mT = plt.subplots(figsize=(6.2, 4.5))
ax_mT.scatter(Nsteps_lst, resultsT[0], marker='o', s=50)
ax_mT.errorbar(Nsteps_lst, resultsT[0], yerr=resultsT[4], fmt='.',  capsize=5, color='black')
ax_mT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_mT.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=15)
ax_mT.grid(True)
plt.show()

fig_eT, ax_eT = plt.subplots(figsize=(6.2, 4.5))
ax_eT.scatter(Nsteps_lst, resultsT[1], marker='o', s=50)
ax_eT.errorbar(Nsteps_lst, resultsT[1], yerr=resultsT[5], fmt='.',  capsize=5, color='black')
ax_eT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_eT.set_ylabel(r'$ \langle E \rangle / N $', fontsize=15)
ax_eT.grid(True)
plt.show()


fig_cT, ax_cT = plt.subplots(figsize=(6.2, 4.5))
ax_cT.scatter(Nsteps_lst, resultsT[2], marker='o', s=50)
ax_cT.errorbar(Nsteps_lst, resultsT[2], yerr=resultsT[6], fmt='.',  capsize=5, color='black')
ax_cT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_cT.set_ylabel(r'$ c_V $', fontsize=15)
ax_cT.grid(True)
plt.show()


lst_cT, lst_scT = c_as_derivative(4,4,resultsT[1], 0.25, resultsT[5]) # Modify dT / it is = resultsT
fig_cTder, ax_cTder = plt.subplots(figsize=(6.2, 4.5))
ax_cTder.scatter(Nsteps_lst[1:len(Nsteps_lst)-1], lst_cT, marker='o', s=50, label='c as numerical derivative')
ax_cTder.errorbar(Nsteps_lst[1:len(Nsteps_lst)-1], lst_cT, yerr=lst_scT, fmt='.',  capsize=5, color='black')
ax_cTder.set_xlabel(r'$ T [K] $', fontsize=15)
ax_cTder.set_ylabel(r'$ c_V  $', fontsize=15)
ax_cTder.legend()
ax_cTder.grid(True)
plt.show()

fig_xT, ax_xT = plt.subplots(figsize=(6.2, 4.5))
ax_xT.scatter(Nsteps_lst, resultsT[3], marker='o', s=50)
ax_xT.errorbar(Nsteps_lst, resultsT[3], yerr=resultsT[7], fmt='.',  capsize=5, color='black')
ax_xT.set_xlabel(r'$ T [K] $', fontsize=15)
ax_xT.set_ylabel(r'$ \chi  $', fontsize=15)
ax_xT.grid(True)
plt.show()


# ---------------------------------
# FITTING ENERGY DATA WITH SIGMOID

"""
resultsT = T_variation(20, 20, 0.5, 5, 0.1, 10*5, 10**6, 100)
Nsteps_lst =  np.arange(0.5, 5, 0.1) 
def fitE(x, L, k, x0, c):
    return L / (1 + np.exp(-k * (x - x0))) + c

parE, covE = curve_fit(fitE, Nsteps_lst, resultsT[1], sigma=resultsT[5], p0=[-0.4, 1, 2.3, -1.5])
p1, p2, p3, p4 = parE

fig_E, ax_E = plt.subplots(figsize=(6.2, 4.5))
ax_E.scatter(Nsteps_lst, resultsT[1], marker='o', s=50)
ax_E.errorbar(Nsteps_lst, resultsT[1], yerr=resultsT[5], fmt='.', capsize=5, color='black')
ax_E.plot(Nsteps_lst, fitE(Nsteps_lst, p1, p2, p3, p4), label='Fit curve', color='crimson')
ax_E.set_xlabel(r'$ T [K] $', fontsize=15)
ax_E.set_ylabel(r'$ \langle E \rangle / N $', fontsize=15)
ax_E.legend()
ax_E.grid(True)
plt.show()
"""



#  L = 4, T in [1,4,0.1], Neq = 5*10**3, Nmc = 10**6

m_4 = np.asarray([0.99935201, 0.99846602, 0.99688805, 0.99479608, 0.99155014, 0.98636422, 0.98023032, 0.96826651, 0.9562327, 0.94020496, 0.92018328, 0.89786763, 0.85961625, 0.83793859, 0.80481712, 0.76536175, 0.73081231, 0.69155694, 0.66638534, 0.62950393, 0.60055439, 0.57446481, 0.55206917, 0.52730556, 0.50692789, 0.4877682, 0.47077847, 0.45572071, 0.44429689, 0.43168909])
e_4 = np.asarray([-1.99743204, -1.99406409, -1.98808019, -1.98016832, -1.9690605, -1.95107678, -1.93064911, -1.89504968, -1.85773428, -1.81206701, -1.75872786, -1.69917281, -1.60588631, -1.5499472, -1.47034047, -1.38186589, -1.30093119, -1.21680853, -1.15672149, -1.07899874, -1.01533575, -0.96040463, -0.91037743, -0.85711829, -0.81529896, -0.7747516, -0.73904018, -0.70746068, -0.68067311, -0.65273356])
c_4 = np.asarray([0.02072616, 0.0409483, 0.07010905, 0.10003814, 0.14133432, 0.19717004, 0.25077334, 0.35350717, 0.43119873, 0.51236687, 0.59854972, 0.66738, 0.76791469, 0.78014703, 0.79639642, 0.81013516, 0.79026185, 0.75892406, 0.71073349, 0.65724342, 0.6009056, 0.55145204, 0.50447891, 0.45200891, 0.41424525, 0.3772927, 0.34225364, 0.31416926, 0.28830309, 0.26093372])
x_4 = np.asarray([0.00133726, 0.00315481, 0.00612411, 0.00964808, 0.01739228, 0.02865087, 0.04148337, 0.07714436, 0.10181075, 0.14122718, 0.19291642, 0.23878523, 0.32535724, 0.35192606, 0.39640556, 0.44994968, 0.47077294, 0.49581935, 0.48840348, 0.49051471, 0.47725718, 0.46181534, 0.44301045, 0.42067518, 0.40542647, 0.38387445, 0.3638779, 0.34610825, 0.32825628, 0.30874602])
sm_4 = np.asarray([0.00044334, 0.00047166, 0.00054135, 0.00057493, 0.00104756, 0.00129101, 0.00139726, 0.00218803, 0.00196583, 0.00223317, 0.00222585, 0.00228847, 0.00216072, 0.00214942, 0.00178767, 0.0016643, 0.00141946, 0.00118743, 0.00092443, 0.00089072, 0.00074239, 0.0007012, 0.00065941, 0.0006992, 0.00065383, 0.00072294, 0.00066478, 0.00075841, 0.00075513, 0.00079739])
se_4 = np.asarray([0.00171878, 0.00164777, 0.001745, 0.00175101, 0.00228074, 0.00231775, 0.00242148, 0.00301325, 0.00317412, 0.00331104, 0.00313122, 0.00308335, 0.00279556, 0.00268614, 0.00234034, 0.00194998, 0.00181894, 0.00150436, 0.00133532, 0.00150591, 0.00163192, 0.00183958, 0.00165659, 0.00180485, 0.00200411, 0.00199987, 0.00209856, 0.00195204, 0.00233117, 0.00212748])
sc_4 = np.asarray([0.00089573, 0.00086257, 0.00095699, 0.00096926, 0.00149783, 0.00151173, 0.00157888, 0.00186378, 0.00175053, 0.00162923, 0.00139233, 0.0012034, 0.00095007, 0.0007977, 0.00067874, 0.00046551, 0.00034818, 0.00024783, 0.00018292, 0.00015993, 0.0001481, 0.0001708, 0.00014765, 0.00018587, 0.00020707, 0.0002036, 0.0002036, 0.00019749, 0.00022646, 0.00019742])
sx_4 = np.asarray([0.00170938, 0.00012306, 0.0016712, 0.01115278, 0.02536135, 0.02848054, 0.02365783, 0.03547872, 0.03634669, 0.03579542, 0.02722939, 0.02227716, 0.01138205, 0.00785788, 0.00461223, 0.00384337, 0.00324015, 0.00266696, 0.00216906, 0.00197793, 0.00164972, 0.00151457, 0.00137577, 0.00125039, 0.00136458, 0.0010967, 0.0010982, 0.00104154, 0.0008407, 0.00086902])



#  L = 10, T in [1,4,0.1], Neq = 10**5, Nmc = 10**6

m_10 = np.asarray([0.99927207, 0.99851615, 0.99679432, 0.9950005, 0.99134687, 0.98664934, 0.97939806, 0.96974703, 0.95860014, 0.93121488, 0.91382062, 0.87706629, 0.81255874, 0.72666133, 0.64041996, 0.53119288, 0.42850315, 0.38137586, 0.3370423, 0.3090291, 0.27224478, 0.25326667, 0.23993401, 0.22583342, 0.21241676, 0.20374963, 0.1890411, 0.18210979, 0.18390361, 0.16918108])
e_10 = np.asarray([-1.99712829, -1.99424858, -1.98769723, -1.98114189, -1.96785921, -1.95127687,-1.92741926, -1.89741826, -1.8629777, -1.80367163, -1.74994901, -1.67277672,-1.56373963, -1.43011699, -1.31222878, -1.17705029, -1.06280572, -0.98593741,-0.93149485, -0.87876412, -0.82464554, -0.78337766, -0.74558144, -0.72034397,-0.690007, -0.66744526, -0.63569243, -0.61239876, -0.599992, -0.57044296])
c_10 = np.asarray([0.02317293, 0.03984046, 0.07183634, 0.09585881, 0.1490538, 0.1888762, 0.27441588, 0.3435426, 0.41911279, 0.6266309, 0.70253593, 0.86240399, 1.10105877, 1.30553138, 1.26198356, 1.09770492, 0.96569999, 0.76923972, 0.61861996, 0.52354042, 0.4395069, 0.37211336, 0.34088392, 0.31084974, 0.27361148, 0.25633455, 0.23007455, 0.20770936, 0.19996671, 0.18695234])
x_10 = np.asarray([1.52285461e-03, 3.04314698e-03, 6.27958802e-03, 9.71306714e-03, 1.72808271e-02, 2.47109904e-02, 4.46203900e-02, 7.17967025e-02, 1.00112222e-01, 4.92559898e-01, 3.40612510e-01, 5.50138915e-01, 1.27275373e+00, 2.00900876e+00, 2.38977034e+00, 2.54960532e+00, 2.52025294e+00, 2.05542212e+00, 1.73067876e+00, 1.45995977e+00, 1.18727629e+00, 1.00002831e+00, 8.84145199e-01, 7.84595718e-01, 6.96796458e-01, 6.36013510e-01, 5.31486029e-01, 4.74745724e-01, 4.74039973e-01, 4.05733255e-01])
sm_10 = np.asarray([0.00020342, 0.00018567, 0.00024304, 0.000293, 0.00037125, 0.0003453,0.00068641, 0.0010717, 0.00125698, 0.00467858, 0.00309856, 0.00356532,0.00591956, 0.00708904, 0.00594799, 0.00435385, 0.0039524, 0.00361734,0.00312214, 0.00283247, 0.0028541, 0.00247706, 0.00237244, 0.00229696,0.00206074, 0.00200542, 0.00186601, 0.00153316, 0.00193269, 0.00162198])
se_10 = np.asarray([0.00077408, 0.00065529, 0.00077267, 0.00079054, 0.00096344, 0.00086058,0.00135544, 0.00158445, 0.00181264, 0.00303257, 0.00270725, 0.00305974,0.00327004, 0.00401525, 0.00347391, 0.0036745, 0.00396356, 0.00398895,0.00328951, 0.00288842, 0.00254371, 0.00207228, 0.00184622, 0.00193369,0.00189984, 0.00162828, 0.00159821, 0.0014401, 0.00148877, 0.00146612])
sc_10 = np.asarray([7.19905519e-05, 9.87500126e-05, 1.04079229e-04, 1.20646581e-04,1.79849040e-04, 1.45410766e-04, 2.73799809e-04, 2.90914448e-04, 3.60940499e-04, 7.49475450e-04, 5.57057263e-04, 5.30364215e-04, 5.62502859e-04, 4.64987591e-04, 3.03801735e-04, 2.98762032e-04, 4.34044514e-04, 3.76699566e-04, 3.72963769e-04, 2.91826864e-04, 2.21044716e-04, 1.60434747e-04, 1.39942688e-04, 1.19064830e-04, 1.37895726e-04, 1.01095780e-04, 8.79150584e-05, 5.95777259e-05,6.76516393e-05, 5.86880643e-05])
sx_10 = np.asarray([5.98555638e-06, 1.01806713e-05, 1.62933818e-05, 8.76623131e-04, 1.01829540e-03, 3.77295888e-05, 1.60173912e-03, 2.24900999e-03, 2.41519129e-03, 8.04362372e-03, 1.09205207e-03, 1.02869701e-03, 1.06243213e-02, 1.25451801e-02, 1.14544325e-02, 8.57927854e-03, 6.21549463e-03, 4.92444382e-03, 4.10955333e-03, 3.47359754e-03, 2.46941701e-03, 1.81800149e-03, 1.67301026e-03, 1.41568865e-03, 1.07362199e-03, 1.11416250e-03, 8.96050840e-04, 6.90635446e-04, 8.00602525e-04, 5.66108743e-04])



#  L = 15, T in [1,4,0.1], Neq = 2*10**5, Nmc = 10**6

m_15 = np.asarray([0.9991881, 0.99835621, 0.99704437, 0.99470666, 0.99133908, 0.98620372, 0.97894463, 0.97148356, 0.95586752, 0.93685389, 0.90621372, 0.86341707, 0.80015498, 0.69848369, 0.53053868, 0.40409049, 0.33270341, 0.26651769, 0.24044694, 0.21159455, 0.19503262, 0.17241945, 0.16040695, 0.14464692, 0.14262117, 0.13515211, 0.12610724, 0.12017598, 0.11398675, 0.11269691])
e_15 = np.asarray([-1.99684439, -1.99366079, -1.98859343, -1.98005049, -1.96788801, -1.94999025, -1.92624122, -1.90190826, -1.85714986, -1.80632021, -1.74226022, -1.66314611, -1.55509161, -1.41786877, -1.25894463, -1.14584277, -1.04423547, -0.96694913, -0.91545557, -0.86270616, -0.8223912, -0.77971654, -0.74200925, -0.71075716, -0.68906387, -0.66031546, -0.63497463, -0.61193351, -0.59419173, -0.57319035])
c_15 = np.asarray([0.02490809, 0.04278235, 0.06632616, 0.0965055, 0.14202489, 0.1899826, 0.2535521, 0.3185532, 0.43665384, 0.59330651, 0.78969821, 1.02073407, 1.14315288, 1.54816095, 1.31681918, 1.00717382, 0.86313636, 0.66529754, 0.5613071, 0.46014073, 0.404927, 0.36944718, 0.32315988, 0.29564641, 0.26312404, 0.24774657, 0.22660879, 0.20689942, 0.19414176, 0.18525813])
x_15 = np.asarray([1.71545171e-03, 3.38135952e-03, 5.63447079e-03, 9.93325069e-03, 1.60067885e-02, 2.63838840e-02, 4.26762988e-02, 6.30474330e-02, 1.16155536e-01, 2.20300844e-01, 6.64632338e-01, 1.43867909e+00, 2.23001925e+00, 3.37247403e+00, 5.22321023e+00, 5.12173091e+00, 3.89724927e+00, 2.82892046e+00, 2.23382365e+00, 1.63544131e+00, 1.35604632e+00, 1.08715187e+00, 9.34368121e-01, 7.72231357e-01, 6.98454218e-01, 6.76360204e-01, 5.47120172e-01, 4.69478891e-01, 4.30214427e-01, 4.11185039e-01])
sm_15 = np.asarray([0.00011998, 0.0001368, 0.00013794, 0.00020093, 0.00019048, 0.00030234, 0.00039546, 0.00059983, 0.00083046, 0.00147898, 0.00218899, 0.00349415, 0.00331049, 0.00494477, 0.00452709, 0.00453313, 0.00489711, 0.00407101, 0.00393887, 0.0030154, 0.00288504, 0.00267972, 0.00251336, 0.00255102, 0.00205543, 0.00227254, 0.00174961, 0.00159558, 0.00148226, 0.00144875])
se_15 = np.asarray([0.00043961, 0.000442, 0.00047403, 0.00054792, 0.00059868, 0.00070333, 0.00078524, 0.00112374, 0.00126901, 0.00182401, 0.00201529, 0.00221347, 0.00236913, 0.003291, 0.00266856, 0.00273582, 0.00283557, 0.00232455, 0.0024684, 0.00191933, 0.00169384, 0.00182739, 0.00158679, 0.00160852, 0.0014181, 0.00157676, 0.00129087, 0.00135919, 0.00138123, 0.00126268])
sc_15 = np.asarray([1.87301336e-05, 2.64264935e-05, 4.55696818e-05, 4.56825268e-05, 5.72522584e-05, 6.25800302e-05, 7.36511213e-05, 1.00584154e-04, 1.28224655e-04, 2.31622724e-04, 2.80901067e-04, 3.04925839e-04, 2.47907982e-04, 2.95879759e-04, 2.46815631e-04, 2.41408790e-04, 2.60015339e-04, 1.89312566e-04, 1.57130556e-04, 1.05714937e-04, 8.07575365e-05, 7.20473448e-05, 7.16125415e-05, 5.14419864e-05, 4.25139731e-05, 5.56974455e-05, 4.69527547e-05, 3.49537656e-05, 3.17092264e-05, 3.22351983e-05])
sx_15 = np.asarray([4.78087573e-04, 4.93803934e-04, 4.54613628e-04, 6.08190592e-04, 7.72286356e-06, 7.79793642e-04, 9.44586778e-04, 4.81080301e-05, 9.34682574e-05, 2.72446495e-04, 3.39323328e-03, 4.61452817e-03, 5.06883341e-03, 6.64041094e-03, 5.34942637e-03, 3.97861439e-03, 3.82192674e-03, 2.62249377e-03, 2.36304215e-03, 1.80447715e-03, 1.42484538e-03, 1.21324999e-03, 1.08394680e-03, 8.15518063e-04, 6.90759651e-04, 7.31747903e-04, 5.68363272e-04, 4.62748341e-04, 3.60637203e-04, 3.93082552e-04])



#  L = 20, T in [1,4,0.1], Neq = 5*10**5, Nmc = 10**6

m_20 = np.asarray([0.99934226, 0.99854258, 0.99713515, 0.99471611, 0.99141144, 0.98711116, 0.97935626, 0.96914834, 0.9559996, 0.94016194, 0.91642943, 0.87195322, 0.78472411, 0.64507797, 0.36519992, 0.3259996, 0.25218513, 0.19093163, 0.17986006, 0.15485206, 0.13316673, 0.12508996, 0.12097761, 0.11080768, 0.10489604, 0.1027529, 0.09441623, 0.09710916, 0.08660736, 0.08519592])
e_20 = np.asarray([-1.99738105, -1.99434226, -1.98890444, -1.98006797, -1.96807677, -1.95314674, -1.9269972, -1.89742903, -1.85790084, -1.81361056, -1.75159536, -1.66193123, -1.55339464, -1.39520592, -1.21872451, -1.12132747, -1.03454618, -0.96627749, -0.91310276, -0.86227109, -0.8217513, -0.77872451, -0.74359056, -0.71365454, -0.69253898, -0.66311475, -0.63672131, -0.61910436, -0.5917593, -0.57726909])
c_20 = np.asarray([0.01969546, 0.04153618, 0.06181951, 0.09824127, 0.15014804, 0.18344557, 0.26147972, 0.37207919, 0.4268554, 0.56094113, 0.63780626, 0.8678387, 1.41568098, 1.70785732, 1.26944864, 0.98538486, 0.75423109, 0.64920958, 0.50407434, 0.43781964, 0.4362438, 0.37188756, 0.32631255, 0.30182524, 0.27476037, 0.25249572, 0.23497903, 0.21892185, 0.19960392, 0.18722765])
x_20 = np.asarray([1.25438189e-03, 3.12060120e-03, 5.15771300e-03, 1.00511722e-02, 1.69380971e-02, 2.45611727e-02, 4.07800454e-02, 8.78882297e-02, 1.15496113e-01, 1.78363132e-01, 2.45958800e-01, 6.13452412e-01, 3.90299950e+00, 6.57266553e+00, 9.72100160e+00, 6.08632393e+00, 4.17980397e+00, 2.88288118e+00, 2.34651574e+00, 1.88860981e+00, 1.45020496e+00, 1.12726289e+00, 9.66927466e-01, 8.36754680e-01, 7.44595008e-01, 6.82413722e-01, 5.31995988e-01, 5.50802347e-01, 4.62191478e-01, 4.03541312e-01])
sm_20 = np.asarray([7.41897861e-05, 1.06659674e-04, 1.02445852e-04, 1.52598697e-04, 2.04267874e-04, 2.10332832e-04, 2.68092795e-04, 6.32553179e-04, 6.43579642e-04, 7.02092376e-04, 7.03447615e-04, 1.38253563e-03, 2.01853186e-03, 3.05223568e-03, 3.17194077e-03, 3.21955678e-03, 3.23153175e-03, 3.18618102e-03, 2.72283655e-03, 2.96738304e-03, 2.81853556e-03, 2.12405410e-03, 2.07586591e-03, 1.95050028e-03, 2.02466501e-03, 1.92312604e-03, 1.53232678e-03, 1.62362037e-03, 1.53417115e-03, 1.29708260e-03])
se_20 = np.asarray([0.00029019, 0.00039622, 0.00036434, 0.00042352, 0.0006247, 0.00055307, 0.00073343, 0.00116834, 0.00122353, 0.00131255, 0.0012487, 0.00182411, 0.00193646, 0.00274224, 0.00203978, 0.002068, 0.00192838, 0.00195223, 0.00185138, 0.00163243, 0.00178452, 0.0014398, 0.00141414, 0.00123702, 0.0014529, 0.00159869, 0.00146043, 0.0011871, 0.0012323, 0.0012787 ])
sc_20 = np.asarray([7.77662357e-06, 1.72554572e-05, 1.92502527e-05, 2.89408599e-05, 4.59049037e-05, 3.06608089e-05, 5.14224304e-05, 7.11178511e-05, 6.13747015e-05, 9.42201237e-05, 8.15290816e-05, 1.34914863e-04, 1.92514553e-04, 1.96316348e-04, 1.90568191e-04, 1.50135471e-04, 1.46349189e-04, 9.06980503e-05, 7.16250142e-05, 6.22531533e-05, 6.20218389e-05, 4.15137283e-05, 4.26891550e-05, 3.78481542e-05, 3.16693531e-05, 3.37970679e-05, 3.33212509e-05, 2.61549007e-05, 2.27077932e-05, 2.10826014e-05])
sx_20 = np.asarray([5.70182977e-07, 3.86211663e-04, 3.39079522e-04, 4.32310158e-06, 5.72508327e-04, 7.01950713e-06, 6.47621613e-04, 3.59235500e-05, 1.33351186e-03, 1.33116192e-03, 5.53528096e-05, 1.67166975e-04, 2.39757297e-03, 1.09090754e-03, 2.65901486e-03, 2.13659699e-03, 1.75915990e-03, 1.25158477e-03, 1.12069502e-03, 1.12297834e-03, 9.06236947e-04, 6.74149433e-04, 5.20191298e-04, 4.72530702e-04, 3.55124194e-04, 4.13167065e-04, 3.08018779e-04, 3.58184669e-04, 2.62070784e-04, 2.46070374e-04])



#  L = 30, T in [1,4,0.1], Neq = 10**6, Nmc = 10**6

m_30 = np.asarray([0.99929856, 0.9986251, 0.99705835, 0.99447842, 0.99128697, 0.98596523, 0.97917466, 0.97052958, 0.95752198, 0.93930855, 0.91175659, 0.87466627, 0.73101519, 0.64745803, 0.33747602, 0.2217446, 0.14745604, 0.12372502, 0.11581135, 0.09302358, 0.08819944, 0.08328537, 0.07230416, 0.07222022, 0.07529576, 0.06853717, 0.06256595, 0.05892486, 0.05743605, 0.05502798])
e_30 = np.asarray([-1.99723421, -1.99460831, -1.98873301, -1.97928058, -1.9677458, -1.94967626, -1.92650679, -1.89926859, -1.86016787, -1.81372502, -1.74288969, -1.67031575, -1.51219424, -1.40039169, -1.21366906, -1.11083533, -1.02734213, -0.9604996, -0.89926459, -0.85904077, -0.80647882, -0.77647482, -0.74567146, -0.71445244, -0.68568745, -0.65609512, -0.63714628, -0.61513189, -0.59367706, -0.57601519])
c_30 = np.asarray([0.02090115, 0.03623532, 0.06506608, 0.10071813, 0.14494888, 0.19313331, 0.27211082, 0.29916487, 0.4236039, 0.55073664, 0.69019153, 0.93083769, 1.45522245, 1.84199669, 1.47666771, 0.99005268, 0.62701866, 0.64821876, 0.5276015, 0.48406117, 0.38441072, 0.3574388, 0.31307336, 0.30679515, 0.25018327, 0.26294146, 0.22185899, 0.20428583, 0.19148487, 0.17818446])
x_30 = np.asarray([1.36773675e-03, 2.66089287e-03, 5.69362256e-03, 1.04954194e-02, 1.65097521e-02, 2.78396156e-02, 4.41487076e-02, 5.66860583e-02, 1.01552683e-01, 2.11629311e-01, 2.88547520e-01, 8.05250885e-01, 5.95785177e+00, 8.49714833e+00, 1.82254205e+01, 9.39213051e+00, 2.94713873e+00, 3.01281264e+00, 2.13405457e+00, 1.86086082e+00, 1.44693840e+00, 1.30999392e+00, 8.51494415e-01, 8.75861734e-01, 8.47932010e-01, 6.48984976e-01, 4.90637929e-01, 4.49361277e-01, 4.22652093e-01, 3.88627891e-01])
sm_30 = np.asarray([5.07141062e-05, 5.78841428e-05, 7.48136627e-05, 1.05561197e-04, 1.21372385e-04, 1.80252409e-04, 2.05578290e-04, 2.53922387e-04, 3.98221400e-04, 4.41054796e-04, 6.93384751e-04, 7.59514451e-04, 1.30658748e-03, 1.57553547e-03, 1.74133446e-03, 1.70154809e-03, 1.47327005e-03, 2.05154795e-03, 1.54648145e-03, 1.68944323e-03, 1.65898644e-03, 1.69941653e-03, 1.45318507e-03, 1.48895530e-03, 1.57688186e-03, 1.29457846e-03, 1.08727991e-03, 1.18843639e-03, 1.28899300e-03, 9.71012574e-04])
se_30 = np.asarray([0.0001945, 0.00021303, 0.00025172, 0.00033369, 0.00038853, 0.00053264, 0.000606, 0.0007173, 0.00099127, 0.00100872, 0.00147552, 0.00147668, 0.00147985, 0.00185151, 0.00176873, 0.00158134, 0.00129338, 0.00144777, 0.00135236, 0.00146991, 0.00129365, 0.0012127, 0.00125029, 0.00121497, 0.00122675, 0.00145999, 0.00110678, 0.00106901, 0.00109117, 0.00129294])
sc_30 = np.asarray([3.22778609e-06, 5.75032390e-06, 6.12520211e-06, 1.01606426e-05, 1.14987804e-05, 2.03111059e-05, 1.80540383e-05, 2.06048839e-05, 3.62615821e-05, 3.81575071e-05, 4.98049662e-05, 7.72674183e-05, 1.00002929e-04, 1.23653394e-04, 1.06869256e-04, 7.70575036e-05, 4.22565135e-05, 4.76135962e-05, 3.97490936e-05, 3.52370724e-05, 2.70747952e-05, 2.32850159e-05, 2.07698321e-05, 1.93272760e-05, 1.79839280e-05, 2.27271691e-05, 1.56391435e-05, 1.19633881e-05, 1.21059814e-05, 1.44794882e-05])
sx_30 = np.asarray([2.21731884e-07, 4.60594378e-07, 7.05339062e-07, 1.34910463e-06, 1.49482785e-06, 3.99408375e-06, 3.60275501e-06, 5.76832781e-04, 8.38321121e-04, 2.72748501e-05, 2.10914738e-05, 7.97017957e-05, 2.18626112e-04, 3.74961433e-04, 1.16797878e-03, 9.33460162e-04, 5.59609346e-04, 5.66447909e-04, 4.37368162e-04, 3.95384364e-04, 2.95673195e-04, 2.98006191e-04, 2.03064355e-04, 2.07667520e-04, 2.43410483e-04, 1.58412691e-04, 1.27586924e-04, 1.18504484e-04, 1.36351797e-04, 1.07650780e-04])




# --------------------------------
# COLLECTIVE PLOTS

Nsteps_lst =  np.arange(1, 4, 0.1)
fig_mTOT, ax_mTOT = plt.subplots(figsize=(6.2, 4.5))
ax_mTOT.plot(Nsteps_lst, m_4, marker='o', label='L = 4' )
ax_mTOT.plot(Nsteps_lst, m_10, marker='o', label='L = 10' )
ax_mTOT.plot(Nsteps_lst, m_15, marker='o', label='L = 15' )
ax_mTOT.plot(Nsteps_lst, m_20, marker='o', label='L = 20' )
ax_mTOT.plot(Nsteps_lst, m_30, marker='o', label='L = 30' )
ax_mTOT.set_xlabel('T [K]', fontsize=15)
ax_mTOT.set_ylabel(r'$ \langle |M| \rangle / N $', fontsize=15)
ax_mTOT.legend()
ax_mTOT.grid(True)
plt.show()


fig_eTOT, ax_eTOT = plt.subplots(figsize=(6.2, 4.5))
ax_eTOT.plot(Nsteps_lst, e_4, marker='o', label='L = 4' )
ax_eTOT.plot(Nsteps_lst, e_10, marker='o', label='L = 10' )
ax_eTOT.plot(Nsteps_lst, e_15, marker='o', label='L = 15' )
ax_eTOT.plot(Nsteps_lst, e_20, marker='o', label='L = 20' )
ax_eTOT.plot(Nsteps_lst, e_30, marker='o', label='L = 30' )
ax_eTOT.set_xlabel('T [K]', fontsize=15)
ax_eTOT.set_ylabel(r'$ \langle E \rangle / N $', fontsize=15)
ax_eTOT.legend()
ax_eTOT.grid(True)
plt.show()


fig_cTOT, ax_cTOT = plt.subplots(figsize=(6.2, 4.5))
ax_cTOT.plot(Nsteps_lst, c_4, marker='o', label='L = 4' )
ax_cTOT.plot(Nsteps_lst, c_10, marker='o', label='L = 10' )
ax_cTOT.plot(Nsteps_lst, c_15, marker='o', label='L = 15' )
ax_cTOT.plot(Nsteps_lst, c_20, marker='o', label='L = 20' )
ax_cTOT.plot(Nsteps_lst, c_30, marker='o', label='L = 30' )
ax_cTOT.set_xlabel('T [K]', fontsize=15)
ax_cTOT.set_ylabel(r'$ c_V $', fontsize=15)
ax_cTOT.legend()
ax_cTOT.grid(True)
plt.show()


fig_xTOT, ax_xTOT = plt.subplots(figsize=(6.2, 4.5))
ax_xTOT.plot(Nsteps_lst, x_4, marker='o', label='L = 4' )
ax_xTOT.plot(Nsteps_lst, x_10, marker='o', label='L = 10' )
ax_xTOT.plot(Nsteps_lst, x_15, marker='o', label='L = 15' )
ax_xTOT.plot(Nsteps_lst, x_20, marker='o', label='L = 20' )
ax_xTOT.plot(Nsteps_lst, x_30, marker='o', label='L = 30' )
ax_xTOT.set_xlabel('T [K]', fontsize=15)
ax_xTOT.set_ylabel(r'$ \chi $', fontsize=15)
ax_xTOT.legend()
ax_xTOT.grid(True)
plt.show()






# -----------------------------------------------------------------------------
# CONFIGURATION SEQUENCE ANIMATION
# -----------------------------------------------------------------------------

animation_Ising(30, 30, 1.5, 200000, 'Ising_2D.gif')
plt.close('all')











