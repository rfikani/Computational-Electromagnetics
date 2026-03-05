import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
##defs approx 
epszero = 8.854187817e-12 
muzero = 4.0 * np.pi * 1e-7
czero = 1.0 / np.sqrt(muzero * epszero)
freqzero = 2.4e9
S = 0.95
src_amp = 1e-3
pmlpow = 3
pmlreflect = 1e-8
## builds 1D sigama profile used for perf_match conductivity
def sigama_side(n, n_npml, smax_v_xx):
    s = np.zeros(n)
    for i in range(n):
        if i < n_npml:
            x = (n_npml - i) / n_npml
            s[i] = smax_v_xx * x**pmlpow
        elif i >= n - n_npml:
            x = (i - (n - n_npml - 1)) / n_npml
            s[i] = smax_v_xx * x**pmlpow
            
    return s
def drive_f(t, dx, dy, taw): 
    J0 = src_amp / (dx * dy)
    return J0 * (1.0 - np.exp(-t / taw)) * np.sin(2.0 * np.pi * freqzero * t)
def steptmz(Ez, Hx, Hy, Esxx_test, Esyy, pec_pol, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, srcij, t_half):
    n_xxx, nyy = Ez.shape
    d1 = Ez[:, 1:] - Ez[:, :-1]
    pmlhx = perf_match[:, :-1] | perf_match[:, 1:]
    Hx[~pmlhx] -= (dt / (muzero * dy)) * d1[~pmlhx]
    ayh = 0.5 * (ay[:, :-1] + ay[:, 1:])   
    byyh = 0.5 * (byy[:, :-1] + byy[:, 1:]) 
    epsh = 0.5 * (eps[:, :-1] + eps[:, 1:])  
    Hx[pmlhx] = (ayh[pmlhx] * Hx[pmlhx] - (epsh[pmlhx] / (muzero * dy)) * d1[pmlhx]) / byyh[pmlhx]
    dtwo = Ez[1:, :] - Ez[:-1, :]
    
    pmlhhy = perf_match[:-1, :] | perf_match[1:, :]

    Hy[~pmlhhy] += (dt / (muzero * dx)) * dtwo[~pmlhhy]
    axhh = 0.5 * (ax_y[:-1, :] + ax_y[1:, :])
    
    bxh = 0.5 * (bx[:-1, :] + bx[1:, :])
    
    epsh2 = 0.5 * (eps[:-1, :] + eps[1:, :])
    Hy[pmlhhy] = (axhh[pmlhhy] * Hy[pmlhhy] + (epsh2[pmlhhy] / (muzero * dx)) * dtwo[pmlhhy]) / bxh[pmlhhy]
    dHy_dx = np.zeros((n_xxx, nyy))
    dHx_dy = np.zeros((n_xxx, nyy))
    dHy_dx[1:-1, :] = (Hy[1:, :] - Hy[:-1, :]) / dx
    dHx_dy[:, 1:-1] = (Hx[:, 1:] - Hx[:, :-1]) / dy

    Esxx_test[perf_match] = (ax_y[perf_match] * Esxx_test[perf_match] + dHy_dx[perf_match]) / bx[perf_match]
    Esyy[perf_match] = (ay[perf_match] * Esyy[perf_match] - dHx_dy[perf_match]) / byy[perf_match]
    Ez[perf_match] = Esxx_test[perf_match] + Esyy[perf_match]

    inside_r = ~perf_match
    a_zero = (eps / dt) - siga / 2.0
    b0 = (eps / dt) + siga / 2.0

    J = np.zeros((n_xxx, nyy))
    if srcij is not None:
        J[srcij[0], srcij[1]] = drive_f(t_half, dx, dy, taw)
    curll = dHy_dx - dHx_dy
    Ez[inside_r] = (a_zero[inside_r] * Ez[inside_r] + curll[inside_r] - J[inside_r]) / b0[inside_r]
    Ez[pec_pol] = 0.0
    Esxx_test[pec_pol] = 0.0
    Esyy[pec_pol] = 0.0
# ----------------validation---------
n_xxx = 240
nyy = 240
n_npml = 20
ppww = 40.0
tmax = 2.167e-9
taw = tmax / 7.0
lam = czero / freqzero
dx = lam / ppww
dy = dx
dt = S / (czero * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
nt = int(np.ceil(tmax / dt))

Ez = np.zeros((n_xxx, nyy))

Hx = np.zeros((n_xxx, nyy - 1))

Hy = np.zeros((n_xxx - 1, nyy))

Esxx_test = np.zeros((n_xxx, nyy))

Esyy = np.zeros((n_xxx, nyy))

eps = epszero * np.ones((n_xxx, nyy))
siga = np.zeros((n_xxx, nyy))
pec_pol = np.zeros((n_xxx, nyy), dtype=bool)

smax_v_xx = -(pmlpow + 1) * epszero * czero * np.log(pmlreflect) / (2.0 * n_npml * dx)

sx1 = sigama_side(n_xxx, n_npml, smax_v_xx)

syy1 = sigama_side(nyy, n_npml, smax_v_xx)

sx = sx1[:, None] * np.ones((1, nyy))
syy = np.ones((n_xxx, 1)) * syy1[None, :]
perf_match = np.zeros((n_xxx, nyy), dtype=bool)
perf_match[:n_npml, :] = True
perf_match[-n_npml:, :] = True

perf_match[:, :n_npml] = True
perf_match[:, -n_npml:] = True
ax_y = (eps / dt) - sx / 2.0
bx = (eps / dt) + sx / 2.0

ay = (eps / dt) - syy / 2.0
byy = (eps / dt) + syy / 2.0
srci = n_xxx // 2
src_j = nyy // 2
for n in range(nt):
    th = (n + 0.5) * dt
    steptmz(Ez, Hx, Hy, Esxx_test, Esyy, pec_pol, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, (srci, src_j), th)
Ezval = Ez.copy()
plt.figure(figsize=(7, 5))
plt.imshow(Ezval, origin="lower", extent=[0, nyy * dy, 0, n_xxx * dx], aspect="auto")
cbee = plt.colorbar()
cbee.set_label("Ez (V/m)")
plt.xlabel("y (m)")
plt.ylabel("x (m)")
plt.title("Validation field")
plt.tight_layout()
xrel = (np.arange(n_xxx) - srci) * dx
rho = np.abs(xrel) + 1e-12
w = 2.0 * np.pi * freqzero
k = w * np.sqrt(muzero * epszero)
Ez_w = -(k**2 * src_amp) / (4.0 * w * epszero) * hankel2(0, k * rho)
tf = (nt - 1) * dt
Ezrref = np.real(Ez_w * np.exp(1j * (w * tf - np.pi / 2.0)))
Ez_num = Ezval[:, src_j].copy()
g00d = rho > 5.0 * dx
scale = np.max(np.abs(Ezrref[g00d])) / (np.max(np.abs(Ez_num[g00d])) + 1e-30)
xabs = np.arange(n_xxx) * dx
plt.tight_layout()
n_xxx = 320
nyy = 320
n_npml = 20
ppww = 25.0
tmax = 4.167e-9
taw = tmax / 7.0
lam = czero / freqzero
dx = lam / ppww
dy = dx
dt = S / (czero * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
nt = int(np.ceil(tmax / dt))
eps = epszero * np.ones((n_xxx, nyy))
siga = np.zeros((n_xxx, nyy))
smax_v_xx = -(pmlpow + 1) * epszero * czero * np.log(pmlreflect) / (2.0 * n_npml * dx)
sxoneD = sigama_side(n_xxx, n_npml, smax_v_xx)
syy1d = sigama_side(nyy, n_npml, smax_v_xx)
sx = sxoneD[:, None] * np.ones((1, nyy))
syy = np.ones((n_xxx, 1)) * syy1d[None, :]
perf_match = np.zeros((n_xxx, nyy), dtype=bool)
perf_match[:n_npml, :] = True
perf_match[-n_npml:, :] = True
perf_match[:, :n_npml] = True
perf_match[:, -n_npml:] = True

ax_y = (eps / dt) - sx / 2.0
bx = (eps / dt) + sx / 2.0
ay = (eps / dt) - syy / 2.0
byy = (eps / dt) + syy / 2.0

Ezone = np.zeros((n_xxx, nyy))
Hx1 = np.zeros((n_xxx, nyy - 1))
Hy1 = np.zeros((n_xxx - 1, nyy))
Esxx_test1 = np.zeros((n_xxx, nyy))
Esyy1 = np.zeros((n_xxx, nyy))
Ez2 = np.zeros((n_xxx, nyy))
Hx2 = np.zeros((n_xxx, nyy - 1))
Hy2 = np.zeros((n_xxx - 1, nyy))
Esxx_test2 = np.zeros((n_xxx, nyy))
Esyy2 = np.zeros((n_xxx, nyy))
pec1 = np.zeros((n_xxx, nyy), dtype=bool) 
pec2 = np.zeros((n_xxx, nyy), dtype=bool)
sheetjj = nyy // 2
pec1[:, sheetjj:sheetjj + 1] = True ##f
pec2[:, sheetjj:sheetjj + 1] = True
slot_len = 0.07
half_oo = max(1, int(round((slot_len / 2.0) / dx)))
c = n_xxx // 2

pec1[max(0, c - half_oo):min(n_xxx, c + half_oo + 1), sheetjj:sheetjj + 1] = False
cA = n_xxx // 3
cbee = 2 * n_xxx // 3

pec2[max(0, cA - half_oo):min(n_xxx, cA + half_oo + 1), sheetjj:sheetjj + 1] = False
pec2[max(0, cbee - half_oo):min(n_xxx, cbee + half_oo + 1), sheetjj:sheetjj + 1] = False
srci = n_xxx // 2
src_j = nyy // 4

y_obs = int(round(0.75 * nyy))
steps_per_period = max(1, int(round((1.0 / freqzero) / dt)))
keep = 10 * steps_per_period
grabone = []
grab2 = []

for n in range(nt):
    th = (n + 0.5) * dt
    steptmz(Ezone, Hx1, Hy1, Esxx_test1, Esyy1, pec1, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, (srci, src_j), th)
    steptmz(Ez2, Hx2, Hy2, Esxx_test2, Esyy2, pec2, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, (srci, src_j), th)
    
    if n >= nt - keep:
        grabone.append(Ezone[:, y_obs].copy())
        grab2.append(Ez2[:, y_obs].copy())
        
Aone = np.stack(grabone, axis=0)
A2 = np.stack(grab2, axis=0)
I0ne = np.mean(Aone**2, axis=0)##
I2 = np.mean(A2**2, axis=0)
plt.figure(figsize=(7, 5))
plt.imshow(Ezone, origin="lower", extent=[0, nyy * dy, 0, n_xxx * dx], aspect="auto")
cbee = plt.colorbar()  ##maybe  thats why color map off check 
cbee.set_label("Ez (V/m)")
plt.title("Single slot")
plt.xlabel("y (m)")
plt.ylabel("x (m)")
plt.tight_layout()
plt.figure(figsize=(7, 5))
plt.imshow(Ez2, origin="lower", extent=[0, nyy * dy, 0, n_xxx * dx], aspect="auto")
cbee = plt.colorbar()
cbee.set_label("Ez (V/m)")
plt.title("Two slots")
plt.xlabel("y (m)")
plt.ylabel("x (m)")
plt.tight_layout()
x = np.arange(n_xxx) * dx
plt.figure(figsize=(7, 5))
plt.plot(x, I0ne, linewidth=1.6, label="Single slot") ##
plt.plot(x, I2, linewidth=1.6, label="Two slots")##
plt.xlabel("x (m)")
plt.ylabel("|Ez|^2 (V^2/m^2)")
plt.title("Diffraction comparison")
plt.legend()
plt.tight_layout()
n_xxx = 320
nyy = 320
n_npml = 20
ppww = 25.0
tmax = 4.167e-9
taw = tmax / 7.0
lam = czero / freqzero
###
dx = lam / ppww
dy = dx
dt = S / (czero * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))
####
nt = int(np.ceil(tmax / dt))
eps = epszero * np.ones((n_xxx, nyy))
siga = np.zeros((n_xxx, nyy))
smax_v_xx = -(pmlpow + 1) * epszero * czero * np.log(pmlreflect) / (2.0 * n_npml * dx)
sxoneD = sigama_side(n_xxx, n_npml, smax_v_xx)
syy1d = sigama_side(nyy, n_npml, smax_v_xx)
sx = sxoneD[:, None] * np.ones((1, nyy))
syy = np.ones((n_xxx, 1)) * syy1d[None, :]
####
perf_match = np.zeros((n_xxx, nyy), dtype=bool)
perf_match[:n_npml, :] = True
perf_match[-n_npml:, :] = True
perf_match[:, :n_npml] = True
perf_match[:, -n_npml:] = True

ax_y = (eps / dt) - sx / 2.0
bx = (eps / dt) + sx / 2.0

ay = (eps / dt) - syy / 2.0
byy = (eps / dt) + syy / 2.0
Ez_c = np.zeros((n_xxx, nyy))
Hx_c = np.zeros((n_xxx, nyy - 1))

Hy_c = np.zeros((n_xxx - 1, nyy))
Esxx_test_c = np.zeros((n_xxx, nyy))
Esyy_c = np.zeros((n_xxx, nyy))

Ezrr = np.zeros((n_xxx, nyy))
Hx_r = np.zeros((n_xxx, nyy - 1))
Hy_r = np.zeros((n_xxx - 1, nyy))

Esxx_test_r = np.zeros((n_xxx, nyy))
Esyy_r = np.zeros((n_xxx, nyy))

srci = n_xxx // 2
src_j = nyy // 4
cxx = n_xxx // 2
cy = nyy // 2 + nyy // 10
r_r = 0.06 / dx
i_i = np.arange(n_xxx)[:, None]
jj = np.arange(nyy)[None, :]
pec_c = (i_i - cxx) ** 2 + (jj - cy) ** 2 <= r_r**2
pec_r = np.zeros((n_xxx, nyy), dtype=bool)
sxr = int(round(0.10 / dx))
syyr = int(round(0.06 / dy))
xnau = cxx - sxr // 2
yano = cy - syyr // 2
pec_r[xnau:xnau + sxr, yano:yano + syyr] = True ##keep true here

for n in range(nt):
    th = (n + 0.5) * dt
    steptmz(Ez_c, Hx_c, Hy_c, Esxx_test_c, Esyy_c, pec_c, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, (srci, src_j), th)
    steptmz(Ezrr, Hx_r, Hy_r, Esxx_test_r, Esyy_r, pec_r, perf_match, eps, siga, ax_y, bx, ay, byy, dx, dy, dt, (srci, src_j), th)
plt.figure(figsize=(7, 5))
plt.imshow(Ez_c, origin="lower", extent=[0, nyy * dy, 0, n_xxx * dx], aspect="auto")
cbee = plt.colorbar()
cbee.set_label("Ez (V/m)")

plt.title("PEC circle")
plt.xlabel("y (m)")
plt.ylabel("x (m)")
plt.tight_layout()

plt.figure(figsize=(7, 5))
plt.imshow(Ezrr, origin="lower", extent=[0, nyy * dy, 0, n_xxx * dx], aspect="auto")
cbee = plt.colorbar()
cbee.set_label("Ez (V/m)")

plt.title("PEC rectangle")
plt.xlabel("y (m)")
plt.ylabel("x (m)")
plt.tight_layout()
theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
ring_r = 0.18 / dx
x_s = cxx + ring_r * np.cos(theta)

ys = cy + ring_r * np.sin(theta)

xnau = np.floor(x_s).astype(int)

yano = np.floor(ys).astype(int)

x1 = np.clip(xnau + 1, 0, n_xxx - 1)
y1 = np.clip(yano + 1, 0, nyy - 1)
xnau = np.clip(xnau, 0, n_xxx - 1)
yano = np.clip(yano, 0, nyy - 1)

wx_x = x_s - xnau
wy = ys - yano
v0o = Ez_c[xnau, yano]
v10 = Ez_c[x1, yano]
v01 = Ez_c[xnau, y1]
v_11 = Ez_c[x1, y1]
Ic = np.abs((1 - wx_x) * (1 - wy) * v0o + wx_x * (1 - wy) * v10 + (1 - wx_x) * wy * v01 + wx_x * wy * v_11) ** 2
v0o = Ezrr[xnau, yano]
v10 = Ezrr[x1, yano]
v01 = Ezrr[xnau, y1]
v_11 = Ezrr[x1, y1]
Ir = np.abs((1 - wx_x) * (1 - wy) * v0o + wx_x * (1 - wy) * v10 + (1 - wx_x) * wy * v01 + wx_x * wy * v_11) ** 2
plt.figure(figsize=(7, 5))
plt.plot(theta * 180.0 / np.pi, Ic, linewidth=1.6, label="Circular PEC")
plt.plot(theta * 180.0 / np.pi, Ir, linewidth=1.6, label="Rectangular PEC")
###fix color map for plot 
plt.xlabel("Angle (deg)")
plt.ylabel("|Ez|^2 (V^2/m^2)")

plt.title("Angular scattering")

plt.legend()
plt.tight_layout()
plt.show()