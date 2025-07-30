import math
import numpy
from matplotlib import pyplot, cm

# 1. Match domain size and grid to paper (assume 0 <= x <= 6.4, 0 <= y <= 6.4 for a 100x100 grid)
Lx = 6.4
Ly = 6.4
nx = 101
ny = 101
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = numpy.linspace(0, Lx, nx)
y = numpy.linspace(0, Ly, ny)
X, Y = numpy.meshgrid(x, y)

# 2. Physical parameters (set as per paper, adjust as needed)
rho = 1.0
nu = 0.01
k = 1.0
c = 1.0
L = 1.0
T0 = 0.5
g = -9.81
beta = 0.01
omega = 0.1
Ap = 1.0
Ap0 = 1.0

# 3. Time stepping
dt = 0.0001
nt = 1000000  # to reach t=100s with dt=0.001

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    pn = numpy.empty_like(p)
    for it in range(50):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                          b[1:-1,1:-1])
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0
    return p

def velocity_update(u, v, dt, dx, dy, p, nu, b, T, delH, L, fl, beta, g, T0):
    un = u.copy()
    vn = v.copy()
    # u-momentum
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 *
                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 *
                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
    # v-momentum with buoyancy
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     dt / dy**2 *
                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])) +
                     dt * (rho * g * beta * ( T[1:-1, 1:-1] - T0)))
    # Boundary conditions
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 0
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0
    return u, v

def update_velocities_and_pressure(u, v, dt, dx, dy, p, rho, nu, b, T, delH, L, fl, beta, g, T0):
    b = build_up_b(b, rho, dt, u, v, dx, dy)
    p = pressure_poisson(p, dx, dy, b)
    u, v = velocity_update(u, v, dt, dx, dy, p, nu, b, T, delH, L, fl, beta, g, T0)
    return u, v, p

def temperature_solver(T, u, v, dt, dx, dy, delH, L, fl, beta, g, T0):
    Tn = T.copy()
    delHn = delH.copy()
    # Explicit update
    T[1:-1, 1:-1] = (Tn[1:-1, 1:-1] -
                     u[1:-1, 1:-1] * dt/dx * (Tn[1:-1, 1:-1] - Tn[1:-1, :-2]) -
                     v[1:-1, 1:-1] * dt/dy * (Tn[1:-1, 1:-1] - Tn[:-2, 1:-1]) +
                     (k/(c*rho)) * (dt/dx**2 * (Tn[1:-1, 2:] - 2*Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) +
                                    dt/dy**2 * (Tn[2:, 1:-1] - 2*Tn[1:-1, 1:-1] + Tn[:-2, 1:-1])) -
                     Ap/Ap0 * omega * c * (Tn[1:-1, 1:-1] - T0))
    # Enthalpy update
    delH[:, :] = delHn[:, :] + Ap/Ap0 * omega * c * (T[:, :] - T0)
    delH[:] = numpy.clip(delH, 0, L)
    fl = delH / L
    # Boundary conditions (match paper: left cold, right hot)
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]
    T[:, 0] = 0.0    # left wall (cold)
    T[:, -1] = 1.0   # right wall (hot)
    return T, fl

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, T, delH, L, fl, beta, g, T0):
    b = numpy.zeros((ny, nx))
    for n in range(nt):
        u, v, p = update_velocities_and_pressure(u, v, dt, dx, dy, p, rho, nu, b, T, delH, L, fl, beta, g, T0)
        T, fl = temperature_solver(T, u, v, dt, dx, dy, delH, L, fl, beta, g, T0)
        u[T <= T0] = 0
        v[T <= T0] = 0
        if n % 10000 == 0:
            print(f"Step {n}, fl max: {fl.max():.3f}, fl min: {fl.min():.3f}")
    return u, v, p, T, fl

def main():
    u = numpy.zeros((ny, nx))
    v = numpy.zeros((ny, nx))
    p = numpy.zeros((ny, nx))
    T = numpy.full((ny, nx), 0.0)  # Initial temperature: all cold
    T[:, -1] = 1.0                 # Right wall hot
    delH = numpy.zeros((ny, nx))
    fl = numpy.zeros((ny, nx))

    u, v, p, T, fl = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, T, delH, L, fl, beta, g, T0)

    # Plotting (match paper: liquid fraction with velocity vectors)
    pyplot.figure(figsize=(8, 6))
    cf = pyplot.contourf(X, Y, fl, levels=20, cmap=cm.jet, vmin=0, vmax=1)
    pyplot.colorbar(cf, label='Liquid Fraction')
    skip = (slice(None, None, 4), slice(None, None, 4))  # reduce arrow density
    pyplot.quiver(X[skip], Y[skip], u[skip], v[skip], color='k', scale=30)
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.title('Liquid Fraction and Velocity Vectors at t=100s')
    pyplot.tight_layout()
    pyplot.savefig('liquid_fraction_with_velocity.png', dpi=300)
    pyplot.show()

if __name__ == "__main__":
    main()