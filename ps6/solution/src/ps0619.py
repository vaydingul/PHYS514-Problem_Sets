import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import flip
from scipy.optimize import fsolve
from timeit import timeit


def mytests():
    """
    mytests

    This function executes the following processes:
        - It propagates the orbits of Jupiter and Sun for different timesteps
        - It visualizes the momentum and energy change
        - It compares the elapsed times of different methods

    Input:
        []

    Output:
        []

    Usage:
    mytests()
    """
    # Some constants
    G = 6.6743e-11 # Gravitational constant
    m_s = 1.988e30  # Mass of Sun
    m_j = 1.898e27  # Mass of Jupiter

    num_iter = 1 # Number of iteration to calculate average time elapsed on the algorithms

    timesteps = [-6, -8, -10, -12]  # Different timesteps
    methods = ["forw_euler", "symp_euler", "imp_euler", "rk4"] # Different methods

    proper_name = {"forw_euler": "Forward Euler", "symp_euler": "Symptactic Euler",
                   "imp_euler": "Implicit Euler", "rk4": "Runge Kutta 4"}
    
    markers = ["o", "^", "s", "h"] 

    time_vec = np.empty((len(timesteps), len(methods))) # Preallocation of time vector

    for (ix2, k) in enumerate(timesteps):
        dt = 2 ** k  # Current timestep
        orbit = 10  # How many order the planet will propagate?
        n = int((orbit / dt) + 1)  # Size calculation for the preallocation
        result = np.empty((len(methods), n, 8)) # Preallocation cumulative result tensor
        t = np.empty(n) # Preallocation of time vector

        for (ix, method) in enumerate(methods):
            # Propogation loop
            result[ix, :, :], t = orbit_propagate(two_body, dt, method)
            time_vec[ix2, ix] = timeit(lambda: orbit_propagate(
                two_body, dt, method), number=num_iter)

        ##### PLOTTING
        plt.figure()
        plt.xlabel("$x \; [m]$")
        plt.ylabel("$y \; [m]$")
        plt.title(
            "Orbit Propagation for Different Methods\n at $\Delta t = {0}$".format(dt))

        for (ix, method) in enumerate(methods):
            plt.scatter(result[ix, :, 0], result[ix, :, 1],
                        label="Jupyter {0}".format(proper_name[method]), marker = markers[ix])
            plt.scatter(result[ix, :, 4], result[ix, :, 5],
                        label="Sun {0}".format(proper_name[method]), marker = markers[ix])
        plt.legend(ncol=2)
        plt.tight_layout()
        #plt.savefig("report/figures/1{0}.png".format(ix2))

        plt.figure()
        plt.xlabel("Time $[sec]$")
        plt.ylabel("Angular Momentum $[kg m^2 s^{-1}]$")
        plt.title(
            "Conservation of Angular Momentum\nfor Different Methods\n at $\Delta t = {0}$".format(dt))

        for (ix, method) in enumerate(methods):
            plt.plot(t, [m_j * np.cross(result[ix, k, 0:2], result[ix, k, 2:4]) + m_s *
                         np.cross(result[ix, k, 4:6], result[ix, k, 6:8]) for k in range(t.shape[0])], label=proper_name[method], marker=markers[ix], markevery=int(t.shape[0] / 30))
        plt.legend(loc="best")
        plt.tight_layout()
        #plt.savefig("report/figures/2{0}.png".format(ix2))

        plt.figure()
        plt.xlabel("Time $[sec]$")
        plt.ylabel("Linear Momentum $[kg m s^{-1}]$")
        plt.title(
            "Conservation of Linear Momentum\nfor Different Methods\n at $\Delta t = {0}$".format(dt))

        for (ix, method) in enumerate(methods):
            plt.plot(t, [np.linalg.norm(m_j * result[ix, k, 2:4] + m_s * result[ix, k, 6:8])
                         for k in range(t.shape[0])], label=proper_name[method], marker=markers[ix], markevery=int(t.shape[0] / 30))
        plt.legend(loc="best")
        plt.tight_layout()
        #plt.savefig("report/figures/3{0}.png".format(ix2))

        plt.figure()
        plt.xlabel("Time $[sec]$")
        plt.ylabel("Total Energy $[kg m^2 s^{-2}]$")
        plt.title(
            "Conservation of Energy\nfor Different Methods\n at $\Delta t = {0}$".format(dt))

        for (ix, method) in enumerate(methods):
            plt.plot(t, [(0.5 * m_j * (np.linalg.norm(result[ix, k, 2:4]) ** 2)) + (0.5 * m_s * (np.linalg.norm(result[ix, k, 6:8])
                                                                                                 ** 2) - G * (m_j * m_s) / (np.linalg.norm(result[ix, k, 0:2] - result[ix, k, 4:6]))) for k in range(t.shape[0])], label=proper_name[method], marker=markers[ix], markevery=int(t.shape[0] / 30))
        plt.legend(loc="best")
        plt.tight_layout()
        #plt.savefig("report/figures/4{0}.png".format(ix2))

    plt.figure()
    labels = [proper_name[method] for method in methods]
    plt.title("Time Comparison of Different Methods at Different Timesteps")
    plt.ylabel("Average Elapsed Time [sec]")
    for (ix2, k) in enumerate(reversed(timesteps)):
        dt = 2 ** k  # Current timestep
        plt.bar(labels, time_vec[len(timesteps) - ix2 - 1, :] /
                num_iter, width=0.2, label="$\Delta t = ${0}".format(dt))
    plt.legend(loc="best")
    #plt.savefig("report/figures/5.png")
    plt.show()


def orbit_propagate(rhs, dt, method):
    """
        orbit_propagate

        It performs the following operations:
            - It propagates the orbit of Jupyter and Sun
            - It is just a wrapper for the ease of calculation
        
        Input:
            rhs = RHS function that will be used in calculation
            dt = Timestep
            method = Propagation method

        Output:
            r = Propagated displacement and velocity vector Jupiter and Sun
            t = Propagated time vector
    
        Usage:
            result[ix, :, :], t = orbit_propagate(two_body, dt, method)

        """
    # Constants
    G = 6.6743e-11  # Gravitational constant
    m_s = 1.988e30  # MAss of Sun
    m_j = 1.898e27  # Mass of Jupiter
    r_p = 7.405e11  # distance of jupyter to the sun at perihelion
    v_p = 13.72e3  # velocity of jupyter at perihelion
    t_0 = np.sqrt((r_p ** 3) / (G * (m_s + m_j)))  # Natural time constant
    mu = (m_s * m_j) / (m_s + m_j)  # Reduced mass

    # Initial conditions
    r_1x = (r_p * m_s) / (m_s + m_j)  # X coordinate of Jupiter
    r_1y = 0.0  # Y coordinate of Jupiter
    r_2x = -(r_p * m_j) / (m_s + m_j)  # X coordinate of Sun
    r_2y = 0.0  # Y coordinate of Sun
    v_1x = 0.0  # X velocity of Jupiter
    v_1y = v_p  # Y velocity of Jupiter
    v_2x = 0.0  # X velocity of Sun
    v_2y = 0.0  # Y velocity of Sun
    # Initial value vector in SI unit
    r_0 = np.array([r_1x, r_1y, v_1x, v_1y, r_2x, r_2y, v_2x, v_2y])
    # Velocities should be multiplied with natural time constant due to the differentiation
    r_0[[2, 3, 6, 7]] *= t_0

    orbit = 10  # How many orbit the planet will propagate?
    n = int((orbit / dt) + 1)  # Size calculation for the preallocation

    rho = np.empty((n, 8,))  # Preallocation of unitless displacement vector
    tau = np.empty(n)  # Preallocation of unitless time vector

    rho[0, :] = r_0 / r_p  # Nondimensiolization of the displacement vector
    tau[0] = 0.0  # Initial unitless time

    for k in range(n-1):
        rho[k+1, :], tau[k+1] = time_step(vec=rho[k, :], t=tau[k],
                                          rhs=rhs, dt=dt, method=method)
    # Recovery of displacement vector
    rho[:, [2, 3, 6, 7]] /= t_0
    r = rho * r_p
    # Recovery of time vector
    t = tau * t_0

    return r, t


def two_body(r, t):
    """
        two_body

        It performs the following operations:
            - It calculates the RHS of the two body problem
        
        Input:
            r = displacement and velocity vector of Jupyter and Sun
            t = time value

        Output:
            r_next = Calculated function value
    
        Usage:
            []

        """
    G = 6.6743e-11
    m_s = 1.988e30
    m_j = 1.898e27
    r_p = 7.405e11  # distance of jupyter to the sun at perihelion
    v_p = 13.72e3  # velocity of jupyter at perihelion
    t_0 = np.sqrt((r_p ** 3) / (G * (m_s + m_j)))
    mu = (m_s * m_j) / (m_s + m_j)

    r_next = np.empty(8)
    r_next[0:2] = r[2:4]
    r_next[2:4] = -(r[0:2] / np.linalg.norm(r[0:2])) / \
        ((np.linalg.norm(r[0:2] + r[4:6]) ** 2) * (m_j / mu))
    r_next[4:6] = r[6:8]
    r_next[6:8] = -(r[4:6] / np.linalg.norm(r[4:6])) / \
        ((np.linalg.norm(r[0:2] + r[4:6]) ** 2) * (m_s / mu))

    return r_next


def time_step(vec, t, rhs, dt, method="forw_euler"):
    """
        time_step

        It performs the following operations:
            - It performs one time-step propagation of the ODE, for a given input parameters
            - It does that via using methods
        
        Input:
            vec = Subject vector
            t = Current time value
            rhs = RHS function of the ODE
            dt = Timestep
            method = Which method will be used for propagation
        Output:
            r_next = Propagated subject vector
            t_next = One step ahead of time
    
        Usage:
            rho[k+1, :], tau[k+1] = time_step(vec=rho[k, :], t=tau[k],
                                          rhs=rhs, dt=dt, method=method)

        """

    # Low level methods are seperated from the high level methods
    if method == "forw_euler":
        return _forw_euler(vec, t, rhs, dt), t + dt
    elif method == "symp_euler":
        return _symp_euler(vec, t, rhs, dt), t + dt
    elif method == "imp_euler":
        return _imp_euler(vec, t, rhs, dt), t + dt
    elif method == "rk4":
        return _rk4(vec, t, rhs, dt), t + dt
    else:
        raise NameError(
            "{0} is not defined within context of the function.\nYou can choose the following methods:\n1) forw_euler\n2) symp_euler\n3) imp_euler\n4) rk4".format(method))


def _forw_euler(vec, t, rhs, dt):
    """
    y_{i+1} = y_i + dt * f
    """
    return vec + rhs(vec, t) * dt


def _symp_euler(vec, t, rhs, dt):
    """
    v_{i+1} = v_i + dt * f
    x_{i+1} = x_i + dt * v_{i+1}
    """

    r_next_1 = vec + rhs(vec, t) * dt
    vec[[2, 3, 6, 7]] = r_next_1[[2, 3, 6, 7]]
    r_next_2 = vec + rhs(vec, t) * dt

    return np.block([r_next_2[0:2], r_next_1[2:4], r_next_2[4:6], r_next_1[6:8]])


def _imp_euler(vec, t, rhs, dt):
    """
    y_{i+1} - dt * f(t_{i+1}, y_{i+1}) - y_i = 0
    Solve for y_{i+1}
    """
    def optimee(y):
        return y - dt * rhs(y, t + dt) - vec

    return fsolve(optimee, vec)


def _rk4(vec, t, rhs, dt):
    """
    y_{i+1} = y_i + dt * (1 / 6) * (f1 + 2*f2 + 2*f3 + f4)
    
    where:

    f1 = f(t,y_i)
    f2 = f(t_{i + 0.5}, f(y_i + 0.5 * dt * f1))
    f3 = f(t_{i + 0.5}, f(y_i + 0.5 * dt * f2))
    f4 = f(t_{i+1}, y_i + dt * f3)
    """
    f_1 = rhs(vec, t)
    f_2 = rhs(vec + 0.5 * dt * f_1, t + dt * 0.5)
    f_3 = rhs(vec + 0.5 * dt * f_2, t + dt * 0.5)
    f_4 = rhs(vec + dt * f_3, t + dt)

    return vec + (f_1 + 2 * f_2 + 2 * f_3 + f_4) * (dt / 6)


if __name__ == "__main__":
    mytests()
