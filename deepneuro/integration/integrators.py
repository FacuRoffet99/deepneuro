import numpy as np
from scipy.integrate import solve_ivp
from IPython.display import clear_output
from numba import jit

# Ode function
@jit(nopython=True)
def ode(t, vars, a=-0.02, w=1, G=0, C=0):
    n = len(a)
    x = vars[:n].flatten()
    y = vars[n:].flatten()

    x_term = np.dot(C,x) - C.sum(axis=1) * x
    y_term = np.dot(C,y) - C.sum(axis=1) * y

    dxdt = a*x - w*y - x*(x**2 + y**2) + G*x_term
    dydt = a*y + w*x - y*(x**2 + y**2) + G*y_term

    return np.concatenate((dxdt, dydt), axis=0)





def integrate_hopf_scipy(ode, params, A, W, C, G):
    # Get params
    n_samples = params['n_samples']
    nodes = params['nodes']
    TR = params['TR']
    t_use = params['t_use']
    init_min = params['init_min']
    init_max = params['init_max']
    t_min = params['t_min']
    t_max = params['t_max']

    n_samples, nodes = A.shape

    # Create arrays with the correct shapes
    X = np.ones((n_samples, nodes, int(t_use/TR)), dtype=float)

    # Create n random samples
    for n in range(n_samples):
        clear_output()
        print(n)

        # Sample params
        a = A[n]
        w = W[n]
        c = C[n]
        g = G[n]

        # Sample initial conditions
        x0 = np.random.uniform(init_min, init_max, size=nodes)
        y0 = np.random.uniform(init_min, init_max, size=nodes)
        vars = np.concatenate((x0, y0), axis=0)

        # Sample time
        t_scipy = np.arange(t_min, t_max, TR)

        # Solve ODE
        sol = solve_ivp(ode, (t_min, t_max), vars, t_eval=t_scipy, args=(a, w, g, c))

        X[n,:,:] = sol.y[:nodes,-int((t_use/TR)):]

    return X





@jit(nopython=True)
def numba_noise(size):
    noise = np.empty(size,dtype=np.float64)
    for i in range(size):
        noise[i] = np.random.normal()
    return noise

@jit(nopython=True)
def integrate_hopf_euler_maruyama(ode, params, A, W, C, G):
    # Get params
    n_samples = params['n_samples']
    nodes = params['nodes']
    TR = params['TR']
    t_use = params['t_use']
    init_min = params['init_min']
    init_max = params['init_max']
    t_min = params['t_min']
    t_max = params['t_max']
    dt = params['dt']
    sigma = params['sigma']

    n_samples, nodes = A.shape

    # Sample time
    t = np.arange(t_min, t_max, dt)

    # Initialize arrays to store the results
    x_solution = np.empty((n_samples, nodes, len(t)))

    for n in range(n_samples):
      # clear_output()
      if n % 100 == 0:
        print(n)

      # Initial conditions for x and y for each node
      x0 = np.random.uniform(init_min, init_max, size=nodes)
      y0 = np.random.uniform(init_min, init_max, size=nodes)
      vars = np.concatenate((x0, y0), axis=0)

      a = A[n]
      w = W[n]
      c = C[n]
      g = G[n]

      # noise = np.sqrt(dt)*sigma*numba_noise(size=2*nodes)

      # Euler-Maruyama integration
      for i in range(1, len(t)):
          # Time
          t_span = (t[i - 1], t[i])
          # Derivates
          d_vars = ode(t_span, vars, a, w, g, c)

          # Euler-Maruyama integration
          vars += d_vars*dt + np.sqrt(dt)*sigma*numba_noise(size=2*nodes)

          # Clamping
          # vars = np.where(vars>init_max, init_max, vars)
          # vars = np.where(vars<init_min, init_min, vars)
          vars[vars > init_max] = init_max
          vars[vars < init_min] = init_min

          # Filter solution
          x_solution[n, :, i] = vars[:nodes]

      if np.isnan(x_solution[n]).any():
          print(f'NaN found! n={n}')
          # raise Exception(f'NaN found! n={n}')
          raise

    return x_solution[:,:,-int((t_use/dt)):]