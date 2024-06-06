import numpy as np
from scipy.integrate import solve_ivp
from IPython.display import clear_output
from numba import jit

@jit(nopython=True)
def ode_hopf(t, vars, a=-0.02, w=1, G=0, C=0):
    '''
    Defines the ordinary differential equations of the Hopf model.

    Args:
        t (int): Dummy parameter.
        vars (np.array): x and y variables of the model.
        a (np.array): Amplitude parameters of each node.
        w (np.array): Frequency parameters of each node.
        G (float): Scaling constant.
        C (np.array): Matrix of structural connectivity.

    Returns:
        dvars (np.array): Derivates of the x and y variables of the model.    
    '''
    n = len(a)
    x = vars[:n].flatten()
    y = vars[n:].flatten()

    x_term = np.dot(C,x) - C.sum(axis=1) * x
    y_term = np.dot(C,y) - C.sum(axis=1) * y

    dxdt = a*x - w*y - x*(x**2 + y**2) + G*x_term
    dydt = a*y + w*x - y*(x**2 + y**2) + G*y_term

    dvars = np.concatenate((dxdt, dydt), axis=0)

    return dvars

def initialize_hopf(n_samples, nodes, seed, a_range, w_range, g, SC):
    '''
    Initializes Hopf parameters by: 
        * Stacking SC and G.
        * Generating random values for a and w using a fixed seed.
    '''
    np.random.seed(seed=seed)
    A = np.random.uniform(a_range[0], a_range[1], size=(n_samples, nodes))
    W = np.random.uniform(w_range[0], w_range[1], size=(n_samples, nodes))
    G = np.repeat(g, n_samples)
    C = np.tile(SC[None,:], (n_samples,1,1))
    return A, W, G, C


@jit(nopython=True)
def numba_noise(size):
    '''
    The parameter 'size' in np.random.normal() is not supported by numba, this function fixes that.
    '''
    noise = np.empty(size,dtype=np.float64)
    for i in range(size):
        noise[i] = np.random.normal()
    return noise


@jit(nopython=True)
def integrate_hopf_euler_maruyama(A, W, C, G, TR, t_use, t_max, init_min=-1, init_max=1, t_min=0, dt=0.5, sigma=0.01):
    '''
    Integrates the Hopf model using the Euler-Maruyama method for each provided subject.

    Args:
        A, W, C, G: Hopf parameters for each subject.
        TR (float): Repetition time of the dataset.
        t_use (int): Number of timesteps to return.
        t_max (float): Last timestep.
        init_min (float): Minimum value for variable initialization.
        init_max (float): Maximum value for variable initialization.
        t_min (float): Initial timestep.
        dt (float): Distance between timesteps.
        sigma (float): Controls the amount of noise.

    Returns:
        x_solution (np.array): Resulting time series for each subject.
    '''
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

      # Euler-Maruyama integration
      for i in range(1, len(t)):
          # Time
          t_span = (t[i - 1], t[i])
          # Derivates
          d_vars = ode_hopf(t_span, vars, a, w, g, c)

          # Euler-Maruyama integration
          vars += d_vars*dt + np.sqrt(dt)*sigma*numba_noise(size=2*nodes)

          # Clamping
          vars[vars > init_max] = init_max
          vars[vars < init_min] = init_min

          # Only save values for x
          x_solution[n, :, i] = vars[:nodes]

      if np.isnan(x_solution[n]).any():
          print(f'NaN found! n={n}')
          raise
    
    # Delete initial unwanted timesteps
    x_solution = x_solution[:,:,-int((t_use/dt)):]
    # Fix sampling rate
    x_solution = x_solution[:,:,::int(TR/dt)]  

    return x_solution