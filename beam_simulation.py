import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint
from IPython.display import HTML

# --- Beam Parameters ---
BEAM = {
    'L': 20.0,         # Length (m)
    'EI': 2.0e9,       # Stiffness (N*m^2)
    'rho_A': 2000.0,   # Mass per unit length (kg/m)
    'zeta': 0.02,      # Damping ratio
    'n_modes': 10,     # Number of vibration modes to simulate
    'dt': 0.01,        # Time step (s)
}

# --- Helper Functions ---
def mode_shapes(x, n, beam=BEAM):
    """Calculates the shape of the beam for mode n at position x."""
    x_arr = np.atleast_1d(x)
    return np.sin(np.outer(n, x_arr) * np.pi / beam['L']).squeeze()

def natural_frequencies(n, beam=BEAM):
    """Calculates the natural frequency (omega) for mode n."""
    return (n * np.pi / beam['L'])**2 * np.sqrt(beam['EI'] / beam['rho_A'])

def create_load_scenario(scenario, total_mass, beam_length=20.0):
    """Creates load configurations based on scenario."""
    g = 9.81
    total_weight = total_mass * g
    
    if scenario == 'one_point':
        return [{'type': 'point', 'offset': 0.0, 'force': total_weight}]
    
    elif scenario == 'two_point':
        # 2-axle truck-like distribution (equal split)
        return [
            {'type': 'point', 'offset': 0.0, 'force': 0.5 * total_weight},
            {'type': 'point', 'offset': 3.0, 'force': 0.5 * total_weight}
        ]
        
    elif scenario == 'three_point':
        w = total_weight / 3.0
        return [
            {'type': 'point', 'offset': 0.0, 'force': w},
            {'type': 'point', 'offset': 2.0, 'force': w},
            {'type': 'point', 'offset': 4.0, 'force': w}
        ]
        
    elif scenario == 'uniform':
        # Distributed over 5m
        length = 5.0
        return [{'type': 'distributed', 'offset': 0.0, 'length': length, 'force': total_weight}]
    
    else:
        # Default fallback (original truck)
        return [
            {'type': 'point', 'offset': 0.0, 'force': 0.65 * total_weight},
            {'type': 'point', 'offset': 3.0, 'force': 0.35 * total_weight}
        ]

def rhs_system(y, t, n, omega, modal_m, speed, beam, loads):
    """The equation of motion (Right Hand Side) for the ODE solver."""
    # Calculate position of the first axle/front
    pos_front = speed * t if speed > 0 else beam['L'] + speed * t
    
    modal_force = np.zeros_like(n, dtype=float)
    
    for load in loads:
        if load['type'] == 'point':
            offset = load['offset']
            force_mag = load['force']
            pos = pos_front - offset if speed > 0 else pos_front + offset
            
            if 0 <= pos <= beam['L']:
                # Project physical force onto modal coordinates
                modal_force += force_mag * mode_shapes(pos, n, beam)
        
        elif load['type'] == 'distributed':
            offset = load['offset']
            length = load['length']
            total_force = load['force']
            q = total_force / length
            
            # Determine the span of the load on the beam
            if speed > 0:
                head = pos_front - offset
                tail = head - length
            else:
                head = pos_front + offset
                tail = head + length
            
            x_start = min(head, tail)
            x_end = max(head, tail)
            
            # Intersect with beam [0, L]
            x1 = max(0, x_start)
            x2 = min(beam['L'], x_end)
            
            if x2 > x1:
                # Analytical integral of sin(k*x)
                k = n * np.pi / beam['L']
                # Force = q * integral(sin(kx)) = q * [-1/k cos(kx)]
                term = (q / k) * (np.cos(k * x1) - np.cos(k * x2))
                modal_force += term

    # State vector y contains [displacement_modes, velocity_modes]
    q_disp = y[:beam['n_modes']]
    dq_disp = y[beam['n_modes']:]
    
    # Equation of motion: m*ddq + c*dq + k*q = F
    ddq = (modal_force / modal_m) - (2 * beam['zeta'] * omega * dq_disp) - (omega**2 * q_disp)
    return np.concatenate([dq_disp, ddq])

def simulate_crossing(mass, speed, scenario='two_point', beam=BEAM):
    """Simulates one vehicle crossing."""
    n = np.arange(1, beam['n_modes'] + 1)
    omega = natural_frequencies(n, beam)
    modal_m = (beam['rho_A'] * beam['L']) / 2.0
    
    loads = create_load_scenario(scenario, mass, beam['L'])
    
    # Calculate max offset to determine duration
    max_offset = 0
    for load in loads:
        if load['type'] == 'point':
            max_offset = max(max_offset, load['offset'])
        elif load['type'] == 'distributed':
            max_offset = max(max_offset, load['offset'] + load['length'])

    # Simulation duration
    duration = (beam['L'] + max_offset) / abs(speed)
    t = np.arange(0, duration + 0.5, beam['dt'])
    
    # Initial conditions (at rest)
    y0 = np.zeros(2 * beam['n_modes'])
    
    # Solve ODE
    sol = odeint(rhs_system, y0, t, args=(n, omega, modal_m, speed, beam, loads))
    return t, sol

def get_sensor_data(sol, beam=BEAM, location_ratio=0.3):
    """Extracts displacement at a specific sensor location (e.g., 0.3 * L)."""
    n = np.arange(1, beam['n_modes'] + 1)
    phi_sensor = mode_shapes(beam['L'] * location_ratio, n, beam)
    q = sol[:, :beam['n_modes']]
    return q @ phi_sensor  # Superposition of modes

if __name__ == "__main__":
    # Simulate different scenarios
    scenarios = ['one_point', 'two_point', 'three_point', 'uniform']
    demo_mass = 3000
    demo_speed = 15.0

    results = {}

    plt.figure(figsize=(10, 5))

    for sc in scenarios:
        t_sc, sol_sc = simulate_crossing(demo_mass, demo_speed, scenario=sc)
        sig_sc = get_sensor_data(sol_sc)
        results[sc] = (t_sc, sol_sc, sig_sc)
        plt.plot(t_sc, sig_sc, label=f'{sc}')

    plt.title(f'Sensor Signal Comparison (Mass={demo_mass}kg, Speed={demo_speed}m/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_file = 'simulation_comparison.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
