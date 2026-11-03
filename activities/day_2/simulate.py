"""Lennard-Jones argon fluid: NVE molecular dynamics with ASE.

This is the "workload" for the context-management activity. It prints one
line per timestep, so a single run produces thousands of lines of output --
far more than you want to paste into a model's context window.

If the timestep is too large, the velocity Verlet integrator becomes unstable:
total energy (which should be conserved) drifts upward, the temperature
climbs, and eventually atoms overlap and the energies become NaN. Like many
real MD codes, the simulation keeps running after that point and still exits
normally, so the only evidence of failure is in the log itself.

Example:
    uv run python simulate.py --dt 5 --steps 3000
"""
import argparse
import time

import numpy as np
from ase import units
from ase.calculators.lj import LennardJones
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

# Lennard-Jones parameters for argon.
SIGMA = 3.405     # Angstrom
EPSILON = 0.0103  # eV


def build_system(cells: int, density: float, temperature: float, seed: int):
    '''Builds an FCC argon lattice at the given reduced density (rho * sigma^3)
    and assigns Maxwell-Boltzmann velocities. The lattice melts into a fluid
    within the first few hundred steps.'''
    # An FCC unit cell holds 4 atoms, so rho = 4 / a^3.
    lattice_constant = SIGMA * (4 / density) ** (1 / 3)
    atoms = FaceCenteredCubic(
        symbol='Ar',
        latticeconstant=lattice_constant,
        size=(cells, cells, cells),
        pbc=True,
    )
    atoms.calc = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=2.5 * SIGMA, smooth=True)
    thermalize_momenta(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    return atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dt', type=float, default=5.0, help='Timestep in femtoseconds (default: 5)')
    parser.add_argument('--steps', type=int, default=3000, help='Number of MD steps (default: 3000)')
    parser.add_argument('--cells', type=int, default=3, help='FCC unit cells per side; atoms = 4 * cells^3 (default: 3)')
    parser.add_argument('--density', type=float, default=0.8, help='Reduced density rho*sigma^3 (default: 0.8)')
    parser.add_argument('--temperature', type=float, default=120.0, help='Initial temperature in K (default: 120)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initial velocities (default: 0)')
    parser.add_argument('--log-every', type=int, default=1, help='Print thermodynamics every N steps (default: 1)')
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error('--log-every must be >= 1')

    atoms = build_system(args.cells, args.density, args.temperature, args.seed)
    dyn = VelocityVerlet(atoms, timestep=args.dt * units.fs)
    e0 = atoms.get_total_energy()
    n = len(atoms)

    print('# Lennard-Jones argon, NVE velocity Verlet (ASE)')
    print(f'# atoms={n} cells={args.cells} density={args.density} box={atoms.cell.lengths()[0]:.3f}A')
    print(f'# dt={args.dt}fs steps={args.steps} T0={args.temperature}K seed={args.seed} log_every={args.log_every}')
    print(f'# initial total energy: {e0 / n:.6f} eV/atom')
    print(f'# {"step":>6} {"time_ps":>9} {"epot_eV/atom":>14} {"ekin_eV/atom":>14} '
          f'{"etot_eV/atom":>14} {"temp_K":>10} {"rel_drift":>11}')

    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        dyn.run(1)
        if step % args.log_every != 0:
            continue
        epot = atoms.get_potential_energy() / n
        ekin = atoms.get_kinetic_energy() / n
        etot = epot + ekin
        drift = (etot - e0 / n) / abs(e0 / n)
        print(f'  {step:>6} {step * args.dt / 1000:>9.3f} {epot:>14.6f} {ekin:>14.6f} '
              f'{etot:>14.6f} {atoms.get_temperature():>10.2f} {drift:>11.3e}')

    elapsed = time.perf_counter() - start
    print(f'# finished {args.steps} steps in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
