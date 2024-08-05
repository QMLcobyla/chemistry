from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer.primitives import Sampler, Estimator
from qiskit_algorithms.optimizers import SLSQP, SPSA
import pylatexenc
from qiskit_ibm_runtime.fake_provider import FakeKolkata
from qiskit.circuit.library import EfficientSU2
import numpy as np
import qiskit_nature
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper, InterleavedQubitMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms.optimizers import SLSQP, SPSA, COBYLA
from qiskit_aer.noise import NoiseModel

from datetime import datetime


def get_var_form(params):
  qr = QuantumRegister(1, name="q")
  cr = ClassicalRegister(1, name="c")
  qc = QuantumCircuit(qr, cr)
  qc.u(params[0], params[1], params[2], qr[0])
  qc.measure(qr, cr[0])
  return qc


def get_qubit_op(molecule):
  '''
  Uses ParityMapper
  '''
  driver = PySCFDriver.from_molecule(molecule)
  properties = driver.run()
  problem = FreezeCoreTransformer(
      freeze_core=True, remove_orbitals=[-3, -2]
  ).transform(properties)

  num_particles = problem.num_particles
  num_spatial_orbitals = problem.num_spatial_orbitals

  mapper = ParityMapper(num_particles=num_particles)
  # mapper = JordanWignerMapper()
  qubit_op = mapper.map(problem.second_q_ops()[0])
  return qubit_op, num_particles, num_spatial_orbitals, problem, mapper


def exact_solver(qubit_op, problem):
  sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
  result = problem.interpret(sol)
  return result


def draw_orbitals(problem, molecule_name=None):
    # Draw orbitals (method provided by Max)
    print(f'Number of particles : {problem.num_particles}')
    print(f'Number of spatial orbitals : {problem.num_spatial_orbitals}')
    print(f'Orbital energies : {problem.orbital_energies}')

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    for i in range(problem.num_spatial_orbitals):
        if problem.orbital_occupations[i] == 1:
            co = 'tab:blue'
        else:
            co = 'tab:green'

        ax.scatter(i,np.log10(np.abs(problem.orbital_energies[i])), s=15, c=co, marker='o')

    ax.set_xlabel('Orbital', fontsize=15)
    ax.set_ylabel(r'$\log\left(|E|\right)$', fontsize=15)
    ax.set_title(f'{molecule_name} Orbital energy')
    
    ax.scatter(-2,2,s=15, c='tab:blue', marker='o', label='Occupied orbitals')
    ax.scatter(-2,2,s=15, c='tab:green', marker='o', label='Empty orbitals')
    ax.set_xlim(-0.2,problem.num_spatial_orbitals+0.2)
    ax.set_ylim(np.amin(np.log10(np.abs(problem.orbital_energies)))-0.2,
                np.amax(np.log10(np.abs(problem.orbital_energies)))+0.2)
    ax.legend(loc='best', fontsize=15)

def get_freezed_problem(properties, indexes = None):
    # Shortcut for the problem definition
    return FreezeCoreTransformer(
        freeze_core=True, remove_orbitals=indexes
    ).transform(properties)

def plot_error(
    vals_1,
    vals_2,
    domain,
    label_x='Distance',
    label_y='Error',
    title='Error',
    lower_bound=-np.Inf,
    upper_bound=np.Inf,
    fig_size=(10, 6),
):
    li = np.searchsorted(np.array(domain), lower_bound, side='right')
    ri = np.searchsorted(np.array(domain), upper_bound, side='left')
    vals_1 = np.array(vals_1[li:ri])
    vals_2 = np.array(vals_2[li:ri])
    err = np.abs(vals_1 - vals_2)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(domain[li:ri],err,'+--')

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    plt.grid()
    plt.title(title)
    plt.show()
    
    
import random
def generate_random_color_hex():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def plot_comparisons(
    data,
    labels,
    domain=None,
    colors = [],
    linestyles = [],
    lower_bound=-np.Inf,
    upper_bound=np.Inf,
    fig_size=(10, 6),
    label_x='Distance',
    label_y='Energy',
    title=None,
):
    '''
    data: [if domain=None] array of pairs [val_array, domain_array] 
    data: [if domain != None] array of data arrays to be compared
    domain: None if data with different domain 
    '''
    plt.figure(figsize=fig_size)
    for (i, el) in enumerate(data):
        if (domain is None):
            vals, dom = el
        else:
            vals = el
            dom = domain
        li = np.searchsorted(np.array(dom), lower_bound, side='right')
        ri = np.searchsorted(np.array(dom), upper_bound, side='left')
        if (len(colors) > i):
            col = colors[i]
        else:
            col = generate_random_color_hex()
        if (len(linestyles) > i):
            style = linestyles[i]
        else:
            style = 'solid'
        
        plt.plot(dom[li:ri], vals[li:ri], label=labels[i], linestyle=style, color=col)
            
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    