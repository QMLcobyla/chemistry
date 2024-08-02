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
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms.optimizers import SLSQP, SPSA, COBYLA
from qiskit_aer.noise import NoiseModel

from IPython.display import display


from other_functions import *

np.random.seed(999999)

class MoleculeManager():
  '''
  MoleculeManager\n

  '''
  def __init__(
      self,
      molecule: MoleculeInfo,
      optimizer=COBYLA(maxiter=500, tol=0.0001),

  ):
    self.molecule = molecule
    self.optimizer = optimizer


  def SetDualAtomDist(self, dist: float):
    '''
    Sets a distance between atoms of a molecule of two atoms
    '''
    self.molecule.coords=([0.0, 0.0, 0.0], [dist, 0.0, 0.0])


  def SetAtomCoords(self, coords):
    '''
    Sets a distance between atoms of a molecule of two atoms
    '''
    self.molecule.coords = coords


  def print_interatomic_distance(self):
    print(f"Interatomic Distance:")
    for atom1_idx in range(len(self.molecule.coords) - 1):
      for atom2_idx in range(atom1_idx + 1, len(self.molecule.coords)):
        vec = np.array(self.molecule.coords[atom1_idx]) - np.array(self.molecule.coords[atom2_idx])
        print('|', self.molecule.symbols[atom1_idx],', ', self.molecule.symbols[atom2_idx],'| = ', round(np.linalg.norm(vec), 5))



  def FindEnergyIdeal(self, print_energy=True, print_ansatz=True, print_circuit_info=True):
    '''
    returns vqe_result -- ground state energy value (float)
    '''

    exact_energies = []
    vqe_energies = []
    optimizer = SLSQP(maxiter=10)
    noiseless_estimator = Estimator(approximation=True)
    (qubit_op, num_particles, num_spatial_orbitals, problem, mapper) = get_qubit_op(self.molecule)

    result = exact_solver(qubit_op, problem)
    exact_energies.append(result.total_energies[0].real)
    init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    ansatz = UCCSD(
        num_spatial_orbitals, num_particles, mapper, initial_state=init_state
    )
    vqe = VQE(
        noiseless_estimator,
        ansatz,
        optimizer,
        initial_point=[0] * ansatz.num_parameters,
    )
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real
    vqe_energies.append(vqe_result)
    
    if (print_ansatz):
      display(ansatz.decompose().decompose().draw(fold=-1))
    if (print_circuit_info):
      print(f'ansatz.depth = {ansatz.depth()}')
      print(f'num of qubits = {ansatz.num_qubits}')
    if (print_energy):
      print(result)
      print(
          f"### TODO! what's the difference between VQE Result and Exact energy?\n",
          f"VQE Result: {vqe_result:.5f}\n",
          f"Exact Energy: {exact_energies[-1]:.5f}\n",
      )
      self.print_interatomic_distance()    
    return vqe_result


  def FindEnergyNoisy(self):
    exact_energies = []
    vqe_energies = []
    device = FakeKolkata()
    coupling_map = device.configuration().coupling_map
    noise_model = NoiseModel.from_backend(device)
    noisy_estimator = Estimator(
        backend_options={"coupling_map": coupling_map, "noise_model": noise_model}
    )
    (qubit_op, num_particles, num_spatial_orbitals, problem, mapper) = get_qubit_op(self.molecule)
    result = exact_solver(qubit_op, problem)
    exact_energies.append(result.total_energies)

    print("Exact Result:", result.total_energies)
    optimizer = SPSA(maxiter=100)
    var_form = EfficientSU2(qubit_op.num_qubits, entanglement="linear")
    vqe = VQE(noisy_estimator, var_form, optimizer)
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_result = problem.interpret(vqe_calc).total_energies
    print("VQE Result:", vqe_result)
