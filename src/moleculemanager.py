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
from qiskit_nature.second_q.problems import ElectronicStructureResult
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms.optimizers import SLSQP, SPSA, COBYLA
from qiskit_aer.noise import NoiseModel

from IPython.display import display
from datetime import datetime


from src.helpers import *

np.random.seed(999999)

class MoleculeManager():
  '''
  MoleculeManager\n
  MoleculeInfo(
    symbols=["Li", "O", "H"],
    coords=([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    multiplicity=1,  # = 2*spin + 1
    charge=0,
  )
  '''
  def __init__(
      self,
      molecule: MoleculeInfo,
      name = None,

  ):
    self.molecule = molecule
    if (name is None):
      self.name = "".join(self.molecule.symbols)
    else:
      self.name = name


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



  def FindGroundStateExactSolver(
    self,
    orbitals_to_remove=[],
    mapper_type=ParityMapper,
    output_info=True,
    timestamps=True,
    # noisy_mode=False,
    # noisy_device = FakeKolkata(),
  ) -> ElectronicStructureResult:
    '''
    Orbitals_to_remove: indeces of orbitals that will be frozen in addition to removed by default by FreezeCoreTransformer.\n
    '''
    if (timestamps):
      print(f'{(datetime.now().time()).strftime("%H:%M:%S")} FindGroundStateExactSolver started.')

    properties_molecule = self.get_problem()
    problem = get_freezed_problem(properties_molecule, orbitals_to_remove)    
    num_particles = problem.num_particles
    mapper = mapper_type(num_particles=num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])
    exact_result = exact_solver(qubit_op, problem)
    if (output_info):
      print('Exact result:')
      print(exact_result)
    if (timestamps):
      print(f'{(datetime.now().time()).strftime("%H:%M:%S")} FindGroundStateExactSolver done.')
    return exact_result
    
  
  def FindGroundStateVQE(
    self,
    orbitals_to_remove=[],
    mapper_type=ParityMapper,
    ansatz_type=UCCSD,
    optimizer_type=COBYLA(maxiter=15, tol=0.0001),
    output_info=True,
    timestamps=True,
    noisy_mode=False,
    noisy_device = FakeKolkata(),    
  )-> ElectronicStructureResult:
    '''
    Orbitals_to_remove: indeces of orbitals that will be frozen in addition to removed by default by FreezeCoreTransformer.\n 
    Note: params of noisy device are only considered if 'noisy_mode=True'.\n
    '''
    if (timestamps):
      print(f'{(datetime.now().time()).strftime("%H:%M:%S")} FindGroundStateVQE started.')

    properties_molecule = self.get_problem()
    problem = get_freezed_problem(properties_molecule, orbitals_to_remove)
    if (noisy_mode):
      coupling_map = noisy_device.configuration().coupling_map
      noise_model = NoiseModel.from_backend(noisy_device)
      estimator = Estimator(
        backend_options={"coupling_map": coupling_map, "noise_model": noise_model}
      )
    else:
      estimator = Estimator(approximation=True)
    
    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals
    mapper = mapper_type(num_particles=num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])
    init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    # TODO: maybe try EfficientSU2 instead of UCCSD
    ansatz = ansatz_type(
        num_spatial_orbitals, num_particles, mapper, initial_state=init_state
    )
    if (timestamps):
      print(f'{(datetime.now().time()).strftime("%H:%M:%S")} ansatz initialized')
    if (output_info):
      print(f'ansatz.depth = {ansatz.depth()}')
      print(f'num of qubits = {ansatz.num_qubits}')
    
    vqe = VQE(
          estimator,
          ansatz,
          optimizer_type,
          initial_point=[0] * ansatz.num_parameters,
      )
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    if (timestamps):
      print(f'{(datetime.now().time()).strftime("%H:%M:%S")} VQE compute_minimum_eigenvalue done')

    vqe_result = problem.interpret(vqe_calc)
    if (output_info):
      print('VQE result:')
      print(vqe_result)
    if (timestamps):
      print(f'{(datetime.now().time()).strftime("%H:%M:%S")} FindGroundStateVQE done.')
    return vqe_result
      
  
  def Experiment(
    self,
    orbitals_to_remove=[],
    mapper_type=ParityMapper,
    ansatz_type=UCCSD,
    optimizer_type=COBYLA(maxiter=15, tol=0.0001),
    output_info=True,
    timestamps=True,
    noisy_mode=False,
    noisy_device = FakeKolkata(),     
  ) -> tuple[ElectronicStructureResult, ElectronicStructureResult]:
    '''
    returns tuple (ExactResult, VQEResult)\n
    Orbitals_to_remove: indeces of orbitals that will be frozen in addition to removed by default by FreezeCoreTransformer.\n 
    Note: params of noisy device are only considered if 'noisy_mode=True'.\n
    '''
    return (
      self.FindGroundStateExactSolver(
        orbitals_to_remove=orbitals_to_remove,
        mapper_type=mapper_type,
        output_info=output_info,
        timestamps=timestamps,
      ),
      self.FindGroundStateVQE(
        orbitals_to_remove=orbitals_to_remove,
        mapper_type=mapper_type,
        ansatz_type=ansatz_type,
        optimizer_type=optimizer_type,
        output_info=output_info,
        timestamps=timestamps,
        noisy_mode=noisy_mode,
        noisy_device=noisy_device,
      ),
    )
  
  def get_problem(self):
    driver = PySCFDriver.from_molecule(self.molecule)
    properties_molecule = driver.run()
    return properties_molecule