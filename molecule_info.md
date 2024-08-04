# H2O
molecule_hoh_dist = [
  'H .0 '+str(dist)+' .0;',
  'O .0 '+' .0'+' .0;',
  'H .0 -'+str(dist)+' .0;',
]
orbitals_to_freeze = []
domain = np.linspace()

molecule_hoh_angle = [
  'H .0 '+str(dist)+' .0;',
  'O .0 '+' .0'+' .0;',
  'H .0 -'+str(dist)+' .0;',
]
orbitals_to_freeze = []
domain = np.linspace()

# LiH
molecule_lih = [
  'Li .0 .0 .0;',
  'H '+str(dist)+' .0 .0;',
]
orbitals_to_freeze = []
domain = np.linspace()

# LiOH
molecule_lioh = [
  'O .0 .0 .0;',
  'H .0 '+' 0.9691'+' .0;',
  'Li .0 -'+str(dist)+' .0;',
]
orbitals_to_freeze = []
domain = np.linspace()

# H2
molecule_hh = [
  'H .0 '+str(dist)+' .0;',
  'H .0 '+' .0'+' .0;',
]
orbitals_to_freeze = []
domain = np.linspace()
