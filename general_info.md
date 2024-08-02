# H2O
molecule_hoh = [
    'H .0 '+str(dist)+' .0;',
    'O .0 '+' .0'+' .0;',
    'H .0 -'+str(dist)+' .0;',
    ]

# LiH
molecule_lih = [
    'Li .0 .0 .0;',
    'H '+str(dist)+' .0 .0;',
    ]

# LiOH
molecule_lioh = [
    'O .0 .0 .0;',
    'H .0 '+' 0.9691'+' .0;',
    'Li .0 -'+str(dist)+' .0;',
  ]

# H2
molecule_hh = [
    'H .0 '+str(dist)+' .0;',
    'H .0 '+' .0'+' .0;',
  ]
