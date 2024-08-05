# chemistry

What we do:
- for each molecule we look at orbitals and fix some of them for computational efficiency (where needed)
- run experiments with geometrical configuration of molecule (distances between atoms and angles between atom connections)
- use VQE to find properties of molecule ground state
- by default COBYLA optimizer and ParityMapper (configurable)
- compare VQE result with NumPyMinimumEigensolver
- verify optimal geometrical configuration w.r.t. https://cccbdb.nist.gov

NOTE! PySCFDriver installation seems to fail on Windows. Works fine on MacOS or Linux machines. 


Please check our slides in 'Quantum - slides.pdf' or [here](https://docs.google.com/presentation/d/1smrlVZoIDfwcGyHLm5yRWoHOyTCi82Xtp7_mtGR91B0/).


## TODO:
- maybe try EfficientSU2 instead of UCCSD ansatz
    

