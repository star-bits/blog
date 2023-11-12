# Quantum of Everything

Index, somewhat chronologically:
- Ultraviolet catastrophe
  - Rayleigh-Jeans law and Planck's law
    - Wave basics
    - Wave equation
      - Taylor series
      - Linear PDEs
        - Heat equation
    - Wien's displacement law
    - Stephen-Boltzmann law
    - Energy distribution functions
      - Average molecular kinetic energy
        - Gaussian integrals
        - Exponential integrals and other frequent equations
- Photoelectric effect
- de Broglie wavelength
  - Quantum realm
- Bohr, Schrodinger, Heisenberg, Pauli
  - Gradient, Divergence, Curl, Laplacian
    - Divergence, Curl, and Maxwell's equations
    - Laplacian in spherical coordinates
    - Separation of Variables on TISE
      - Angular part
      - Radial part
        - No potential
        - Infinite spherical well
        - Hydrogen atom
          - Bohr formula and radius, the classical way
          - Quantum number n, l, m_l, s, m_s
            - L the angular momentum and space quantization
              - Uncertainty principle
                - Hilbert space and Hermitian operators
                  - State vector, Eigenvector, Eigenfunction
                  - Determinate state
            - Zeeman effect and space quantization
            - Selection rules
            - Spin
              - Magnetism
                - Moving charges
                  - Time Dilation and Length Contraction
                - Intrinsic magnetic moment
  - Measurement and Copenhagen interpretation
    - Schrodinger's cat
    - Einstein's Moon
    - EPR paradox
- Bell inequality









## Ultraviolet catastrophe
### Rayleigh-Jeans law and Planck's law
- Rayleigh-Jeans law: $B(\nu,T) = \frac{2\nu^2}{c^2} kT$
- Planck's law: $B(\nu,T) = \frac{2\nu^2}{c^2} \frac{h \nu}{e^{\frac{h\nu}{kT}} - 1}$
- B: spectral radiance of the black body radiation
  - Energy per unit time (Watts) per unit area (m^2) per unit solid angle (steradians sr) per unit frequency (Hz)

### Wave basics
### Wave equation
### Taylor series
### Linear PDEs
### Heat equation
### Wien's displacement law
### Stephen-Boltzmann law
### Energy distribution functions
- The distribution function f(E) is the probability that a particle is in energy state E.
  - The probability that a particle to occupy a given energy state E decreases exponentially with increasing energy.
  - The probability that a particle to occupy a given energy state E increases exponentially with increasing temperature.
- Maxwell-Boltzmann: $f(E) = \frac{1}{A e^{\frac{E}{kT}}}$ for indentical, distinguishable particles
- Bose-Einstein: $f(E) = \frac{1}{A e^{\frac{E}{kT}} - 1}$ for identical, indistinguishable particles with integer spin (bosons)
- Fermi-Dirac: $f(E) = \frac{1}{A e^{\frac{E}{kT}} + 1}$ for indentical, indistinguishable particles with half-integer spin (fermions)

### Average molecular kinetic energy
- Average molecular kinetic energy derived from Maxwell-Boltzmann distribution $f(E) = A e^{-\frac{E}{kT}}$
  - $f(v_z) = A e^{-\frac{m{v_z}^2}{2}\frac{1}{kT}}$
    - $\int_{-\infty}^{\infty} A e^{-\frac{m{v_z}^2}{2}\frac{1}{kT}} dv = 1$
    - As $\int_{-\infty}^{\infty} e^{-ax^2} dx = \sqrt{\frac{\pi}{a}}$,
    - $A \int_{-\infty}^{\infty} e^{-(\frac{m}{2kT})v^2} dv = A \sqrt{\frac{2\pi kT}{m}} = 1$
    - $A = \sqrt{\frac{m}{2\pi kT}}$
  - $f(v_z) = \sqrt{\frac{m}{2\pi kT}} e^{-\frac{m{v_z}^2}{2}\frac{1}{kT}}$

### Gaussian integrals
### Exponential integrals and other frequent equations

## Photoelectric effect
- frequency -> energy of ejected electrons
- amplitude^2 -> intensity -> number of photons -> number of ejected electrons

## de Broglie wavelength
### Quantum realm
- Wavelength of a particle < Distance between the particles --> Classical
- Wavelength of a particle > Distance between the particles --> Quantum
- $\lambda = \frac{h}{p} = \frac{h}{mv}$, $\frac{1}{2}mv^2 = \frac{3}{2}kT$ (average molecular kinetic energy), $m^2v^2 = 3mkT$, $\lambda = \frac{h}{\sqrt{3mkT}}$
  - Oxygen molecule: $\lambda = 2.58 \times 10^{-11} \text{m}$, where m = 5.31E-26 kg, T = 298 K
  - Electron: $\lambda = 6.24 \times 10^{-9} \text{m}$, where m = 9.11E-31 kg, T = 298 K
  - Note that $\lambda \propto \frac{1}{\sqrt{T}}$
- The wave nature of particles leads to superposition, which in turn leads to quantum characteristics such as discrete orbits and uncertainty principle.

## Bohr, Schrodinger, Heisenberg, Pauli
### Gradient, Divergence, Curl, Laplacian
### Divergence, Curl, and Maxwell's equations
### Laplacian in spherical coordinates
### Separation of Variables on TISE
### Angular part
### Radial part
### No potential
### Infinite spherical well
### Hydrogen atom
### Bohr formula and radius, the classical way
### Quantum number n, l m_l, s, m_s
### L the angular momentum and space quantization
### Uncertainty principle
### Hilbert space and Hermitian operators
### State vector, Eigenvector, Eigenfunction
### Determinate state
### Zeeman effect and space quantization
### Selection rules
### Spin
### Magnetism
### Moving charges
### Time Dilation and Length Contraction
### Intrinsic magnetic moment
### Measurement and Copenhagen interpretation
### Schrodinger's cat
### Einstein's Moon
### EPR paradox

## Bell inequality
