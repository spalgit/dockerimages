#!/bin/sh

# Prepare ligand topology as usual
ligand="REX-010714_NEW.pdb"
#acpype -i $ligand
basename="${ligand%.pdb}"
newname="${basename}.acpype"
echo "$basename"
echo "$newname"
cp $newname/${basename}_GMX.itp .
cp $newname/${basename}_GMX.top .
cp $newname/${basename}_NEW.pdb .
cp $newname/posre_${basename}.itp .
mv ${basename}_GMX.itp Ligand_GMX.itp

mv ${basename}_GMX.itp Ligand_GMX.itp

tail -n +9 ${basename}_GMX.top>Ligand_GMX_2.top

include_amber='#include "amber99sb.ff/forcefield.itp"'
include_spce='#include "amber99sb.ff/spce.itp"'
include_ions='#include "amber99sb.ff/ions.itp"'
include_lig='#include "Ligand_GMX.itp"'

{ echo "$include_ions"; cat Ligand_GMX_2.top; } > temp1
{ echo "$include_spce"; cat temp1; } > temp2
{ echo "$include_lig"; cat temp2; } > temp3
{ echo "$include_amber"; cat temp3; } > temp4 && mv temp4 Complex.top

mapdir=$(pwd)

DOCKER="docker run --user $(id -u):$(id -g) --rm --gpus all -it -v ${mapdir}:${mapdir} -w ${mapdir} gcr.io/cheminfosolutions-prod-02/gromacs-plumed:2024-3_AND_2-9-3 bash -c \"source /usr/local/gromacs/bin/GMXRC &&"
DOCKER_mdrun="docker run --user $(id -u):$(id -g) --rm --gpus all -dit -v ${mapdir}:${mapdir} -w ${mapdir} gcr.io/cheminfosolutions-prod-02/gromacs-plumed:2024-3_AND_2-9-3 bash -c \"source /usr/local/gromacs/bin/GMXRC &&"


# Box setup and solvation within Dockerized GROMACS

eval "${DOCKER} gmx_mpi editconf -bt triclinic -f ${basename}_NEW.pdb -o Complex.pdb -d 1.0\""
eval "${DOCKER} gmx_mpi solvate -cp Complex.pdb -cs spc216.gro -o Complex_b4ion.pdb -p Complex.top\""

# Create ions.mdp file
cat << EOF >| ions.mdp
; ions.mdp - used as input into grompp to generate ions.tpr
; Parameters describing what to do, when to stop and what to save
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01          ; Minimization step size
nsteps      = 200           ; Maximum number of (minimization) steps to perform (should be 50000)

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 1         ; Frequency to update the neighbours list and long range forces
cutoff-scheme   = Verlet    ; Buffered neighbours searching
ns_type         = grid      ; Method to determine neighbours list (simple, grid)
coulombtype     = PME       ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions
EOF

# Create em.mdp file
cat << EOF >| em.mdp
; em.mdp - used as input into grompp to generate em.tpr
; Parameters describing what to do, when to stop and what to save
integrator  = steep         ; Algorithm (steep = steepest descent minimization)
emtol       = 1000.0        ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
emstep      = 0.01          ; Minimization step size
nsteps      = 200           ; Maximum number of (minimization) steps to perform (should be 50000)

; Parameters describing how to find the neighbors of each atom and how to calculate the interactions
nstlist         = 1         ; Frequency to update the neighbours list and long range forces
cutoff-scheme   = Verlet    ; Buffered neighbours searching
ns_type         = grid      ; Method to determine neighbours list (simple, grid)
coulombtype     = PME       ; Treatment of long range electrostatic interactions
rcoulomb        = 1.0       ; Short-range electrostatic cut-off
rvdw            = 1.0       ; Short-range Van der Waals cut-off
pbc             = xyz       ; Periodic Boundary Conditions in all 3 dimensions
EOF


# Create md.mdp file
cat << EOF >| md.mdp
;define                  = -DPOSRES  ; position restrain the protein
; Run parameters
integrator              = md        ; leap-frog integrator
nsteps                  = 1000      ; 2 * 1000 = 2 ps (should be 50000: 100 ps)
dt                      = 0.002     ; 2 fs
; Output control
nstxout-compressed      = 2        ; save compressed coordinates every 20 fs
nstxout                 = 0        ; save coordinates
nstvout                 = 0        ; save velocities
nstenergy               = 10       ; save energies every 20 fs
nstlog                  = 10       ; update log file every 1.0 ps
; Bond parameters
continuation            = no        ; first dynamics run
constraint_algorithm    = lincs     ; holonomic constraints
constraints             = h-bonds   ; bonds involving H are constrained
lincs_iter              = 1         ; accuracy of LINCS
lincs_order             = 4         ; also related to accuracy
; Nonbonded settings
cutoff-scheme           = Verlet    ; Buffered neighbour searching
ns_type                 = grid      ; search neighbouring grid cells
nstlist                 = 10        ; 20 fs, largely irrelevant with Verlet
rcoulomb                = 1.0       ; short-range electrostatic cutoff (in nm)
rvdw                    = 1.0       ; short-range van der Waals cutoff (in nm)
DispCorr                = EnerPres  ; account for cut-off vdW scheme
; Electrostatics
coulombtype             = PME       ; Particle Mesh Ewald for long-range electrostatics
pme_order               = 4         ; cubic interpolation
fourierspacing          = 0.16      ; grid spacing for FFT
; Temperature coupling is on
tcoupl                  = V-rescale             ; modified Berendsen thermostat
tc-grps                 = Protein Non-Protein   ; two coupling groups - more accurate
tau_t                   = 0.1     0.1           ; time constant, in ps
ref_t                   = 300     300           ; reference temperature, one for each group, in K
; Pressure coupling is off
pcoupl                  = no        ; no pressure coupling in NVT
; Periodic boundary conditions
pbc                     = xyz       ; 3-D PBC
; Velocity generation
gen_vel                 = yes       ; assign velocities from Maxwell distribution
gen_temp                = 300       ; temperature for Maxwell distribution
gen_seed                = -1        ; generate a random seed
EOF

cat << EOF >| min2.mdp
integrator               = steep
nsteps                   = 50000

nstenergy                = 500
nstlog                   = 500
nstxout-compressed       = 1000

cutoff-scheme            = Verlet

coulombtype              = PME
rcoulomb                 = 1.0

vdwtype                  = Cut-off
rvdw                     = 1.0
DispCorr                 = EnerPres
EOF

cat << EOF >|eql.mdp
integrator               = md
dt                       = 0.002     ; 2 fs
nsteps                   = 50000     ; 100 ps

nstenergy                = 200
nstlog                   = 2000
nstxout-compressed       = 10000

gen-vel                  = yes
gen-temp                 = 298.15

constraint-algorithm     = lincs
constraints              = h-bonds

cutoff-scheme            = Verlet

coulombtype              = PME
rcoulomb                 = 1.0

vdwtype                  = Cut-off
rvdw                     = 1.0
DispCorr                 = EnerPres

tcoupl                   = Nose-Hoover
tc-grps                  = System
tau-t                    = 2.0
ref-t                    = 298.15
nhchainlength            = 1
EOF

cat << EOF >|eql2.mdp
integrator               = md
dt                       = 0.002     ; 2 fs
nsteps                   = 500000    ; 1.0 ns

nstenergy                = 200
nstlog                   = 2000
nstxout-compressed       = 10000

continuation             = yes
constraint-algorithm     = lincs
constraints              = h-bonds

cutoff-scheme            = Verlet

coulombtype              = PME
rcoulomb                 = 1.0

vdwtype                  = Cut-off
rvdw                     = 1.0
DispCorr                 = EnerPres

tcoupl                   = Nose-Hoover
tc-grps                  = System
tau-t                    = 2.0
ref-t                    = 298.15
nhchainlength            = 1

pcoupl                   = Parrinello-Rahman
tau_p                    = 2.0
compressibility          = 4.46e-5
ref_p                    = 1.
EOF

cat << EOF >|prd.mdp
;       Input file
;
; *** Disclaimer: Gromacs 2020 officially ***
; *** does not support vacuum simulations ***
;
define              =
; integrator
integrator          =  md
nsteps              =  5000000
dt                  =  0.002
cutoff-scheme       =  verlet
;
; removing CM translation and rotation
comm_mode           =  Linear
nstcomm             =  1000
;
; output control
nstlog                   = 5000
nstenergy                = 5000
nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstxout-compressed       = 500
;
; neighbour searching
;nstlist             = 0
;ns-type             = simple
pbc                 = xyz
;rlist               = 1.0
periodic_molecules  = no
;
; electrostatic
rcoulomb            = 1.0
coulombtype         = Cut-off
;
; vdw
vdw-type            = Cut-off
rvdw                = 1.0
;
; constraints
constraints              = h-bonds
constraint-algorithm     = lincs
lincs_iter               = 4
;
; temperature
Tcoupl              = v-rescale
tc_grps             = system
tau_t               = 0.2
ref_t               = 300.000
;
; pression
Pcoupl              =  no
;
; initial velocities
gen_vel             = yes
gen_temp            = 300.000
gen_seed            = -1
EOF



# For each grompp call (TPR generation), use Docker with suitable working dir

eval "${DOCKER} gmx_mpi grompp -f ions.mdp -c Complex_b4ion.pdb -p Complex.top -o Complex_b4ion.tpr -maxwarn 1\""
cp Complex.top Complex_ion.top
eval "${DOCKER} echo 4 | gmx_mpi genion -s Complex_b4ion.tpr -o Complex_b4em.pdb -neutral -conc 0.15 -p Complex_ion.top\""
mv Complex_ion.top Complex.top


# EM
eval "${DOCKER} gmx_mpi grompp -f em.mdp -c Complex_b4em.pdb -p Complex.top -o em.tpr\""
docker run --user $(id -u):$(id -g) --rm --gpus all -it -v "${mapdir}:${mapdir}" -w "${mapdir}" \
  -e OMP_NUM_THREADS=1 gcr.io/cheminfosolutions-prod-02/gromacs-plumed:2024-3_AND_2-9-3 \
  bash -c "mpirun -n 1 gmx_mpi mdrun -v -deffnm em"

# MIN2
eval "${DOCKER}  gmx_mpi grompp -f min2.mdp -o min2.tpr -p Complex.top -po min2.mdp -c em.gro -maxwarn 1\""

docker run --user $(id -u):$(id -g) --rm --gpus all -it \
  -v ${mapdir}:${mapdir} -w ${mapdir} \
  -e OMP_NUM_THREADS=1 \
  gcr.io/cheminfosolutions-prod-02/gromacs-plumed:2024-3_AND_2-9-3 \
  bash -c \
  "mpirun -n 1 gmx_mpi mdrun -v -s min2.tpr -o min2.trr -x min2.xtc -c min2.gro -e min2.edr -g min2.log"

# EQL1

eval "${DOCKER} gmx_mpi grompp -f eql.mdp -o eql.tpr -p Complex.top -po eql.mdp -c min2.gro -maxwarn 1\""

docker run --user $(id -u):$(id -g) --rm --gpus all -it \
  -v ${mapdir}:${mapdir} -w ${mapdir} \
  -e OMP_NUM_THREADS=1 \
  gcr.io/cheminfosolutions-prod-02/gromacs-plumed:2024-3_AND_2-9-3 \
  bash -c \
  "mpirun -n 2 gmx_mpi mdrun -s eql.tpr -o eql.trr -x eql.xtc -c eql.gro -e eql.edr -g eql.log"


# EQL2


eval "${DOCKER} gmx_mpi grompp -f eql2.mdp -o eql2.tpr -p Complex.top -po eql2.mdp -c eql.gro -maxwarn 1\""

docker run --user $(id -u):$(id -g) --rm --gpus all -it \
  -v ${mapdir}:${mapdir} -w ${mapdir} \
  -e OMP_NUM_THREADS=1 \
  gcr.io/cheminfosolutions-prod-02/gromacs-plumed:2024-3_AND_2-9-3 \
  bash -c \
  "mpirun -n 4 gmx_mpi mdrun -s eql2.tpr -o eql2.trr -x eql2.xtc -c eql2.gro -e eql2.edr -g eql2.log"


# PRD


eval "${DOCKER} gmx_mpi grompp -f prd.mdp -o prd.tpr -p Complex.top -po prd.mdp -c eql2.gro -maxwarn 1\""


# List of temperatures (edit as needed)
temps=(300.         421.69650343 592.31752949 800.)


# Required input files
BASE_MDP="prd.mdp"  # Your template .mdp file
TOP="Complex.top"          # Your topology file
GRO="eql2.gro"             # Your input structure

# Clean up old replicas if any
rm -rf rep*

# Loop through each temperature
for i in "${!temps[@]}"; do
    T=${temps[$i]}
    REP_DIR="rep${i}"

    # Copy and modify the .mdp file for this temperature
    cp "$BASE_MDP" "prd_${T}.mdp"
    sed -i "s/ref-t.*/ref-t = $T/" "prd_${T}.mdp"
    sed -i "s/gen-temp.*/gen-temp = $T/" "prd_${T}.mdp"

    eval "${DOCKER} gmx_mpi grompp -f prd_${T}.mdp -p $TOP -c $GRO -o prd_${T}.tpr -maxwarn 1 -po prd_${T}.mdp\""

    # Create replica directory and move the .tpr file
    mkdir "$REP_DIR"
    mv "prd_${T}.tpr" "$REP_DIR/remd.tpr"
    cp plumed.dat $REP_DIR
    cp $ligand $REP_DIR 

    echo "Prepared $REP_DIR with T = $T K"
done


docker run --user $(id -u):$(id -g) --rm --gpus all -dit \
  -v ${mapdir}:${mapdir} \
  -w ${mapdir} \
  -e OMP_NUM_THREADS=1 \
  gcr.io/cheminfosolutions-prod-02/gromacs-plumed:2024-3_AND_2-9-3 \
  bash -c "source /usr/local/gromacs/bin/GMXRC && export OMP_NUM_THREADS=1 && mpirun -n 8 gmx_mpi mdrun -multidir rep0 rep1 rep2 rep3 -replex 1000  -plumed plumed.dat -pin on -pinoffset 0 -nsteps 250000000 --deffnm remd& > remds.log&"

