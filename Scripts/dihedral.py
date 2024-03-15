#!/ usr / bin / env python
import mdtraj as md
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def plot_torsion(dihe_dist, dihe_name, maxima):
    #Histogram of the data
    n, bins, patches = plt.hist(dihe_dist, 30, density=True, facecolor='g', alpha=0.75)
    #Inidcate Maxima
    for i in range(len(maxima)):
        plt.axvline(x = maxima[i], color = 'k')

    plt.xlabel('Torsional Angle(rad)')
    plt.ylabel('Probability')
    plt.xlim(-180, 180)
    plt.title('Histogram of Torsion Angle ' + dihe_name)
    plt.grid(True)
    plt.savefig('dihedrals/dihe_angle_' + dihe_name + '.png')
    plt.close()

def mdtraj_load(File_traj, File_gro, rm_solvent=True):
    if File_traj.split('.')[-1] != 'xtc': #Add file extension if not in input
        File_traj = File_traj + '.xtc'
    if File_gro.split('.')[-1] != 'gro': #Add default file extension if not in input
        File_gro = File_gro + '.gro'

    #Load trajectories
    traj = md.load(File_traj, top=File_gro)
    
    print('Trajectory Loaded')
    if rm_solvent == True:
        traj = traj.remove_solvent()
    return traj

def input_torsion(file_input, traj):
    input_ind = open(file_input, 'r').readlines()
    torsion_ind = np.zeros((len(input_ind), 4))
    torsion_name = []
    for i in range(len(input_ind)):
        line = input_ind[i].split()
        torsion_name.append(line[0])
        for j in range(4):
            torsion_ind[i,j] = traj.topology.select('resid ' + str(int(line[1])-offset) + ' and name ' + str(line[j+2]))
    return torsion_name, torsion_ind

def deter_multimodal(dihedrals, name, i):
    #Seperate dihedral angles
    dihe_dist = dihedrals[:,i]
    
    #Determine maxima for probability distribution
    maxima = compute_max(dihe_dist)

    #Determine data not in the main peak
    main_peak, other_peak = [], []
    for i in dihe_dist:
        if abs(i - maxima) < 40 or abs(i + 360 - maxima) < 40 or abs(i - 360 - maxima) < 40:
            main_peak.append(i)
        else:
            other_peak.append(i)
    all_maxima = [compute_max(main_peak)]

    #If greater than 5% outliers count as seperate peak
    while len(other_peak)/len(dihe_dist) > 0.15:
        maxima = compute_max(other_peak)
        new_dist = other_peak
        main_peak, other_peak = [], []
        for i in new_dist:
            if abs(i - maxima) < 40 or abs(i + 360 - maxima) < 40 or abs(i - 360 - maxima) < 40:
                main_peak.append(i)
            else:
                other_peak.append(i)
        if len(main_peak) > (0.10*len(dihe_dist)):
            all_maxima.append(compute_max(main_peak))
        else:
            break
    return all_maxima, dihe_dist

def compute_max(data):
    from scipy.stats import gaussian_kde
    import numpy as np

    kde = gaussian_kde(data)
    samples = np.linspace(min(data), max(data), 50)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]
    
    return maxima

#Declare arguments
parser = argparse.ArgumentParser(description = 'Determination of Ligand Conformers')
parser.add_argument('-t', required=True, help='File name for input trajectory')
parser.add_argument('-g', required=True, help= 'File name for input topology (gro format)')
parser.add_argument('-s', required=True, type = str, help= 'name res# name_atom1 name_atom2 name_atom3 name_atom4')

#Import Arguments
args = parser.parse_args()
File_traj = args.t
File_gro = args.g
file_input = args.s

#Output file
input_file = open(file_input, 'r').readlines()
output_file = open('dihe_ind_max.txt', 'w')

#Load Trajectory
traj = mdtraj_load(File_traj, File_gro, True, True)

#Set protein offset based on missing residues
offset = 1

#Load atom indices for torisonal angles from file
torsion_name, torsion_ind = input_torsion(file_input, traj)

#Compute dihedral angles for ligand
dihedral = md.compute_dihedrals(traj, indices=torsion_ind)

#Convert to degree
dihedral = dihedral*(180/np.pi)

#Plot and print angle distribution
dihe_max, dihe_ind = [],[]
for i in range(len(torsion_name)):
    maxima, dihe_dist = deter_multimodal(dihedral, torsion_name, i)
    plot_torsion(dihe_dist, torsion_name[i], maxima)
    #If multiple peaks add to dihe_max array
    dihe_max.append(maxima)
    dihe_ind.append(torsion_name[i])
    if len(maxima) > 1:
        define_torsion = input_file[i].strip('\n')
        output_file.write(define_torsion)
        for max in maxima:
            output_file.write(f' {max}')
        output_file.write('\n')

#Print conformer angle combinations, percent ligand is in conformation, and frame in which the ligand is in that conformation
df = pd.DataFrame({'Dihedral': dihe_ind, 'Max Values': dihe_max})
df.to_csv('conf_id.csv')


