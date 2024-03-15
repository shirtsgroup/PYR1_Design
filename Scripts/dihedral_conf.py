#!/ usr / bin / env python
import mdtraj as md
import numpy as np
import argparse
import sys
import os.path
from itertools import product
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.constants import k, Avogadro
import random
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

def input_torsion(file_input, traj):
    input_ind = open(file_input, 'r').readlines()
    torsion_ind = np.zeros((len(input_ind), 4))
    torsion_name, max_values, peak_options = [], [], []
    for i in range(len(input_ind)):
        line = input_ind[i].split()
        torsion_name.append(line[0])
        for j in range(4):
            torsion_ind[i,j] = traj.topology.select('resid ' + str(int(line[1])-offset) + ' and name ' + str(line[j+2]))
        max_values.append(line[6:])
        opt = np.linspace(1, len(line[6:]), num=len(line[6:]), dtype=int)
        peak_options.append(list(opt))
    return torsion_name, torsion_ind, max_values, peak_options

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    idx2 = (np.abs(array - value + 360)).argmin()
    idx3 = (np.abs(array - value - 360)).argmin()
    if np.abs(array[idx]-value) <= np.abs(array[idx2] - value + 360) and np.abs(array[idx] - value) <= np.abs(array[idx3] - value - 360):
        return idx+1
    elif np.abs(array[idx2] - value + 360) <= np.abs(array[idx] - value) and np.abs(array[idx2] - value + 360) <= np.abs(array[idx3] - value - 360):
        return idx2+1
    else:
        return idx3+1

def clust_conf(traj, per, file_name):
    #Compute pairwise RMSD
    distances = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        distances[i] = md.rmsd(traj, traj, i, atom_indices=traj.topology.select('element != H'))
    
    #Perform Clustering
    reduced_distances = squareform(distances, checks=False)
    #np.savetxt('test_reduce_dist.txt', reduced_distances)
    link = linkage(reduced_distances, method='single') #The hierarchical clustering encoded as a matrix
    frame_list = dendrogram(link, no_labels=False, count_sort='descendent')['leaves']
    frame_cat = dendrogram(link, no_labels=False, count_sort='descendent')['color_list']

    #Keep only one file per cluster
    frames_sep = [] #List of frames that are unique and will be processed
    cat = frame_cat[0]
    frames_indv = [frame_list[0]]
    for frame in range(1, len(frame_list)-1):
        if frame_cat[frame] == cat:
            frames_indv.append(frame_list[frame])
        else:
            frames_sep.append(frames_indv)
            cat = frame_cat[frame]
            frames_indv = [frame_list[frame]]
    frames_sep.append(frames_indv)
    
    #Analyze each cluster (determine centroid and plot Intercluster RMSD)
    frames_unique, per_unique= compare_within_cluster(traj, frames_sep, per, distances[0], file_name)

    traj_clust = traj.slice(frames_unique)
    traj_clust.save_pdb(f'{file_name}.pdb')
    return traj_clust, per_unique, frames_sep

def compare_within_cluster(traj, cluster_frames, per, rmsd_all, file_name):
    per_unique = np.zeros(len(cluster_frames))
    frames_unique = []
    df_clust = pd.DataFrame()
    for f, frames in enumerate(cluster_frames):
        cluster_traj = traj.slice(frames)
        rmsd_clust = np.empty((cluster_traj.n_frames, cluster_traj.n_frames))
        rmsd_clust_mean = np.zeros(cluster_traj.n_frames)
        for i in range(cluster_traj.n_frames):
            rmsd_clust[i] = md.rmsd(cluster_traj, cluster_traj, i, atom_indices=cluster_traj.topology.select('element != H'))
            rmsd_clust_mean[i] = np.mean(rmsd_clust[i])
        min_index = np.argmin(rmsd_clust_mean)
        centroid_frame = frames[min_index]
        frames_unique.append(centroid_frame)
        for frame in frames:
            per_unique[f] += per[frame]
        df = pd.DataFrame({'Cluster ID': f+1, r'RMSD ($\AA$)': rmsd_clust[min_index]*10})
        df_clust = pd.concat([df_clust, df])
    #Save CSV
    df_clust.to_csv(f'{file_name}-clust-rmsd.csv')
    
    #Add Comparison of Not Clustered
    #df = pd.DataFrame({'Cluster ID': 'Not Clustered', r'RMSD ($\AA$)': rmsd_all*10})
    #df_clust = pd.concat([df_clust, df])

    #Plot Comparison
    plt.figure()
    g = sns.FacetGrid(df_clust, col="Cluster ID", col_wrap=5, xlim=(0,3))
    g.map(sns.histplot, r'RMSD ($\AA$)')
    plt.savefig(f'{file_name}-clust-rmsd.png')
    plt.close()    
    return frames_unique, per_unique

def process_confs(traj, per, file_name):
    #Save PDB
    traj.save_pdb(f'{file_name}.pdb')

    #Compute relative conformer energy
    rel_ener = get_rel_ener(per)

    #compute radius of gyration
    rg = md.compute_rg(traj)

    if end_pts != None:
        #Compute end point distance
        atom_pairs = [[traj.topology.select(f'name {end_pts[0]}')[0], traj.topology.select(f'name {end_pts[1]}')[0]]]
        ep_dist = md.compute_distances(traj, atom_pairs)
    
    #Save CSV
    df_clust = pd.DataFrame({'Conformer ID': np.linspace(1, len(per), num=len(per), dtype=int), 'Occupancy': per, 'Relative FE': rel_ener})
    df_non_zero = df_clust[df_clust['Occupancy'] > 0]
    df_non_zero['Radius of Gyration'] = rg
    if end_pts != None:
        df_non_zero['Molecule Width'] = ep_dist
    df_non_zero.to_csv(f'{file_name}.csv')

    labels = []
    for i, per in enumerate(df_non_zero['Occupancy']):
        if per > 1.5:
            labels.append(df_non_zero['Conformer ID'].values[i])
        else:
            labels.append('')
    df_order = df_non_zero.sort_values('Occupancy', ascending=False, inplace=False)
    plt.figure()
    plt.pie(df_non_zero['Occupancy'], labels=labels)
    plt.title(name)
    plt.savefig(f'{file_name}_pie.png')
    plt.close()

    plt.figure()
    sns.barplot(df_order, x='Conformer ID', hue='Relative FE', y='Relative FE', palette='cool_r', legend=False, order=df_order['Conformer ID'])
    plt.xlabel('Cluster ID', fontsize=16)
    plt.ylabel('Relative Free Energy (kcal/mol)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.savefig(f'{file_name}_FE.png')
    plt.close()

    plt.figure()
    sns.barplot(df_non_zero, x='Conformer ID', y='Radius of Gyration')
    plt.ylabel(r'Radius of Gyration($\AA$)')
    plt.savefig(f'{file_name}_rg.png')
    plt.close()

    plt.figure()
    sns.histplot(df_non_zero, x='Radius of Gyration')
    plt.xlabel(r'Radius of Gyration($\AA$)')
    plt.savefig(f'{file_name}_rg_hist.png') 
    plt.close()
    
    if end_pts != None:
        plt.figure()
        sns.histplot(df_non_zero, x='Molecule Width')
        plt.xlabel(r'Molecule Width ($\AA$)')
        plt.savefig(f'{file_name}_mw_hist.png') 
        plt.close()

def get_rel_ener(per_all):
    ref_per = np.max(per_all)
    rel_ener = np.zeros(len(per_all))
    for p, per in enumerate(per_all):
        rel_ener[p] = -(k/1000)*300*np.log(per/ref_per)*Avogadro/4.184
    return rel_ener

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

#Declare arguments
parser = argparse.ArgumentParser(description = 'Determination of Ligand Conformers')
parser.add_argument('-t', required=True, help='File name for input trajectory')
parser.add_argument('-g', required=True, help= 'File name for input topology (gro format)')
parser.add_argument('-s', required=True, type = str, help= 'name res# name_atom1 name_atom2 name_atom3 name_atom4')
parser.add_argument('-n', required=False, default = 'Lig', type = str, help= 'Ligand Name')
parser.add_argument('-e', required=False, nargs = '*', default = None, type = str, help= 'Ligand End Point Atoms')

#Import Arguments
args = parser.parse_args()
File_traj = args.t
File_gro = args.g
file_input = args.s
name = args.n
end_pts = args.e

#Load Trajectory
traj = mdtraj_load(File_traj, File_gro, True, True)

#Cluster all conformers
#traj_clust, per_clust, group = clust_conf(traj, np.ones(traj.n_frames), f'{name}_clust')
#process_confs(traj_clust, per_clust, f'{name}_clust')

#Set protein offset based on missing residues
offset = 1

#Load atom indices for torisonal angles from file
torsion_name, torsion_ind, max_values, peak_options = input_torsion(file_input, traj)

#Compute dihedral angles for ligand
dihedral = md.compute_dihedrals(traj, indices=torsion_ind)

#Convert to degree
dihedral = dihedral*(180/np.pi)

#Determine which dihderal peak is being sampled per frame
num_dihe = len(torsion_name)
dihe_peak_sampled = np.zeros((traj.n_frames, num_dihe))
for t in range(traj.n_frames):
    for i in range(num_dihe):
        max_value_i = np.array(max_values[i], dtype=float)
        value = dihedral[t,i]
        dihe_peak_sampled[t,i] = find_nearest(max_value_i, value)
print('Peaks Found')

#Name conformations
conf = list(product(*peak_options))
conformer = np.zeros((len(conf), len(conf[0])))
for c in range(len(conf)):
    conf_c = conf[c]
    conformer[c,:] = conf_c

#Classify dihedrals into conformations
count = np.zeros(len(conformer))
frame_select = []
for t in range(traj.n_frames):
    for i, conf in enumerate(conformer):
        if (conf == dihe_peak_sampled[t,:]).all():
            if count[i] == 0:
                frame_select.append(t)
            count[i] += 1
            break
per = 100*(count/traj.n_frames)

#Print conformer angle combinations, percent ligand is in conformation, and frame in which the ligand is in that conformation
df = pd.DataFrame({'Conformer ID': np.linspace(1, len(per), num=len(per), dtype=int)})
for i in range(num_dihe):
    df[f'Max for d{i+1}'] = conformer[:,i]
df.to_csv(f'{name}_dihe_def.csv')
traj_confs = traj.slice(frame_select)
process_confs(traj_confs, per, f'{name}_dihe')

#Cluster conformers
traj_dihe_clust, per_dihe_clust, group = clust_conf(traj_confs, per, f'{name}_dihe_clust')
df = pd.DataFrame({'Conformer ID': np.linspace(1, len(per_dihe_clust), num=len(per_dihe_clust), dtype=int), 'Grouped Confs': group})
df.to_csv(f'{name}_dihe_clust_def.csv')
process_confs(traj_dihe_clust, per_dihe_clust, f'{name}_dihe_clust')
