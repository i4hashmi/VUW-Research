from itertools import islice
import glob
import numpy
import math
import sys

#############################################################################################################
# Define dictionary to convert atomic symbols to covalent radii (in Angstrom)
SymbolToRadius = { "H"  : 0.37, "He" : 0.32, "Li" : 1.34, "Be" : 0.90, "B": 0.82,
"C"  : 0.77, "N"  : 0.75, "O"  : 0.73, "F"  : 0.71, "Ne" : 0.69, "Na" : 1.54,
"Mg" : 1.30, "Al" : 1.18, "Si" : 1.11, "P"  : 1.06, "S"  : 1.02, "Cl" : 0.99,
"Ar" : 0.97, "K"  : 1.96, "Ca" : 1.74, "Sc" : 1.44, "Ti" : 1.36, "V"  : 1.25,
"Cr" : 1.27, "Mn" : 1.39, "Fe" : 1.25, "Co" : 1.26, "Ni" : 1.21, "Cu" : 1.38,
"Zn" : 1.31, "Ga" : 1.26, "Ge" : 1.22, "As" : 1.19, "Se" : 1.16, "Br" : 1.14,
"Kr" : 1.10, "Rb" : 2.11, "Sr" : 1.92, "Y"  : 1.62, "Zr" : 1.48, "Nb" : 1.37,
"Mo" : 1.45, "Tc" : 1.56, "Ru" : 1.26, "Rh" : 1.35, "Pd" : 1.31, "Ag" : 1.53,
"Cd" : 1.48, "In" : 1.44, "Sn" : 1.41, "Sb" : 1.38, "Te" : 1.35, "I"  : 1.33,
"Xe" : 1.30, "Cs" : 2.25, "Ba" : 1.98, "La" : 1.69, "Ce" : 1.70, "Pr" : 1.70,
"Nd" : 1.70, "Pm" : 1.70, "Sm" : 1.70, "Eu" : 1.70, "Gd" : 1.70, "Tb" : 1.70,
"Dy" : 1.70, "Ho" : 1.70, "Er" : 1.70, "Tm" : 1.70, "Yb" : 1.70, "Lu" : 1.60,
"Hf" : 1.50, "Ta" : 1.38, "W"  : 1.46, "Re" : 1.59, "Os" : 1.28, "Ir" : 1.37,
"Pt" : 1.28, "Au" : 1.44, "Hg" : 1.49, "Tl" : 1.48, "Pb" : 1.47, "Bi" : 1.46,
"Po" : 1.50, "At" : 1.50, "Rn" : 1.45, "Fr" : 1.50, "Ra" : 1.50, "Ac" : 1.50,
"Th" : 1.50, "Pa" : 1.50, "U"  : 1.50, "Np" : 1.50, "Pu" : 1.50, "Am" : 1.50,
"Cm" : 1.50, "Bk" : 1.50, "Cf" : 1.50, "Es" : 1.50, "Fm" : 1.50, "Md" : 1.50,
"No" : 1.50, "Lr" : 1.50, "Rf" : 1.50, "Db" : 1.50, "Sg" : 1.50, "Bh" : 1.50,
"Hs" : 1.50, "Mt" : 1.50, "Ds" : 1.50, "Rg" : 1.50, "Cn" : 1.50, "Uut" : 1.50,
"Uuq" : 1.50, "Uup" : 1.50, "Uuh" : 1.50, "Uus" : 1.50, "Uuo" : 1.50}


#############################################################################################################
# This section is for the definition of all the functions used by the script
#############################################################################################################
#========================================================================================#
#-----------------------------------------------#
# Define a Function to read the XYZ input File  #
#-----------------------------------------------#
#-------------- Start of Function --------------#
def read_input_file(input_filename):
    f = open(input_filename, 'r')
    fr = f.readlines()[2:]
    f.close()
    # Split the string coord into sublists.
    for a in range(0,len(fr)):
        fr[a]=(fr[a].split())
        # Convert numbers to floats
        for b in range(1,4):
            fr[a][b]=float(fr[a][b])
    return fr
#--------------- End of Function ---------------#
#-----------------------------------------------#
# Define a Function to read all the XYZ Files   #
#-----------------------------------------------#
#-------------- Start of Function --------------#
def Read_Input_Files(input_filename):
    all_conformers = []
    for filename in input_filename:
        f = open(filename, 'r')
        fr = f.readlines()[2:]
        f.close()
        conf = []
        # Split the string coord into sublists.
        for a in fr:
            sublist = []
            a = a.split()
            sublist.append(a[0])
            sublist.append(float(a[1]))
            sublist.append(float(a[2]))
            sublist.append(float(a[3]))
            conf.append(sublist)
        all_conformers.append(conf)
    return all_conformers
#--------------- End of Function ---------------#

#========================================================================================#
#-----------------------------------------------#
# A function to Read VegaZZ Conformers XYZ File #
#-----------------------------------------------#
#-------------- Start of Function --------------#
def ReadVegaConf(file):
    file_open = open(file)
    n_atoms = int(file_open.readline())
    file_open.close()
    conformers = []
    with open(file) as f:
        iterations = iter(f)  # Iterate over the file and save to variable 'iterations'
        for line in iterations:
            sub = [l.split() for l in islice(iterations, 1, n_atoms + 1)]
            for j in sub:
                j[1] = float(j[1])
                j[2] = float(j[2])
                j[3] = float(j[3])
            conformers.append(sub)
    return conformers
#--------------- End of Function ---------------#
#========================================================================================#
#----------------------------------------------------------------------------#
# Define a custom range function (xfrange) to support floating point numbers #
#----------------------------------------------------------------------------#
#-------------- Start of Function --------------#
def xfrange(start, stop, step): #This one seems to be performing better
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1
#--------------- End of Function ---------------#
#========================================================================================#
#========================================================================================#
#-----------------------------------------------#
# Define a function to create a Distance Matrix #
#-----------------------------------------------#
#-------------- Start of Function --------------#
def DistanceMatrix(coordinates):
    distances = []
    for i in range(0, len(coordinates)):
        bond_dist = []
        for j in range(0, len(coordinates)):
            bond_dist.append(dist(coordinates[i], coordinates[j]))
        distances.append(bond_dist)
    return numpy.array(distances)
#--------------- End of Function ---------------#
#----------------------------------------------------------------------------------------------#
# Define a function to create a Distance Matrix and replace all the values greater than 4 to 4 #
#----------------------------------------------------------------------------------------------#
#-------------- Start of Function --------------#
def DistanceMatrix2(coordinates):
    distances = []
    for i in range(0, len(coordinates)):
        bond_dist = []
        for j in range(0, len(coordinates)):
            bond_dist.append(dist(coordinates[i], coordinates[j]))
        distances.append(bond_dist)
    # Below we look for elements in the list of lists 'distances' and if the element is greater than 4, we make its value 4.0
    for k in distances:
        for l in range(0, len(k)):
            if k[l] > 4.0:
                k[l] = 4.00000
    matrix = numpy.array(distances)
    return matrix
#--------------- End of Function ---------------#
#========================================================================================#
#-----------------------------------------------------#
# Define a function to Subtract two Distance Matrices #
#-----------------------------------------------------#
#-------------- Start of Function --------------#
def SubtractDistanceMatrices(Dist_matrices_conf):
    matrices = []
    for i in range(0, len(Dist_matrices_conf)):
        for j in range(i+1, len(Dist_matrices_conf)):
            new_matrix = numpy.subtract(Dist_matrices_conf[i], Dist_matrices_conf[j])
            max_element = new_matrix.max()
            print('Max Element', max_element)
            rms = numpy.sqrt(numpy.mean(numpy.square(new_matrix)))
            print('RMS', rms)
            matrices.append(new_matrix)
    return matrices
#--------------- End of Function ---------------#
#========================================================================================#
#--------------------------------------------------#
# Define a Function to Calculate the Bond Distance #
#--------------------------------------------------#
#-------------- Start of Function --------------#
def dist(atom1, atom2):
    dist = (atom1[1]-atom2[1])**2+(atom1[2]-atom2[2])**2+(atom1[3]-atom2[3])**2
    return math.sqrt(dist)
#--------------- End of Function ---------------#
#========================================================================================#
#========================================================================================#
#-------------------------------------------------------------------------------#
#       Define a Function to get the Bond Angles from three atoms provided      #
#-------------------------------------------------------------------------------#
#-------------- Start of Function --------------#
def GetBondAngle(atom1, atom2, atom3):
    vector1 = [atom1[1]-atom2[1], atom1[2]-atom2[2], atom1[3]-atom2[3]]
    vector2 = [atom3[1]-atom2[1], atom3[2]-atom2[2], atom3[3]-atom2[3]]
    mv1 = math.sqrt(numpy.dot(vector1, vector1)) #For vectors we cannot square ordinarily but use the dot product
    mv2 = math.sqrt(numpy.dot(vector2, vector2))
    a = math.acos((numpy.dot(vector1, vector2))/(mv1*mv2))
    return math.degrees(a)
#--------------- End of Function ---------------#
#========================================================================================#
#-------------------------------------------------------------------------------#
#   Function to find dihedral angles between four bonded atoms, e.g. A-B-C-D.   #
#-------------------------------------------------------------------------------#
# The A-B bond may be called as v1, B-C bond may be called by v2, and C-D bond may be called as v3
#-------------- Start of Function --------------#
def GetDihedralAngle(a1,a2,a3,a4):
    v1 = [a1[1]-a2[1],a1[2]-a2[2],a1[3]-a2[3]]
    v2 = [a3[1]-a2[1],a3[2]-a2[2],a3[3]-a2[3]]
    v3 = [a4[1]-a3[1],a4[2]-a3[2],a4[3]-a3[3]]
    n1 = numpy.cross(v1,v2)
    n2 = numpy.cross(v2,v3)
    v = numpy.divide(v2,(math.sqrt(numpy.dot(v2,v2))))
    m1 = numpy.cross(n1,n2)
    x_di = numpy.dot(n1,n2)
    y_di = numpy.dot(m1,v)
    di = math.atan2(y_di,x_di)
    return math.degrees(di)
#--------------- End of Function ---------------#
#========================================================================================#
#-----------------------------------------------------------------------------------------------------#
# Define a function to get the Bond Lengths, Bond Angles, & Dihedral Angles from provided Coordinates #
#-----------------------------------------------------------------------------------------------------#
#----------------------------------------- Start of Function -----------------------------------------#
def BondLengthsAnglesDihedrals(coordinates):
    List_of_Bonds = []
    List_of_Angles = []
    List_of_Dihedrals = []
    Bond_Lengths = []
    Bond_Angles = []
    Dihedral_Angles = []

    # A Loop to go through the coordinates and make a list of all possible bonds, omitting all the C-H bonds
    #-------------- Start of Loop --------------#
    for i in range(0,len(coordinates)):
        for j in range(i+1,len(coordinates)):
            if (coordinates[i][0] == 'C' and coordinates[j][0] == 'H') or (coordinates[i][0] == 'H' and coordinates[j][0] == 'C'): # If we detect a C-H or H-C bond, skip it and continue to the next
                continue# We are omitting all the CH bonds here
            else:
                if dist(coordinates[i],coordinates[j])<=(SymbolToRadius[coordinates[i][0]]+SymbolToRadius[coordinates[j][0]])*1.1: #We have to add 10% tolerance to show some C-C bonds so multiply with 1.1 for adding 10% tolerance.
                    List_of_Bonds.append([i,j])
    #--------------- End of Loop ---------------#
    #========================================================================================#
    # A Loop to go through the coordinates and make a list of all possible bond lengths
    #-------------- Start of Loop --------------#
    for i in range(0,len(coordinates)):
        for j in range(i+1,len(coordinates)):
            Bond_Lengths.append(dist(coordinates[i],coordinates[j]))
    #--------------- End of Loop ---------------#
    #========================================================================================#
    # A Loop to get the List of All Possible Bond Angle Combinations from the List of Bonds
    #-------------- Start of Loop --------------#
    for i in range(0, len(List_of_Bonds)):
        for j in range(i+1,len(List_of_Bonds)):
            if List_of_Bonds[i][0] == List_of_Bonds[j][0]:
                List_of_Angles.append((List_of_Bonds[i][1], List_of_Bonds[i][0], List_of_Bonds[j][1]))
            if List_of_Bonds[i][0] == List_of_Bonds[j][1]:
                List_of_Angles.append((List_of_Bonds[i][1], List_of_Bonds[i][0], List_of_Bonds[j][0]))
            if List_of_Bonds[i][1] == List_of_Bonds[j][0]:
                List_of_Angles.append((List_of_Bonds[i][0], List_of_Bonds[i][1], List_of_Bonds[j][1]))
            if List_of_Bonds[i][1] == List_of_Bonds[j][1]:
                List_of_Angles.append((List_of_Bonds[i][0], List_of_Bonds[i][1], List_of_Bonds[j][0]))
    #--------------- End of Loop ---------------#
    #========================================================================================#
    # A Loop to get the List of All Bond Angles from the List of Bonds
    #-------------- Start of Loop --------------#
    for angle in range(0, len(List_of_Angles)):
        Bond_Angles.append(GetBondAngle(coordinates[List_of_Angles[angle][0]], coordinates[List_of_Angles[angle][1]], coordinates[List_of_Angles[angle][2]]))
    #--------------- End of Loop ---------------#
    #========================================================================================#
    # A Loop to get the List of All Dihedral Angles from the List of Angles
    #-------------- Start of Loop --------------#
    for i in range(0,len(List_of_Angles)):
        for j in range(i+1,len(List_of_Angles)):
            if List_of_Angles[i][1] == List_of_Angles[j][0] and List_of_Angles[i][2] == List_of_Angles[j][1]:
                List_of_Dihedrals.append((List_of_Angles[i][0], List_of_Angles[i][1], List_of_Angles[i][2], List_of_Angles[j][2]))
            if List_of_Angles[i][1] == List_of_Angles[j][2] and List_of_Angles[i][2] == List_of_Angles[j][1]:
                List_of_Dihedrals.append((List_of_Angles[i][0], List_of_Angles[i][1], List_of_Angles[i][2], List_of_Angles[j][0]))
            if List_of_Angles[i][1] == List_of_Angles[j][0] and List_of_Angles[i][0] == List_of_Angles[j][1]:
                List_of_Dihedrals.append((List_of_Angles[i][2], List_of_Angles[i][0], List_of_Angles[i][1], List_of_Angles[j][2]))
            if List_of_Angles[i][1] == List_of_Angles[j][2] and List_of_Angles[i][0] == List_of_Angles[j][1]:
                List_of_Dihedrals.append((List_of_Angles[i][2], List_of_Angles[i][2], List_of_Angles[i][1], List_of_Angles[j][0]))
    #--------------- End of Loop ---------------#
    #========================================================================================#
    # A Loop to get the All the Dihedral Angles from the List of Dihedrals
    #-------------- Start of Loop --------------#
    for i in range(0, len(List_of_Dihedrals)):
        Dihedral_Angles.append(GetDihedralAngle(coordinates[List_of_Dihedrals[i][0]], coordinates[List_of_Dihedrals[i][1]],\
                                                coordinates[List_of_Dihedrals[i][2]], coordinates[List_of_Dihedrals[i][3]]))
    #--------------- End of Loop ---------------#
    #========================================================================================#

    #return Bond_Lengths, Bond_Angles, Dihedral_Angles
    return List_of_Bonds, Bond_Angles, Dihedral_Angles
#----------------------------------------- End of Function -----------------------------------------#
#========================================================================================#
#---------------------------------------------------------#
# Define a function to calculate RMS of a list of numbers #
#---------------------------------------------------------#
#-------------- Start of Function --------------#
def RMS(num_list):
    return math.sqrt(sum(n*n for n in num_list)/len(num_list))
#--------------- End of Function ---------------#
#========================================================================================#
#-----------------------------------------------------------------------------------------------------#
# Define a function to compare the Bond Lengths of conformers and produce a list of unique conformers #
#-----------------------------------------------------------------------------------------------------#
#-------------- Start of Function --------------#
def CompareConformers(conformers, dihedral_rms, max_element_dihedrals_cutoff):
    unique_conformers = [] # Unique_conformers is a list which contains only the unique conformers
    unique_conformers_names = []
    unique_conformers.append(conformers[0]) # Append the first conformer to the list of unique conformers
    Max_Elements = []
    for i in range(1, len(conformers)):
        for j in range(0, len(unique_conformers)):
            bond_len_conf01, bond_angles_conf01, dihedrals_conf01 = BondLengthsAnglesDihedrals(conformers[i])  # All the bond lengths, bond angles, and dihedrals of the ith conformer in 'conformers'
            bond_len_conf02, bond_angles_conf02, dihedrals_conf02 = BondLengthsAnglesDihedrals(conformers[j])  # All the bond lengths, bond angles, and dihedrals of jth conformer in 'unique_conformers'

            '''subtracted_bond_len = numpy.array(bond_len_conf01) - numpy.array(bond_len_conf02)
            RMS_bond_lengths = abs(RMS(subtracted_bond_len))  # Absolute value of the RMS difference between bond lengths of conf01 and conf02
            max_elem_bond_len = subtracted_bond_len.max()

            subtracted_bond_angles = numpy.array(bond_angles_conf01) - numpy.array(bond_angles_conf02)
            RMS_bond_angles = abs(RMS(subtracted_bond_angles)) # Absolute value of the RMS difference between bond angles of conf01 and conf02
            max_elem_bond_angles = subtracted_bond_angles.max()'''

            dihedrals_conf01 = [abs(k) for k in dihedrals_conf01] # Making all the values +ve
            dihedrals_conf02 = [abs(k) for k in dihedrals_conf02]
            subtracted_dihedrals = [x1 - x2 if x1 > x2 else x2 - x1 for (x1, x2) in zip(dihedrals_conf01, dihedrals_conf02)] # Subtract the two lists, i.e. dihedrals of conf01 and conf02, and look for the bigger value and subtract the smaller from it
            max_elem_dihedrals = numpy.array(subtracted_dihedrals).max()  # Get the max element from the suntracted list above
            RMS_dihedrals = abs(RMS(subtracted_dihedrals)) # Absolute value of the RMS difference between dihedral angles of conf01 and conf02

            #print(RMS_dihedrals)
            #print(max_elem_dihedrals)
            #print(subtracted_dihedrals)

            dm01 = DistanceMatrix2(conformers[i])  # Distance Matrix of the ith conformer in 'conformers'
            dm02 = DistanceMatrix2(unique_conformers[j])  # Distance matrix of jth conformer in 'unique_conformers'
            subtracted_dm = numpy.subtract(dm01, dm02)  # Subtraction of the matrices (dm01 - dm02)
            max_element_dm = subtracted_dm.max()  # Maximum element in 'subtracted'
            #RMS_dm = abs(RMS(subtracted_dm))  # Absolute value of the RMS difference between dm01 and dm02
            RMS_dm = numpy.sqrt(numpy.mean(numpy.square(subtracted_dm)))
            #print("RMS_Distance Matrix", i, RMS_dm)

            #if RMS_difference_dihedrals > float(bond_length_rms):  # Manual: RMS_diff_angles > 0.0655, RMS_diff_lengths > 0.00057, RMS_diff_dihedrals > 0.2307
            if RMS_dihedrals > float(dihedral_rms) and max_elem_dihedrals > max_element_dihedrals_cutoff: # and RMS_dm > 0.35:#\
                    #and RMS_dm > 0.02 or max_element_dm > 2:  # VEGA ZZ: RMS_diff_angles > 0.0655, RMS_diff_lengths > 0.00057, RMS_diff_dihedrals > 5.9

                unique_conformers.append(conformers[i]) if conformers[i] not in unique_conformers else None  # If a conformer meets the above criteria, add it to the list 'unique_conformers' avoiding duplicates
                Max_Elements.append(max_elem_dihedrals)
                #print("RMS_Dihedrals", i, RMS_dihedrals)
                #print("Max Element Dihedrals", j, max_elem_dihedrals)
                break
            else:
                None
    return unique_conformers#, Max_Elements
#--------------- End of Function ---------------#

#========================================================================================#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
###############################################################################
#                                                                             #
#                    The Main Part of the Program Starts Here                 #
#                                                                             #
###############################################################################
'''
#--------------------------------------------------------------#
# Conformational Sampling using OpenBabel through command line #
#--------------------------------------------------------------#
import subprocess
# Uncomment for a Boltzmann Weighted Conformer Search
#subprocess.call(["babel -i xyz obabel_conformers/xesto-d.xyz -o xyz -O obabel_conformers/xesto_confs.out --conformer --weighted --writeconformers"], shell=True)

# Uncomment for a Systematic Conformer Search
subprocess.call(["babel -i xyz obabel_conformers/xesto-d.xyz -o xyz -O obabel_conformers/xesto_confs.out --conformer --systematic --writeconformers"], shell=True)

conformers = ReadVegaConf('obabel_conformers/xesto_confs.out')

#--------------------------------------------------------------#'''

# Conformers is a list containing the coordinates of all the structures as individual lists of lists
#conformers = ReadVegaConf('vega_conformers.log')
#conformers = ReadVegaConf('test_01_minimized_vega.log')
#conformers = ReadVegaConf('test_conformers45.ali')
#conformers = read_input_file('xa01.log.xyz')
conformers = Read_Input_Files(glob.glob('final_selected_conformers_xyz/*.xyz'))
'''
#Test
bond_len_conf01, bond_angles_conf01, dihedrals_conf01 = BondLengthsAnglesDihedrals(conformers)

print(bond_len_conf01)

#Print Bonds and Bond Distances
print("The List of Bonds in Given Molecule is Given Below:")
print("Bond Representation (Bonding Atom Numbers)")
#for item in bond:
#    print(item)

for i in range(0,len(bond_len_conf01)):
    print(conformers[bond_len_conf01[i][0]][0],"-",conformers[bond_len_conf01[i][1]][0], (bond_len_conf01[i][0], bond_len_conf01[i][1]))
'''

Unique_Conformers = CompareConformers(conformers, 45.0, 45.0) # 4.0000, 150
print("#========================================================================================#")
print('Here are all the Unique Conformers')
print('Total conformers: ', len(conformers))
print('Unique conformers:', len(Unique_Conformers), '\n')

print("\nBelow are the Unique Conformers:\n")
for i in Unique_Conformers:
    print(i)
print("#========================================================================================#")
print("#========================================================================================#")

#rmsd_confs = RMSD_Spread(conformers)
#print(rmsd_confs)

'''
#========================================================================================#
# This loop is to save all the unique conformers as individual xyz files
#-------------- Start of Loop --------------#
for i, name in enumerate(Unique_Conformers):
    Title = (Unique_Conformers.index(name))+1 # Get the list index to use as number of conformer
    N_atoms = len(name) # Get number of atoms to print in the first line of xyz file
    f = open("Unique_Conformer"+str(i+1)+".xyz", "w")
    f.write(str(N_atoms)+"\n")
    f.write("Unique Conformer "+str(Title)+"\n")
    for j in name:
        f.write("{:<3}  {: .8f}  {: .8f}  {: .8f}".format(j[0], j[1], j[2], j[3])+"\n")
    f.close()
#--------------- End of Loop ---------------#
#========================================================================================#'''
'''
#======================================#
#final_result = []
rms_unique_conf = [['RMS Cutoff', 'Unique Conformers']]
for i in xfrange(45.0, 200.0, 3.0):
    for j in xfrange(45.0, 200.0, 3.0):
        unique_conformers = CompareConformers2(conformers, float(i), float(j))
        sublist = []
        if len(unique_conformers) < 500:
            #final_result.append(unique_conformers)
            del sublist[:]
            sublist.append(i)
            sublist.append(len(unique_conformers))
    rms_unique_conf.append(sublist)

#========================================================================================#
output_file_dihedral_angles = sys.stdout
sys.stdout = open('Dihedral_Angles_RMS.txt', "w")
print('The range of RMS is 45.0-200.0 with increments of 3.0')
for items in rms_unique_conf:
    print(items)
sys.stdout.close()
sys.stdout = output_file_dihedral_angles'''

#========================================================================================#

print('All Done Boss!')


#========================================================================================##========================================================================================#
#========================================================================================##========================================================================================#
#========================================================================================##========================================================================================#
