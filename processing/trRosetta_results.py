import numpy as np
from Bio import SeqIO


X = {"A": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], "E": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], "K": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
         "L": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], "D": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], "R": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
         "F": [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], "I": [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], "Q": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
         "W": [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], "G": [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], "S": [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
         "H": [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], "N": [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "M": [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         "P": [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "Y": [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "C": [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
         "V": [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "T": [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "X": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
         "Z": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], "B": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
         "U": [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "O": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], "ERROR": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}


chem_feat = {"A": [0.382,0.389,0.400,0.623,0.112,0.559,0.000], "E": [0.693,0.608,-0.778,0.946,0.369,0.299,0.335], "K": [0.689,0.734,-0.867,0.869,0.535,0.905,0.120],
         "L": [0.608,0.710,0.844,0.377,0.455,0.556,0.000], "D": [0.618,0.505,-0.778,1.000,0.257,0.257,0.502], "R": [0.839,0.832,-1.000,0.808,0.711,1.000,0.236],
         "F": [0.791,0.835,0.622,0.400,0.709,0.509,0.000], "I": [0.608,0.712,1.000,0.400,0.455,0.559,0.000], "Q": [0.688,0.645,-0.778,0.808,0.440,0.525,0.324],
         "W": [1.000,1.000,-0.200,0.415,1.000,0.547,0.047], "G": [0.306,0.275,-0.089,0.692,0.000,0.555,0.269], "S": [0.468,0.407,-0.178,0.708,0.152,0.528,0.516],
         "H": [0.736,0.688,-0.711,0.800,0.562,0.705,0.211], "N": [0.613,0.550,-0.778,0.892,0.328,0.503,0.484], "M": [0.705,0.724,0.422,0.438,0.540,0.533,0.000],
         "P": [0.521,0.531,-0.356,0.615,0.320,0.602,0.142], "Y": [0.876,0.851,-0.289,0.477,0.729,0.526,0.073], "C": [0.554,0.447,0.556,0.423,0.313,0.471,1.000],
         "V": [0.532,0.600,0.933,0.454,0.342,0.555,0.000], "T": [0.543,0.518,-0.156,0.662,0.264,0.546,0.258], "X": [0.639,0.628,-0.109,0.640,0.430,0.562,0.226],
         "Z": [0.691,0.627,-0.778,0.877,0.405,0.412,0.33], "B": [0.616,0.528,-0.778,0.946,0.293,0.38,0.493], "J": [0.608,0.711,0.922,0.389,0.455,0.558,0],
         "U": [0.554,0.447,0.556,0.423,0.313,0.471,1.000], "O": [0.689,0.734,-0.867,0.869,0.535,0.905,0.120],"ERROR": [0,0,0,0,0,0,0]}


def get_chemical_features(aa):
    sequence = np.array([])
    for i in range(700):
        try:
            if sequence.shape[0] == 0:
                sequence = np.array(chem_feat[aa[i]]).reshape(1, 7)
            else:
                sequence = np.concatenate((sequence, np.array(chem_feat[aa[i]]).reshape(1, 7)))
        except:
            if sequence.shape[0] == 0:
                sequence = np.array(chem_feat["ERROR"]).reshape(1, 7)
            else:
                sequence = np.concatenate((sequence, np.array(chem_feat["ERROR"]).reshape(1, 7)))

    return sequence.reshape(1, 700, 7)


def get_binary_sequence(aa):
    sequence = np.array([])
    for i in range(700):
        try:
            if sequence.shape[0] == 0:
                sequence = np.array(X[aa[i]]).reshape(1,21)
            else:
                sequence = np.concatenate((sequence, np.array(X[aa[i]]).reshape(1,21)))

        except:
            if sequence.shape[0] == 0:
                sequence = np.array(X["ERROR"]).reshape(1, 21)
            else:
                sequence = np.concatenate((sequence, np.array(X["ERROR"]).reshape(1, 21)))

    return sequence.reshape(1,700,21)


def get_condition(protein_class):
    condition = np.zeros([85])
    if 'IV' in protein_class:
        condition[56] = 1
    elif 'III' in protein_class:
        condition[42] = 1
    elif 'II' in protein_class:
        condition[28] = 1
    elif 'I' in protein_class:
        condition[14] = 1
    elif 'lipase' in protein_class:
        condition[0] = 1

    if "_S" in protein_class:
        condition[70] = 1
    elif "_F" in protein_class:
        condition[84] = 1

    return condition


def get_onehot_representation(fasta_file):
    seq = ""
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
    if len(seq) > 10:
        onehot = get_binary_sequence(seq)
        return onehot
    else:
        return None

def get_d_matrix(distances):
    # creating a matrix with two aa and the distances between them
    d_matrix = np.empty((0, 3), int)
    for i in range(0, len(distances[:, 0, 0])):
        for e in range(0, len(distances[:, 0, 0])):
            maxim = np.argmax(distances[i, e, 1:])
            if maxim < 12 and distances[i, e, 0] < 0.0005:  # only distances 2A-8A are considered
                d_matrix = np.append(d_matrix, [[i, i-e, maxim]], axis=0)

    # removing adjacent aa
    d_matrix_f = np.copy(d_matrix)

    ls = []
    for i in range(len(d_matrix[:, 0])):
        if d_matrix[i, 1] == d_matrix[i, 0] + 1 or d_matrix[i, 1] == d_matrix[i, 0] - 1:
            ls.append(i)
    d_matrix_f = np.delete(d_matrix_f, ls, axis=0)
    return d_matrix_f

def normalize_column(A):
    for col in range(1,6):
        A[:, col] = (A[:, col] - np.min(A[:, col])) / (np.max(A[:, col]) - np.min(A[:, col]))
    return A

def adding_angles(d_matrix_f, omega, phi, theta):
    matrix_a = np.empty((0, 6), float)

    for i in range(len(d_matrix_f[:, 0])):
        max_omega = np.argmax(omega[d_matrix_f[i, 0], d_matrix_f[i, 1], 1:])
        max_phi = np.argmax(phi[d_matrix_f[i, 0], d_matrix_f[i, 1], 1:])
        max_theta = np.argmax(theta[d_matrix_f[i, 0], d_matrix_f[i, 1], 1:])
        matrix_a = np.append(matrix_a,
                             [[d_matrix_f[i, 0], d_matrix_f[i, 1], d_matrix_f[i, 2], max_omega, max_phi, max_theta]],
                             axis=0)

    matrix_a = matrix_a[matrix_a[:, 2].argsort()]
    matrix_a = normalize_column(matrix_a)
    return matrix_a


def create_distance_matrix(npz_file):
    data = np.load(npz_file) #loading npz file

    distances = (data["dist"])
    omega = (data["omega"])
    phi = (data["phi"])
    theta = (data["theta"])

    d_matrix_f = get_d_matrix(distances)



    #creating a matrix with 6 columns having, aa1, aa2, dist, omega, phi, theta
    matrix_a = adding_angles(d_matrix_f, omega, phi, theta)

    def add_data(new_row, position, matrix_a, index):
        new_row[position] = matrix_a[index, 1]
        new_row[position + 1] = matrix_a[index, 2]
        new_row[position + 2] = matrix_a[index, 3]
        new_row[position + 3] = matrix_a[index, 4]
        new_row[position + 4] = matrix_a[index, 5]
        return new_row

    #creating a matrix with 100 columns, c0 = position of contact, c1 = distance, c2-c4 = angles
    matrix_b = np.empty((0, 75), float)
    for i in range(distances.shape[0]):
        result = np.where(matrix_a[:, 0] == i)[0]
        new_row = np.zeros((75), dtype=float)
        pos_in_row = 0
        for j in result[0:15]:
            #print(matrix_a[j])
            add_data(new_row, pos_in_row, matrix_a, j)
            pos_in_row = pos_in_row + 5
        #print(new_row)
        matrix_b = np.append(matrix_b, new_row.reshape(1, 75), axis=0)
    rest = np.zeros((700-matrix_b.shape[0], 75), dtype=float)
    matrix_b = np.append(matrix_b, rest, axis=0)

    return matrix_b

def process_secondary_structure_files(sec_file):
    matrix_s = np.empty((0, 3), int)
    secC = np.zeros([700])
    secH = np.zeros([700])
    secB = np.zeros([700])
    try:
        secfile = open(sec_file, "r")
        i = 0
        for line in secfile:
            if "Pred: " in line:
                line = line.replace("Pred: ", "")
                line = line.replace(" ", "")
                for letter in line:
                    if 'C' == letter:
                        secC[i] = 1
                        secH[i] = 0
                        secB[i] = 0
                        i = i + 1
                    elif 'H' == letter:
                        secC[i] = 0
                        secH[i] = 1
                        secB[i] = 0
                        i = i + 1
                    elif 'E' == letter:
                        secC[i] = 0
                        secH[i] = 0
                        secB[i] = 1
                        i = i + 1
        matrix_s = np.append(secC, secH)
        matrix_s = np.append(matrix_s, secB).reshape(1,3,700)
        matrix_s = np.swapaxes(matrix_s,1,2)
    except:
        print("No sec")

    return matrix_s
