#@title [RUN] Import python modules
import numpy as np
import math
import torch
import rdkit.Chem as Chem



def cal_node_spatial_feature(pos, cut_off = 3.5):
    """
    Calculate the node spatial feature based on the distance between atoms
    :param pos: the position of atoms/ tensor of shape (N, 3)
    :param cut_off: the cut off distance

    Return:
    spatial_feature: number of atom within the cut off distance, maximun distance and minimun distance in the cut off distance
    """

    dist = torch.cdist(pos, pos, p = 2)
    #Define minimum distance value are within the cut-off distance of each atom
    min_max_matrix = torch.zeros((len(pos), 2))
    for i in range (len(pos)):
        row_i_values = dist[i, (dist[i] < cut_off) &(dist[i] > 0)]
        if len(row_i_values) == 0:
            max_distance = 0
            min_distance = 0
        else:


            max_distance = torch.max(row_i_values)
            min_distance = torch.min(row_i_values)

            if min_distance == max_distance:
                min_distance = 0
            else:
                pass
        min_max_matrix[i] = torch.tensor([max_distance, min_distance])
    mask = (dist < cut_off) & (dist > 0)

    num_neighbors   = mask.sum(dim = 1).reshape(-1, 1)
    spatial_information = torch.cat((min_max_matrix,num_neighbors), dim = 1)

    return  spatial_information



def count_lone_pairs(a):
    """
    Count the number of lone pairs of an atom
    :param a: the atom object
    Return:
    lone_pairs: the number of lone pairs of the atom
    """
    tbl = Chem.GetPeriodicTable()
    v=tbl.GetNOuterElecs(a.GetAtomicNum())
    symbol_a = a.GetSymbol()
    c=a.GetFormalCharge()
    bond_type = []
    bond_type_encode = []
    for bond in a.GetBonds():
        bond_type.append(bond.GetBondType())
        bond_type_encode.append(bond.GetBondTypeAsDouble())
    
    num_aromatic_bond = bond_type.count(Chem.rdchem.BondType.AROMATIC)
    
    #Check C aromatic
    if (num_aromatic_bond != 0) & (symbol_a == "C"):
        lone_pairs = 0
        num_neighbors = len(a.GetNeighbors())
        #print("bond:",num_neighbors, "lone pairs", lone_pairs, "Aromatic", num_aromatic_bond )
        return lone_pairs
    
    else:
        b = sum(bond_type_encode)
        lone_pairs  = math.floor(0.5*(v-b-c))
        num_neighbors = len(a.GetNeighbors())
        #print("bond:",num_neighbors, "lone pairs", lone_pairs, "Aromatic", num_aromatic_bond )
        return lone_pairs

def atomic_shape(smiles):
    """
    Calculate the atomic shape of the molecule
    :param smiles: the smiles of the molecule
    Return:
    atomic_information: the number of neighbors, lone pairs and atomic shape of each atom in the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    #Number of neigbors of each atom
    neighbor_atoms = []
    for atom in mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        neighbor_atoms.append([neighbor.GetSymbol() for neighbor in neighbors])
    
    num_neighbors = [len(neighbors) for neighbors in neighbor_atoms]
    num_neighbors = torch.tensor(num_neighbors, dtype = torch.float).reshape(-1, 1)

    #Lone pairs

    lone_pairs = [count_lone_pairs(atom) for atom in mol.GetAtoms()]
    lone_pairs = torch.tensor(lone_pairs,dtype = torch.float).reshape(-1, 1)

    #Atomic shape
    atomic_shape_dict = {"Linear": torch.tensor([2,0], dtype = torch.float),
                     "Non_linear_1": torch.tensor([2,1],dtype = torch.float),
                     "None_linear_2": torch.tensor([2,2],dtype = torch.float),
                     "Trig_planar": torch.tensor([3,0],dtype = torch.float),
                        "Pyramidal": torch.tensor([3,1],dtype = torch.float),
                        "Tetrahedral": torch.tensor([4,0],dtype = torch.float),
                        "Square_planar": torch.tensor([4,2],dtype = torch.float),
                        "Trig_bipyramidal": torch.tensor([5,0], dtype = torch.float),
                        "Octahedral": torch.tensor([6,0], dtype = torch.float)}
    atomic_shape = [0,1,2,3,4,5,6,7,8,9]
    onehot_atomic_shape = torch.nn.functional.one_hot(torch.tensor(atomic_shape), num_classes = 10).to(torch.float)
    pairs_neighbors = torch.cat((num_neighbors, lone_pairs), dim = 1)
    atomic_shape_encode = torch.zeros((len(pairs_neighbors), 10))
    for key, value in enumerate(pairs_neighbors):
        if torch.allclose(value, atomic_shape_dict['Linear']):
            atomic_shape_encode[key] = onehot_atomic_shape[0]
        elif torch.allclose(value, atomic_shape_dict['Non_linear_1']):
            atomic_shape_encode[key] =onehot_atomic_shape[1]
        elif torch.allclose(value, atomic_shape_dict['None_linear_2']):
            atomic_shape_encode[key] = onehot_atomic_shape[2]
        elif torch.allclose(value, atomic_shape_dict['Trig_planar']):
            atomic_shape_encode[key] = onehot_atomic_shape[3]
        elif torch.allclose(value, atomic_shape_dict['Pyramidal']):
            atomic_shape_encode[key] = onehot_atomic_shape[4]
        elif torch.allclose(value, atomic_shape_dict['Tetrahedral']):
            atomic_shape_encode[key] = onehot_atomic_shape[5]
        elif torch.allclose(value, atomic_shape_dict['Square_planar']):
            atomic_shape_encode[key] = onehot_atomic_shape[6]
        elif torch.allclose(value, atomic_shape_dict['Trig_bipyramidal']):
            atomic_shape_encode[key] = onehot_atomic_shape[7]
        elif torch.allclose(value, atomic_shape_dict['Octahedral']):
            atomic_shape_encode[key] = onehot_atomic_shape[8]
        else:
            atomic_shape_encode[key] = onehot_atomic_shape[9]

    atomic_information = torch.cat((lone_pairs, atomic_shape_encode), dim = 1)
    return atomic_information, lone_pairs


def ring_len_from_bond(bond):
    """
    Get the length of the ring from the bond if the bond is in the ring
    """
    ri = bond.GetOwningMol().GetRingInfo()
    return [len(ring) for ring in ri.BondRings() if bond.GetIdx() in ring][0]

def ring_information(mol, edge_index):
    """
    Calculate the ring information of the molecule
    :param mol: the molecule object
    :param edge_index: the edge index of the molecule
    Return:
    ring_feature: the ring information of the molecule
    """
    #Determine if bond is in the ring or not
    bond_shape = []
    isinring = []
    for i in range(edge_index.shape[1]):
        bond = mol.GetBondBetweenAtoms(int(edge_index[0, i]), int(edge_index[1, i]))
        bond.IsInRing()
        if bond.IsInRing() == True:
            #isinring.append(1)
            bond_shape.append(ring_len_from_bond(bond))
        else:
            #isinring.append(0)
            bond_shape.append(0)
    #isinring = torch.tensor(isinring, dtype = torch.float).reshape(-1, 1)
    #print("isinring", isinring.shape)
    #isinring_onehot = torch.nn.functional.one_hot(isinring.to(torch.int64), num_classes = 2).to(torch.float)
    bond_shape = torch.tensor(bond_shape, dtype = torch.float).reshape(-1, 1)
    #print("isinring",isinring_onehot.shape, "bond_shape", bond_shape.shape)
    #ring_feature = torch.cat((isinring_onehot, bond_shape), dim = 1)
    return bond_shape


def calculate_cosine_angle(a, b, c):
    """
    Calculate the cosine angle between three points
    Parameters
    ----------
    a : array
        3D coordinates of point a
    b : array
        3D coordinates of point b
    c : array
        3D coordinates of point c
    Returns
    -------
    float
        The cosine angle between the three points
    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    return cosine_angle

def cal_spatial_angle(pos,mol,edge_index):
    """
    Calculate the spatial angle between the atoms
    Parameters
    ----------
    pos : array
        3D coordinates of the atoms
    mol : RDKit molecule
        RDKit molecule
    edge_index : array  
        The edge index of the molecule
    Returns
    -------
    tensor
        The spatial angle between the atoms
    """
    source_node_list = torch.zeros(edge_index.shape[1],8, dtype = torch.float)
    initial_list = list(range(0,len(mol.GetAtoms()),1))
    # for atom in mol.GetAtoms():
    #     atom_idx = atom.GetIdx()
    #     initial_list.append(atom_idx)

    for node_index in range(edge_index.shape[1]):
        node_source_idx = int(edge_index[0][node_index])
        node_target_idx = int(edge_index[1][node_index])
        exclude_idx = [node_source_idx, node_target_idx]
        neighbor_list = [x for x in initial_list if x not in exclude_idx]
        cosine_angles_source = []
        cosine_angles_target = []
        for neighbor_idx in neighbor_list:
            a = pos[node_target_idx]
            b = pos[node_source_idx]
            c = pos[neighbor_idx]
            cosine_angle_source = calculate_cosine_angle(pos[node_target_idx], pos[node_source_idx], pos[neighbor_idx])
            cosine_angles_source.append(cosine_angle_source)

            cosine_angle_target = calculate_cosine_angle(pos[node_source_idx], pos[node_target_idx], pos[neighbor_idx])
            cosine_angles_target.append(cosine_angle_target)




        max_cosine_angle_source = max(cosine_angles_source)
        min_cosine_angle_source = min(cosine_angles_source)
        mean_cosine_angle_source = np.mean(cosine_angles_source)
        std_cosine_angle_source = np.std(cosine_angles_source)

        max_cosine_angle_target = max(cosine_angles_target)
        min_cosine_angle_target = min(cosine_angles_target)
        mean_cosine_angle_target = np.mean(cosine_angles_target)
        std_cosine_angle_target = np.std(cosine_angles_target)
        
        angle_values = [max_cosine_angle_source, min_cosine_angle_source, mean_cosine_angle_source, std_cosine_angle_source,
                        max_cosine_angle_target, min_cosine_angle_target, mean_cosine_angle_target, std_cosine_angle_target]

        source_node_list[node_index] = torch.tensor(angle_values,dtype = torch.float)
        
    return source_node_list
    

def calculate_spatial_distance(a, b):
    """
    Calculate the spatial distance between two points
    Parameters
    ----------
    a : tensor
        3D coordinates of point a
    b : tensor
        3D coordinates of point b

    Returns
    -------
    float
        The spatial distance between the two points
    """
    distance = torch.norm(a - b).numpy()

    return distance

def cal_spatial_distance(pos):
    """
    Calculate the spatial distance between atoms
    Parameters
    ----------
    pos: torch.tensor
        The position of atoms
    Returns
    -------
    tensor of shape (num_atoms, 4)
    max_distance: the maximum distance between the atom and other atoms in space
    min_distance: the minimum distance between the atom and other atoms in space
    mean_distance: the mean distance between the atom and other atoms in space
    std_distance: the standard deviation of the distance between the atom and other atoms in space
    """

    num_atoms = pos.shape[0]
    source_node_list = torch.zeros(num_atoms,4, dtype = torch.float)
    initial_list = list(range(0, num_atoms,1))


    for node_index in initial_list:
        neighbor_list = [x for x in initial_list if x not in [node_index]]
        distances = []
        for neighbor_idx in neighbor_list:
            distance = calculate_spatial_distance(pos[node_index], pos[neighbor_idx])
            distances.append(distance)


        max_distance = np.max(distances)
        min_distance = np.min(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        distance_values = [max_distance, min_distance, mean_distance, std_distance]



        source_node_list[node_index] = torch.tensor(distance_values,dtype = torch.float)
    return source_node_list


