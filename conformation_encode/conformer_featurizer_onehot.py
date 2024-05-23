import numpy as np
from rdkit import Chem
import rdkit
from rdkit.Chem import AllChem
import deepchem as dc
from deepchem.feat.graph_data import GraphData
from deepchem.feat import MolecularFeaturizer
from rdkit.Chem import rdDistGeom
import torch

from .generate_conformations import generate_conformations


# This following codes are inherited from the deepchem.feat.molecule_featurizers.featurize.py of deepchem library

# similar to SNAP featurizer. both taken from Open Graph Benchmark (OGB) github.com/snap-stanford/ogb
# The difference between this and the SNAP features is the lack of masking tokens, possible_implicit_valence_list, possible_bond_dirs
# and the prescence of possible_bond_stereo_list,  possible_is_conjugated_list, possible_is_in_ring_list,

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],  # type: ignore
    'possible_chirality_list': [
        'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER', 'misc'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [
        -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

full_atom_feature_dims = list(
    map(
        len,  # type: ignore
        [
            allowable_features['possible_atomic_num_list'],
            allowable_features['possible_chirality_list'],
            allowable_features['possible_degree_list'],
            allowable_features['possible_formal_charge_list'],
            allowable_features['possible_numH_list'],
            allowable_features['possible_number_radical_e_list'],
            allowable_features['possible_hybridization_list'],
            allowable_features['possible_is_aromatic_list'],
            allowable_features['possible_is_in_ring_list']
        ]))

full_bond_feature_dims = list(
    map(
        len,  # type: ignore
        [
            allowable_features['possible_bond_type_list'],
            allowable_features['possible_bond_stereo_list'],
            allowable_features['possible_is_conjugated_list']
        ]))


def safe_index(feature_list, e):
    """
    Return index of element e in list l. If e is not present, return the last index

    Parameters
    ----------
    feature_list : list
        Feature vector
    e : int
        Element index to find in feature vector
    """
    try:
        return feature_list.index(e)
    except ValueError:
        return len(feature_list) - 1

############################################################################################################
#                                       Featurizer for one conformations                                   #
############################################################################################################
    

class RDKitConformerFeaturizer(MolecularFeaturizer):
    """
    A featurizer that featurizes an RDKit mol object as a GraphData object with 3D coordinates. 

    The ETKDGv3 algorithm is used to generate 3D coordinates for the molecule.
    MMMFF94s force field is used to optimize the 3D coordinates.
    The documentation can be found here:
    https://rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.ETKDGv3

    This featurization requires RDKit.
    Parameters
    ----------
    seed : int, optional, default=42
        The seed to use for the random number generator.
    max_attemp : int, optional, default=1000
        The maximum number of attempts to generate 3D coordinates for the molecule.
    
    Returns
    -------
    mol : RdkitMol
        RDKit molecule object containing the 3D conformer.
    graph : GraphData

    Examples
    --------
    >>> featurizer = RDKitConformerFeaturizer()
    >>> molecule = Chem.MolFromSmiles("CCO")
    >>> conformer, data = featurizer._featurize(molecule)
    >>> node_features = data.node_features  
    >>> edge_features = data.edge_features
    >>> edge_index = data.edge_index
    >>> node_pos_features = mol.GetConformer().GetPositions()
    """
    def __init__(self, seed: int = 42, max_attemp: int = 1000):
        self.seed = seed
        self.max_attemp = max_attemp


    def atom_to_feature_vector(self, atom):
        """
        Converts an RDKit atom object to a feature list of indices.

        Parameters
        ----------
        atom : Chem.rdchem.Atom
            RDKit atom object.

        Returns
        -------
        List[int]
            List of feature indices for the given atom.
        """
        atom_feature = [
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_atomic_num_list'],
                                    atom.GetAtomicNum())),
            num_classes=len(allowable_features['possible_atomic_num_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_chirality_list'],
                                    str(atom.GetChiralTag()))),
            num_classes=len(allowable_features['possible_chirality_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_degree_list'],
                                    atom.GetTotalDegree())),
            num_classes=len(allowable_features['possible_degree_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_formal_charge_list'],
                                    atom.GetFormalCharge())),
            num_classes=len(allowable_features['possible_formal_charge_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_numH_list'],
                                    atom.GetTotalNumHs())),
            num_classes=len(allowable_features['possible_numH_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_number_radical_e_list'],
                                    atom.GetNumRadicalElectrons())),
            num_classes=len(allowable_features['possible_number_radical_e_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_hybridization_list'],
                                    str(atom.GetHybridization()))),
            num_classes=len(allowable_features['possible_hybridization_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(allowable_features['possible_is_aromatic_list'].index(
                atom.GetIsAromatic())),
            num_classes=len(allowable_features['possible_is_aromatic_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(allowable_features['possible_is_in_ring_list'].index(
                atom.IsInRing())),
            num_classes=len(allowable_features['possible_is_in_ring_list'])
        ),
        ]
        return atom_feature
        
    def bond_to_feature_vector(self, bond):
        """
        Converts an RDKit bond object to a feature list of indices.

        Parameters
        ----------
        bond : Chem.rdchem.Bond
            RDKit bond object.

        Returns
        -------
        List[int]
            List of feature indices for the given bond.
        """
        bond_feature = [
        torch.nn.functional.one_hot(
            torch.tensor(safe_index(allowable_features['possible_bond_type_list'],
                                    str(bond.GetBondType()))),
            num_classes=len(allowable_features['possible_bond_type_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(allowable_features['possible_bond_stereo_list'].index(
                str(bond.GetStereo()))),
            num_classes=len(allowable_features['possible_bond_stereo_list'])
        ),
        torch.nn.functional.one_hot(
            torch.tensor(allowable_features['possible_is_conjugated_list'].index(
                bond.GetIsConjugated())),
            num_classes=len(allowable_features['possible_is_conjugated_list'])
        ),
        ]
        return bond_feature

    def _featurize(self, datapoint):
        """
        Featurizes a molecule into a graph representation with 3D coordinates.

        Parameters
        ----------
        datapoint : RdkitMol
            RDKit molecule object

        Returns
        -------
        mol: RdkitMol
        graph: List[GraphData] contains node_features, edge_features, edge_index
        """

        # add hydrogen bonds to molecule because they are not in the smiles representation
        self.mol = Chem.AddHs(datapoint)
        ps = AllChem.ETKDGv3()
        ps.useRandomCoords = False
        ps.randomSeed = self.seed
        ps.verbose = False
        ps.maxAttempts = self.max_attemp
        AllChem.EmbedMolecule(self.mol, ps)

        AllChem.MMFFOptimizeMolecule(self.mol, confId=0)
        self.conf = self.mol.GetConformer()
        self.coordinates = self.conf.GetPositions()

        atom_features_list = []
        for atom in self.mol.GetAtoms():
            atom_features_list.append(self.atom_to_feature_vector(atom))

        edges_list = []
        edge_features_list = []
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = self.bond_to_feature_vector(bond)
        

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)


        return self.mol, GraphData(
                         node_features=np.array(atom_features_list),
                         edge_features=np.array(edge_features_list),
                         edge_index=np.array(edges_list).T)
    



    

############################################################################################################
#                                       Featurizer for multiple conformations                              #
############################################################################################################
    
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot

from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_implicit_valence_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_explicit_valence_one_hot
from deepchem.utils.rdkit_utils import compute_all_pairs_shortest_path
from deepchem.utils.rdkit_utils import compute_pairwise_ring_info
    
class RDKitMultipleConformerFeaturizer(MolecularFeaturizer):
    """
    A featurizer that featurizes an RDKit mol object containing multiple conformations as a GraphData object with 3D coordinates.

    The ETKDGv3 algorithm is used to generate 3D coordinates for the molecule.
    MMMFF94s force field is used to optimize the 3D coordinates.
    The documentation can be found here:
    https://rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.ETKDGv2

    This featurization requires RDKit.
    Parameters
    ----------
    num_confs : int, optional, default=1
        The number of conformers to generate for each molecule.
    rmsd : float, optional, default=0.5
        The root-mean-square deviation (RMSD) cutoff value. Conformers with an RMSD
        greater than this value will be discarded.
    energy : float, optional, default=10
        The maximum energy difference between the lowest energy conformer and the highest energy conformer.
    seed : int, optional, default=42    
        The seed to use for the random number generator.
    max_attemp : int, optional, default=1000
        The maximum number of attempts to generate 3D coordinates for the molecule.
    num_threads : int, optional, default=-1
        The number of threads to use for conformer generation. If set to -1, the number of threads
        will be set to the number of available CPU cores.
    
    Returns
    -------
    mol: RdkitMol
        RDKit molecule object containing the 3D conformer.
    graph: List[GraphData] contains node_features, edge_features, edge_index

    """
    def __init__(self, num_confs: int = 1, seed: int = 42, rmsd: float = 0.5, energy: float=10, max_attemp: int = 1000,
                 num_threads: int = -1):
        self.num_confs = num_confs
        self.seed = seed
        self.max_attemp = max_attemp
        self.rmsd = rmsd
        self.energy = energy
        self.num_threads = num_threads


    def atom_to_feature_vector(self, atom, h_bond_infos):
        """
        Converts an RDKit atom object to a feature list of indices.

        Parameters
        ----------
        atom : Chem.rdchem.Atom
            RDKit atom object.

        Returns
        -------
        List[int]
            List of feature indices for the given atom.
        """
        # atom_feature_num = [
        #     safe_index(allowable_features['possible_formal_charge_list'],
        #                atom.GetFormalCharge()),
        #     safe_index(allowable_features['possible_number_radical_e_list'],
        #                atom.GetNumRadicalElectrons()),
        # ]
        atom_feature_num = [
                 get_atom_formal_charge(atom)[0],
                float(atom.GetNumRadicalElectrons()),
        ]
        atom_type  = get_atom_type_one_hot(atom)
        hybridization = get_atom_hybridization_one_hot(atom)
        acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
        aromatic = get_atom_is_in_aromatic_one_hot(atom)
        degree = get_atom_total_degree_one_hot(atom)
        total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
        chirality = get_atom_chirality_one_hot(atom)
        #partial_charge = get_atom_partial_charge(atom)

        atom_feat = np.concatenate([atom_type,
            atom_feature_num, hybridization, acceptor_donor, aromatic,
            degree, total_num_Hs,chirality
        ])

        return atom_feat
    
    def bond_to_feature_vector(self, bond):
        """
        Converts an RDKit bond object to a feature list of indices.

        Parameters
        ----------
        bond : Chem.rdchem.Bond
            RDKit bond object.

        Returns
        -------
        List[int]
            List of feature indices for the given bond.
        """
        bond_type = get_bond_type_one_hot(bond)
        same_ring = get_bond_is_in_same_ring_one_hot(bond)
        conjugated = get_bond_is_conjugated_one_hot(bond)
        stereo = get_bond_stereo_one_hot(bond)
        return np.concatenate([bond_type, same_ring, conjugated, stereo])

    def _featurize(self, datapoint):
        """
        Featurizes a molecule into a graph representation with 3D coordinates.

        Parameters
        ----------
        datapoint : RdkitMol
            RDKit molecule object

        Returns
        -------
        graph: List[GraphData]
            list of GraphData objects of the molecule conformers with 3D coordinates.
        """
        # add hydrogen bonds to molecule because they are not in the smiles representation
        self.mol = Chem.AddHs(datapoint)
        nconf = self.num_confs

        #Multiple conformers
        conf_generator = generate_conformations(self.mol, nconf, rmsd = self.rmsd, energy = self.energy, seed= self.seed, num_threads =self.num_threads , 
                                                max_attemp = self.max_attemp)
        self.mol, cids, remain_conf = conf_generator.gen_confs()
        #if len(cids) == 0:
            #return None, None, 0


        #Padding if number of conformers is less than nconf

        # while len(self.mol.GetConformers()) < nconf:
        #     self.mol.AddConformer(self.mol.GetConformer(0), assignId=True)
            
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features_list = []
        for atom in self.mol.GetAtoms():
            atom_features_list.append(self.atom_to_feature_vector(atom,h_bond_infos))

        edges_list = []
        edge_features_list = []
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = self.bond_to_feature_vector(bond)
        

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)


        return self.mol, GraphData(
                         node_features=np.array(atom_features_list),
                         edge_features=np.array(edge_features_list),
                         edge_index=np.array(edges_list).T
                         ), remain_conf