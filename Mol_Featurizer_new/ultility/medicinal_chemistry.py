from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Draw import IPythonConsole
from tqdm.auto import tqdm
tqdm.pandas()


def calculate_ro5_properties(smiles, fullfill = 4):
    """
    Test if input molecule (SMILES) fulfills Lipinski's rule of five.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.
    fullfill: int
        Number of rules fullfill RO5

    Returns
    -------
    bool
        Lipinski's rule of five compliance for input molecule.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate Ro5-relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    logp = Descriptors.MolLogP(molecule)
    tpsa = Descriptors.TPSA(molecule)
    # Check if Ro5 conditions fulfilled
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5, tpsa <= 140]
    ro5_fulfilled = sum(conditions) >= fullfill
    # Return True if no more than one out of four conditions is violated
    # return pd.Series(
    #     [molecular_weight, n_hba, n_hbd, logp, tpsa, ro5_fulfilled],
    #     index=["molecular_weight", "n_hba", "n_hbd", "logp", "tpsa", "ro5_fulfilled"],
    return ro5_fulfilled

def calculate_pfizer_rule(smiles):
    """
    Test if input molecule (SMILES) fulfills Pfizer Rule.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    bool
        Pfizer Rule compliance for input molecule.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate Pfizer Rule relevant chemical properties
    logp = Descriptors.MolLogP(molecule)
    tpsa = Descriptors.TPSA(molecule)
    # Check if Pfizer Rule conditions fulfilled
    conditions = [logp > 3, tpsa < 75]
    pfizer_fulfilled = sum(conditions) == 2
    # Return True if 2 conditions are both fulfilled
    return pfizer_fulfilled

def calculate_gsk_rule(smiles):
    """
    Test if input molecule (SMILES) fulfills GSK Rule.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    bool
        GSK Rule compliance for input molecule.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate GSK Rule relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    logp = Descriptors.MolLogP(molecule)
    # Check if GSK Rule conditions fulfilled
    conditions = [molecular_weight <= 400, logp <= 4]
    gsk_fulfilled = sum(conditions) == 2
    # Return True if 2 conditions are fulfilled
    return gsk_fulfilled

def calculate_goldentriangle_rule(smiles):
    """
    Test if input molecule (SMILES) fulfills GoldenTriangle Rule.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    bool
        GoldenTriangle Rule compliance for input molecule.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate GoldenTriangle Rule relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    logp = Descriptors.MolLogP(molecule)
    # Check if GoldenTrianlge Rule conditions fulfilled
    conditions = [200 <= molecular_weight <= 450,-2 <= logp <= 5]
    goldentriangle_fulfilled = sum(conditions) == 2
    # Return True if 2 conditions are fulfilled
    return goldentriangle_fulfilled


def calculate_qed(smiles):
    """
    Calculate QED and test if input molecule (SMILES) is 'attractive'.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [numpy.float64, bool]
        [QED, QED_excellent]
        QED for input molecule and 'attractive'-ness.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate QED of input molecule
    qed = Chem.QED.qed(molecule)
    # Check if QED conditions fulfilled
    qed_excellent = qed > 0.67
    # Return True if condition is fulfilled
    return [qed, qed_excellent]


from rdkit.Chem import RDConfig
import os
import sys
sascore_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
if sascore_path not in sys.path: sys.path.append(sascore_path)
import sascorer

def calculate_sascore(smiles):
    """
    Calculate sascore and test if input molecule (SMILES) is easy to synthesize.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [numpy.float64, bool]
        [SAscore, SAscore_excellent]
        SAscore for input molecule and synthetic accessibility.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate SAscore of input molecule
    SAscore = sascorer.calculateScore(molecule)
    # Check if SAscore condition is fulfilled
    SAscore_excellent = SAscore <= 6
    # Return True if condition is fulfilled
    return [SAscore, SAscore_excellent]

def calculate_fsp3(smiles):
    """
    Calculate Fsp3 and test if input molecule (SMILES) has suitable Fsp3 value.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [float, bool]
        [fsp3, fsp3_excellent]
        Fsp3 for input molecule and its suitability.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate Fsp3 of input molecule
    fsp3 = Chem.rdMolDescriptors.CalcFractionCSP3(molecule)
    # Check if Fsp3 condition is fulfilled
    fsp3_excellent = fsp3 >= 0.42
    # Return True if condition is fulfilled
    return [fsp3, fsp3_excellent]


from scopy.ScoFH import fh_filter

def pains_filter(smiles):
    """
    PAINS filter for an input molecule (SMILES).

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [bool, list, list]
        [pains_accepted, pains_matched_name, pains_matched_atoms]
        Check if PAINS not violated and matched names, atoms.
    """
    
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Check PAINS
    pains = fh_filter.Check_PAINS(molecule, detail = True)
    pains_accepted = pains['Disposed'] == 'Accepted' # Return True if not violating PAINS
    pains_matched_atoms = pains['MatchedAtoms']
    pains_matched_names = pains['MatchedNames']
    # Return PAINS
    return [pains_accepted, pains_matched_names, pains_matched_atoms]
    
    
def old_pains_filter(list_of_smiles):
    """
    PAINS filter for a list of input molecules (SMILES).

    Parameters
    ----------
    list_of_smiles : list of str
        List of SMILES.

    Returns
    -------
    [pd.DataFrame, pd.DataFrame]
        [matches, unmatches]
        Matches and unmatches for PAINS.
    """
    
    # Initialize filter
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    
    # Initialize list
    matches = []
    unmatches = []
    
    # Check each molecule
    for smiles in tqdm(list_of_smiles, total=list_of_smiles.shape[0]):
        molecule = Chem.MolFromSmiles(smiles)
        entry = catalog.GetFirstMatch(molecule)  # Get the first matching PAINS
        if entry is not None:
            # store PAINS information
            matches.append(
                {
                    "smiles": smiles,
                    "pains": entry.GetDescription().capitalize(),
                }
                )
        else:
            # collect molecules without PAINS
            unmatches.append({"smiles": smiles})
    
    matches = pd.DataFrame(matches)
    unmatches = pd.DataFrame(unmatches) # keep molecules without PAINS
    
    # Print result
    print(f"Number of compounds with PAINS: {len(matches)}")
    print(f"Number of compounds without PAINS: {len(unmatches)}")
    
    return [matches, unmatches]
    
def calculate_mce18(smiles):
    """
    Calculate MCE-18 and test if input molecule (SMILES) is interesting.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [float, bool]
        [MCE18, MCE18_excellent]
        MCE-18 for input molecule and its complex suitability.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate MCE-18 relevant properties
    from rdkit.Chem import rdMolDescriptors
    AR = rdMolDescriptors.CalcNumAromaticRings(molecule) > 0
    NAR = rdMolDescriptors.CalcNumAliphaticRings(molecule) > 0
    CHIRAL = len(Chem.FindMolChiralCenters(molecule, force = True, includeUnassigned = True)) > 0
    SPIRO = rdMolDescriptors.CalcNumSpiroAtoms(molecule)
    SP3 = Chem.rdMolDescriptors.CalcFractionCSP3(molecule)
    
    # Calculate Cyc and Acyc
    Csp3_cyclic = 0
    Csp3_acyclic = 0
    C_total = 0
    CYC = 0
    ACYC = 0
    
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 6: C_total+=1
        if sum([atom.GetAtomicNum() == 6, atom.IsInRing(), atom.GetHybridization() == Chem.HybridizationType.SP3]) == 3:
            Csp3_cyclic += 1
        if sum([atom.GetAtomicNum() == 6, not atom.IsInRing(), atom.GetHybridization() == Chem.HybridizationType.SP3]) == 3:
            Csp3_acyclic += 1
    
    if C_total>0:
        CYC = Csp3_cyclic/C_total
        ACYC = Csp3_acyclic/C_total
    
    # Calculate Q1
    deltas=[x.GetDegree() for x in molecule.GetAtoms()]
    M = sum(np.array(deltas)**2)
    N = molecule.GetNumAtoms()
    Q1 = 3-2*N+M/2.0
    
    # Calculate MCE-18
    mce18 = (AR + NAR + CHIRAL + SPIRO + (SP3 + CYC - ACYC)/(1 + SP3))*Q1
    
    # Check if MCE-18 condition is fulfilled
    mce18_excellent = mce18 >= 45
    # Return True if condition is fulfilled
    return [mce18, mce18_excellent]

from rdkit.Chem import RDConfig
import os
import sys
npscore_path = os.path.join(RDConfig.RDContribDir, 'NP_Score')
if npscore_path not in sys.path: sys.path.append(npscore_path)
import npscorer
fscore = npscorer.readNPModel()

def calculate_npscore(smiles):
    """
    Calculate NPscore of molecule.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    float
        NPscore for input molecule.
    """
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Calculate NPscore of input molecule
    npscore = npscorer.scoreMol(molecule, fscore)
    # Return NPscore
    return npscore

from scopy.ScoFH import fh_filter

def alarm_nmr_filter(smiles):
    """
    ALARM NMR filter for an input molecule (SMILES).

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [bool, list, list]
        [alarmnmr_accepted, alarmnmr_matched_names, alarmnmr_matched_atoms]
        Check if ALARM NMR not violated and matched names, atoms.
    """
    
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Check ALARM NMR
    alarmnmr = fh_filter.Check_Alarm_NMR(molecule, detail = True)
    alarmnmr_accepted = alarmnmr['Disposed'] == 'Accepted' # Return True if not violating ALARM NMR
    alarmnmr_matched_atoms = alarmnmr['MatchedAtoms']
    alarmnmr_matched_names = alarmnmr['MatchedNames']
    # Return ALARM NMR
    return [alarmnmr_accepted, alarmnmr_matched_names, alarmnmr_matched_atoms]

from scopy.ScoFH import fh_filter

def bms_filter(smiles):
    """
    BMS filter for an input molecule (SMILES).

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [bool, list, list]
        [bms_accepted, bms_matched_names, bms_matched_atoms]
        Check if BMS not violated and matched names, atoms.
    """
    
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Check BMS
    bms = fh_filter.Check_BMS(molecule, detail = True)
    bms_accepted = bms['Disposed'] == 'Accepted' # Return True if not violating BMS
    bms_matched_atoms = bms['MatchedAtoms']
    bms_matched_names = bms['MatchedNames']
    # Return BMS
    return [bms_accepted, bms_matched_names, bms_matched_atoms]

from scopy.ScoFH import fh_filter

def chelator_filter(smiles):
    """
    Chelator filter for an input molecule (SMILES).

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    [bool, list, list]
        [chelator_accepted, chelator_matched_names, chelator_matched_atoms]
        Check if Chelator not violated and matched names, atoms.
    """
    
    # RDKit molecule from SMILES
    molecule = Chem.MolFromSmiles(smiles)
    # Check Chelator
    chelator = fh_filter.Check_Chelating(molecule, detail = True)
    chelator_accepted = chelator['Disposed'] == 'Accepted' # Return True if not violating Chelator
    chelator_matched_atoms = chelator['MatchedAtoms']
    chelator_matched_names = chelator['MatchedNames']
    # Return Chelator
    return [chelator_accepted, chelator_matched_names, chelator_matched_atoms]


def calculate_all(smiles):
    """
    Calculate all rules.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.
    descriptors : bool
        Extract molecular descriptors of molecule. Default is 'False'.

    Returns
    -------
    pandas.Series
        All rules w/wo descriptors.
    """
    # Calculate all rules of molecule
    ro5_rule = calculate_ro5_properties(smiles)
    pfizer_rule = calculate_pfizer_rule(smiles)
    gsk_rule = calculate_gsk_rule(smiles)
    goldentriangle_rule = calculate_goldentriangle_rule(smiles)
    qed = calculate_qed(smiles)
    sascore = calculate_sascore(smiles)
    fsp3 = calculate_fsp3(smiles)
    mce18 = calculate_mce18(smiles)
    npscore = calculate_npscore(smiles)
    pains = pains_filter(smiles)
    alarmnmr = alarm_nmr_filter(smiles)
    bms = bms_filter(smiles)
    chelator = chelator_filter(smiles)
    
    # Calculate molecular desciptors
    return pd.Series(
            [ro5_rule, pfizer_rule, gsk_rule, goldentriangle_rule, qed[0], qed[1], 
            sascore[0], sascore[1], fsp3[0], fsp3[1], mce18[0], mce18[1], npscore, 
            pains[0], pains[1], pains[2], 
            alarmnmr[0], alarmnmr[1], alarmnmr[2], 
            bms[0], bms[1], bms[2],
            chelator[0], chelator[1], chelator[2]],
            index = ["ro5_rule", "pfizer_rule", "gsk_rule", "goldentriangle_rule", "qed", "qed_excellent", 
            "sascore", "sascore_excellent", "fsp3", "fsp3_excellent", "mce18", "mce18_excellent", "npscore",
            "pains_accepted", "pains_matched_names", "pains_matched_atoms",
            "alarmnmr_accepted", "alarmnmr_matched_names", "alarmnmr_matched_atoms",
            "bms_accepted", "bms_matched_names", "bms_matched_atoms",
            "chelator_accepted", "chelator_matched_names", "chelator_matched_atoms"]
        )
    
  
        
def calculate_filters(smiles):
    """
    Calculate only filters.

    Parameters
    ----------
    smiles : str
        SMILES for a molecule.

    Returns
    -------
    pandas.Series
        All rules.
    """
    # Calculate all filters of molecule
    pains = pains_filter(smiles)
    alarmnmr = alarm_nmr_filter(smiles)
    bms = bms_filter(smiles)
    chelator = chelator_filter(smiles)
    
    # Calculate molecular desciptors
    return pd.Series(
        [pains[0], pains[1], pains[2], 
         alarmnmr[0], alarmnmr[1], alarmnmr[2], 
         bms[0], bms[1], bms[2],
         chelator[0], chelator[1], chelator[2]],
        index = ["pains_accepted", "pains_matched_names", "pains_matched_atoms",
                 "alarmnmr_accepted", "alarmnmr_matched_names", "alarmnmr_matched_atoms",
                 "bms_accepted", "bms_matched_names", "bms_matched_atoms",
                 "chelator_accepted", "chelator_matched_names", "chelator_matched_atoms"]
    )
