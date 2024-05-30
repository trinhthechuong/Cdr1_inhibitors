from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import combinations
import rdkit
from rdkit.Chem import rdDistGeom
import py3Dmol
from rdkit.Chem.Draw import IPythonConsole

class generate_conformations:
    """
    Generate number of conformations for a molecule by using ETKDG version 3 algorithm and 
    MMFF force field to optimize the conformations.

    Parameters
    ----------
    mol : RDKit molecule object
        The molecule for which the conformations are to be generated.
    num_confs : int
        The number of conformations to be generated.
    rmsd : float, optional
        The RMSD value to be used to remove similar conformations. The default is 0.5.
    energy : float, optional
        The energy value to be used to remove conformations with high energy. The default is 10.
    seed : int, optional
        The seed value for the random number generator. The default is 42.
    num_threads : int, optional 
        The number of threads to be used for the ETKDG algorithm. The default is 8.
    max_attemp : int, optional
        The maximum number of attempts to generate the conformations. The default is 1000.
    Returns
    -------
    mol : RDKit molecule object
        The molecule with the generated conformations.
    cids : list
    """
    def __init__(self, mol, num_confs, rmsd = 0.5, energy = 10, seed= 42, num_threads = 8, max_attemp = 1000):
        self.mol = Chem.AddHs(mol)
        self.num_confs = num_confs
        self.rmsd = rmsd
        self.energy = energy
        self.seed = seed
        self.num_threads = num_threads
        self.max_attemp = max_attemp
        self.remove_ids = []
    

    def remove_confs(self, mol, energy, rms):
        """
        Remove conformations with high energy and similar conformations based on RMSD.
        """
        e = []
        for conf in mol.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf.GetId())
            #ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())

            if ff is None:
                print(Chem.MolToSmiles(mol))
                return
            e.append((conf.GetId(), ff.CalcEnergy()))
        e = sorted(e, key=lambda x: x[1])

        if not e:
            return

        kept_ids = [e[0][0]]
        self.remove_ids = []

        for item in e[1:]:
            if item[1] - e[0][1] <= energy:
                kept_ids.append(item[0])
            else:
                self.remove_ids.append(item[0])

        if rms is not None:
            rms_list = [(i1, i2, AllChem.GetConformerRMS(mol, i1, i2)) for i1, i2 in combinations(kept_ids, 2)]
            while any(item[2] < rms for item in rms_list):
                for item in rms_list:
                    if item[2] < rms:
                        i = item[1]
                        self.remove_ids.append(i)
                        break
                rms_list = [item for item in rms_list if item[0] != i and item[1] != i]

        #print(f'Remove {len(self.remove_ids)} conformations')


        for cid in set(self.remove_ids):
            mol.RemoveConformer(cid)

    def acquỉred_energy_(self, mol):

        e = []
        for conf in mol.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf.GetId())

            if ff is None:
                print(f"Error molecule {Chem.MolToSmiles(mol)}")
                return
            e.append(ff.CalcEnergy())
        return e

        
            
            

    def gen_confs(self):
        mol = self.mol
        nconf = self.num_confs
        mol = Chem.AddHs(mol)
        etkdg = rdDistGeom.ETKDGv3()
        etkdg.randomSeed = self.seed
        etkdg.verbose = False
        etkdg.numThreads = self.num_threads
        etkdg.useRandomCoords = False
        etkdg.maxAttempts = self.max_attemp
        cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=nconf, params = etkdg)

        if len(cids) == 0:
            print('No conformations generated')
            
            # confs = self.gen_confs_obabel(mol, nconf=nconf)
            # for conf in confs:
            #     mol.AddConformer(conf.GetConformer())
            # cids = list(range(0, len(confs)))

        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid)
                #AllChem.UFFOptimizeMolecule(mol, confId=cid)
            except:
                continue
        self.remove_confs(mol, energy = self.energy, rms = self.rmsd)
        remain_conf = len(mol.GetConformers())
        #energy = self.acquỉred_energy_(mol)
        return mol, cids, remain_conf
    


def visualize_conformations(m, cids=[-1], p=None, removeHs=True,
           colors=('cyanCarbon','redCarbon','blueCarbon','magentaCarbon','whiteCarbon','purpleCarbon')):
        if removeHs:
            m = Chem.RemoveHs(m)
        if p is None:
            p = py3Dmol.view(width=1000, height=500)
        p.removeAllModels()
        for i,cid in enumerate(cids):
            IPythonConsole.addMolToView(m,p,confId=cid)
        for i,cid in enumerate(cids):
            p.setStyle({'model':i,},
                            {'stick':{'colorscheme':colors[i%len(colors)]}})
        p.zoomTo()
        return p.show()
            
            

            