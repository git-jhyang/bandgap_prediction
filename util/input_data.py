import os, torch, pickle, json
import numpy as np
import pandas as pd
from mendeleev.fetch import fetch_table
from mendeleev import element
from pymatgen.core.structure import Structure
from .data import DistributionData
from .math import angle_v, GaussianDistance
from tqdm import tqdm

'''
Data scaling Issue
- RDF/BDF를 STD로 나누어 편차를 1로 맞추게 되면 값이 증폭되어 훈련에 좋지 않은 영향을 줌
  (STD가 1 미만임. 0 데이터가 너무 많아서. STD 계산을 th 이상인 값으로 수정하면?)
'''
# af0
_atom_feat_names = [
    'atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
    'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat'
]

# af1
_atom_feat_names = [
    'atomic_number', 'atomic_volume', 'block', 'covalent_radius_pyykko',
    'electron_affinity', 'electronic_configuration', 'en_pauling', 'fusion_heat',
    'metallic_radius', 'vdw_radius_bondi', 'period'
]

# af2
#_atom_feat_names = [
#    'atomic_number', 'atomic_volume', 'covalent_radius_pyykko',
#    'electron_affinity', 'electronic_configuration', 'en_pauling', 'fusion_heat',
#    'metallic_radius', 'vdw_radius_bondi', 'period'
#]

class Dataset:
    def __init__(self, path=None, dfn='id_target.csv', pfx='POSCAR_ICSD-', sfx='',
                idx_id=None, idx_tgt=None, idx_ref=None,
                atom_feat_names=None, r_max=8, rb_max=5, dr=0.1, dt=0.1, 
                scale_atom=False, scale_rdf=False, scale_bdf=False):

        self._set(
            path=path, dfn=dfn, pfx=pfx, sfx=sfx,
            idx_id=idx_id, idx_tgt=idx_tgt, idx_ref=idx_ref,
            afn=atom_feat_names, r_max=r_max, rb_max=rb_max, dr=dr, dt=dt,
            scale_atom=scale_atom, scale_rdf=scale_rdf, scale_bdf=scale_bdf
            )


    def train_test_split(self, train_ratio=0.6, valid_ratio=0.2, rseed=35, metal_ratio=0.3, save_ids=False, path=None):
        np.random.seed(seed=rseed)
        crystal = self.crystal.copy()
        np.random.shuffle(crystal)
        target = np.array([cry.gap_hse.numpy().squeeze() for cry in crystal])
        metal_mask = (target == 0).astype(int)
        insul_mask = target != 0
        metal_mask = metal_mask * np.random.rand(metal_mask.shape[0]) > (1 - metal_ratio)
        mask = insul_mask | metal_mask

        crystal = np.array(crystal)[mask]
        i_train = int(len(crystal)*train_ratio)
        i_valid = i_train + int(len(crystal)*valid_ratio)

        print('{:.0f}% of metal data used - {} metal with {} insulator ({:.0f}%)'.format(
            metal_ratio*100, np.sum(metal_mask), np.sum(insul_mask), 100.0*np.sum(metal_mask)/np.sum(mask)
        ))

        if save_ids and path:
            if not os.path.isdir(path):
                raise NotADirectoryError(path)
            ids = dict(
                train_ids = [cry.id for cry in crystal[:i_train]],
                valid_ids = [cry.id for cry in crystal[i_train:i_valid]],
                test_ids  = [cry.id for cry in crystal[i_valid:]],
            )
            with open(os.path.join(path, 'ids.json'), 'w') as f:
                json.dump(ids, f, indent=4)

        if valid_ratio > 0:
            return crystal[:i_train], crystal[i_train:i_valid], crystal[i_valid:]
        else:
            return crystal[:i_train], crystal[i_train:]


    def generate_data(self):
        crystals = list()
        data = pd.read_csv(os.path.join(self.path, self.dfn))

        for _, data_row in tqdm(data.iterrows(), total=data.shape[0], desc='Generating data'):
            crys = self._read_cif(icsd_id=data_row[self.idx_id], gap_gga=data_row[self.idx_ref], gap_hse=data_row[self.idx_tgt])
            if crys is not None:
                crystals.append(crys)       
        self.crystal = crystals


    def save_dataset(self, fn, info=False, protocol=2):
        with open(fn, 'wb') as f:
            pickle.dump(self, f, protocol=protocol)
        if info: self.info()


    def load_dataset(self, fn, info=False):
        with open(fn, 'rb') as f:
            cls = pickle.load(f)
        if cls.__class__.__name__ == self.__class__.__name__:

            self._set(
                path=cls.path, dfn=cls.dfn, pfx=cls.pfx, sfx=cls.sfx,
                idx_id=cls.idx_id, idx_tgt=cls.idx_tgt, idx_ref=cls.idx_ref,
                afn=cls.afn, r_max=cls.r_max, rb_max=cls.rb_max, dr=cls.dr, dt=cls.dt, 
                scale_atom=cls._scale_atom, scale_rdf=cls._scale_rdf, scale_bdf=cls._scale_bdf
                )
            self.crystal = cls.crystal
            del cls
        if info: self.info()


    def scale_all_data(self, scale_atom:bool=False, scale_rdf:bool=False, scale_bdf:bool=False, 
                       atom_feat_names:list=None):

        is_scale_atom = np.sum([self._scale_atom, scale_atom]) == 1
        if not is_scale_atom and atom_feat_names: 
            is_scale_atom = True
        is_scale_rdf  = np.sum([self._scale_rdf, scale_rdf]) == 1
        is_scale_bdf  = np.sum([self._scale_bdf, scale_bdf]) == 1

        if np.sum([is_scale_atom, is_scale_rdf, is_scale_bdf]) == 0: 
            return
        
        crystal = []

        self._scale_atom = scale_atom
        self._scale_rdf  = scale_rdf
        self._scale_bdf  = scale_bdf

        if is_scale_atom:
            if atom_feat_names: self.afn = atom_feat_names
            self._load_mat_atom_feats()

        for data in tqdm(self.crystal, total=len(self.crystal), desc='Scaling data'):
            rdf_feature  = data.rdf_feature.numpy()
            bdf_feature  = data.bdf_feature.numpy()
            std_rdf      = data.std_rdf
            std_bdf      = data.std_bdf
            i_ele1       = data.idx_ele1.numpy()
            i_ele2       = data.idx_ele2.numpy()
            element      = data.element
            if is_scale_atom:
                element = np.array(element) - 1
                an_atom = element[i_ele1]
                an_ele  = element[i_ele2]
                data.atom_feature = torch.tensor(
                    np.hstack([
                        self.mat_atom_feats[an_atom, :], 
                        self.mat_atom_feats[an_ele, :]
                    ]), dtype=torch.float)
            if is_scale_rdf:
                if scale_rdf:
                    rdf_feature = rdf_feature / std_rdf
                else:
                    rdf_feature = rdf_feature * std_rdf
                data.rdf_feature = torch.tensor(rdf_feature, dtype=torch.float)
            if is_scale_bdf:
                if scale_bdf:
                    bdf_feature = bdf_feature / std_bdf
                else:
                    bdf_feature = bdf_feature * std_bdf
                data.bdf_feature = torch.tensor(bdf_feature, dtype=torch.float)


    def info(self):
        print('* Data ==============================================')
        print('   data path       : {}'.format(self.path))
        print('   n_Crystals      : {}'.format(len(self.crystal)))
        print('* Function ==========================================')
        print('   RDF r_max       : {}'.format(self.r_max))
        print('   RDF d_r         : {}'.format(self.dr))
        print('   RDF n_features  : {}'.format(self.n_rdf_feature))
        print('   BDF r_max       : {}'.format(self.rb_max))
        print('   BDF d_theta     : {}'.format(self.dt))
        print('   BDF n_features  : {}'.format(self.n_bdf_feature))
        print('* Others ============================================')
        print('   n_atom_features : {}'.format(len(self.afn)))
        print('   atom_features   : {}'.format(json.dumps(self.afn, indent=15)))
        print('* Scale data ========================================')
        print('   atom_features   : {}'.format(self._scale_atom))
        print('   rdf_features    : {}'.format(self._scale_rdf))
        print('   bdf_features    : {}'.format(self._scale_bdf))


    def _set(self, path, dfn, pfx, sfx, afn, idx_id, idx_tgt, idx_ref,
             r_max, rb_max, dr, dt, scale_atom, scale_rdf, scale_bdf):
        
        self.path   = path
        self.r_max  = r_max
        self.rb_max = rb_max
        self.dr     = dr
        self.dt     = dt
        self.pfx    = pfx
        self.sfx    = sfx
        self.dfn    = dfn
        self.afn    = afn if afn else _atom_feat_names

        self.idx_id  = idx_id if idx_id else 0
        self.idx_tgt = idx_tgt if idx_tgt else 3
        self.idx_ref = idx_ref if idx_ref else 1

        self._scale_atom = scale_atom
        self._scale_rdf  = scale_rdf
        self._scale_bdf  = scale_bdf

        self.rgd = GaussianDistance(0, dmax=self.r_max,  step=self.dr)
        self.bgd = GaussianDistance(0, dmax=3.15, step=self.dt)
        
        self.n_rdf_feature = self.rgd.filter.shape[0]
        self.n_bdf_feature = self.bgd.filter.shape[0]

        self.crystal = list()

        self._load_mat_atom_feats()


    def _read_cif(self, icsd_id, gap_gga, gap_hse):
        fn = os.path.join(self.path, 'source', f'{self.pfx}{icsd_id}{self.sfx}')
        try:
            crys = Structure.from_file(fn)
        except:
            print('Fail to read file: {}'.format(fn))
            return None
        element = np.array(sorted(set(crys.atomic_numbers)))
        if np.min(element) < 0 or np.max(element) > 95:
            print('Unsupported element type: {}'.format(fn))
            return None
        atom_feature, rdf_feature, idx_atom, idx_ele1, idx_ele2, idx_nbr = self._get_rdf(crys)
        bdf_feature = self._get_bdf(crys)

        std_rdf = np.std(rdf_feature)
        if std_rdf < 1e-6: std_rdf = 1
        std_bdf = np.std(bdf_feature)
        if std_bdf < 1e-6: std_bdf = 1

#        std_rdf = np.std(rdf_feature, axis=1).reshape(-1,1)
#        std_rdf[std_rdf < 1e-6] = 1
#        std_bdf = np.std(bdf_feature, axis=1).reshape(-1,1)
#        std_bdf[std_bdf < 1e-6] = 1

        if self._scale_rdf: rdf_feature = rdf_feature/std_rdf
        if self._scale_bdf: bdf_feature = bdf_feature/std_bdf

        rdf_feature  = torch.tensor(rdf_feature, dtype=torch.float)
        bdf_feature  = torch.tensor(bdf_feature, dtype=torch.float)
        atom_feature = torch.tensor(atom_feature, dtype=torch.float)
        idx_atom = torch.tensor(idx_atom, dtype=torch.int)
        idx_ele1 = torch.tensor(idx_ele1, dtype=torch.int)
        idx_ele2 = torch.tensor(idx_ele2, dtype=torch.int)
        idx_nbr = torch.tensor(idx_nbr, dtype=torch.int)
        gap_gga = torch.tensor(gap_gga, dtype=torch.float).view(-1, 1)
        gap_hse = torch.tensor(gap_hse, dtype=torch.float).view(-1, 1)
        
        return DistributionData(
            atom_feature=atom_feature, rdf_feature=rdf_feature, bdf_feature=bdf_feature,
            idx_atom=idx_atom, idx_ele1=idx_ele1, idx_ele2=idx_ele2, idx_nbr=idx_nbr, 
            gap_gga=gap_gga, gap_hse=gap_hse, element=element, 
            std_rdf=std_rdf, std_bdf=std_bdf, fn=fn, id=icsd_id)

    def _get_rdf(self, crys):
        atomic_numbers = np.array(crys.atomic_numbers) - 1
        element = sorted(set(atomic_numbers))
        map_element = [element.index(an) for an in atomic_numbers]
        rdf_feature = np.zeros((0, self.n_rdf_feature), dtype=float)
        an_atom = list()
        an_ele  = list()
        i_atom  = list()
        i_ele1  = list()
        i_ele2  = list()
        i_nbr   = list()
        all_nbrs = crys.get_all_neighbors(self.r_max, include_index=True)

        div = 1/(self.rgd.filter + 1e-6)**2

        for i, i_nbrs in enumerate(all_nbrs):
            an_atom += [atomic_numbers[i]]*len(element)
            an_ele  += element
            i_atom  += [i]*len(element)
            i_ele1  += [map_element[i]]*len(element)
            i_ele2  += [j for j in range(len(element))]
            rs   = {j:list() for j in range(len(element))}
            nbrs = {j:list() for j in range(len(element))}
            for nbr in i_nbrs:
                r, j = nbr[1:3]
                e_j  = map_element[j]
                rs[e_j].append(r)
                nbrs[e_j].append((r, j))
            for j in range(len(element)):
                if len(rs[j]) == 0:
                    feature = np.zeros((1, self.n_rdf_feature))
                else:
                    rdf = self.rgd.expand(np.array(rs[j]))
                    feature = np.sum(rdf, axis=0) * div
                nbr = [x[1] for x in sorted(nbrs[j], key=lambda x: x[0])][:6]
                if len(nbr) < 6:
                    nbr += [-1]*(6-len(nbr))
                
                rdf_feature = np.vstack([rdf_feature, feature])
                i_nbr.append(nbr)

        atom_feature = np.hstack([
            self.mat_atom_feats[an_atom, :], 
            self.mat_atom_feats[an_ele, :]
        ])

        i_atom = np.array(i_atom, dtype=int)
        i_ele1 = np.array(i_ele1, dtype=int)
        i_ele2 = np.array(i_ele2, dtype=int)
        i_nbr = np.array(i_nbr, dtype=int)
        
        return atom_feature, rdf_feature, i_atom, i_ele1, i_ele2, i_nbr

    def _get_bdf(self, crys):
        atomic_numbers = np.array(crys.atomic_numbers) - 1
        element = sorted(set(atomic_numbers))
        map_element = [element.index(an) for an in atomic_numbers]
        bdf_feature = np.zeros((0, self.n_bdf_feature), dtype=float)
        all_nbrs = crys.get_all_neighbors(self.rb_max, include_index=True)

        for i, nbrs in enumerate(all_nbrs):
            ci = crys[i].coords
            data = list()
            vecs = list()
            for nj, nbr_j in enumerate(nbrs):
                cj, rj, j = nbr_j[0:3]
                ij = map_element[j]
                v1 = cj.coords - ci
                for nk, nbr_k in enumerate(nbrs):
                    if nj == nk: continue
                    ck, rk, k = nbr_k[0:3]
                    ik = map_element[k]
                    v2 = ck.coords - ci
                    r  = rj + rk
                    data.append([ij, ik, r])
                    vecs.append([v1, v2])
            if len(data) == 0:
                bdf_feature = np.vstack([
                    bdf_feature,
                    np.zeros((len(element), self.n_bdf_feature))
                ])
                continue
            ij, ik, r = np.array(data).T
            vecs = np.array(vecs)
            a = self.bgd.expand(angle_v(vecs[:,0,:], vecs[:,1,:])).squeeze()
            w = self._w_bdf(r).reshape(-1,1)
            bdf = a*w
            for j in range(len(element)):
                bdf_feature = np.vstack([
                    bdf_feature, 
                    np.sum(bdf[ij == j], axis=0) + np.sum(bdf[ik == j], axis=0)
                ])
        return bdf_feature

    def _w_bdf(self, r):
        out = 1/np.array(r)
        out[out > 1] = 1
        out[out < 1e-3] = 0
        return out

    def _load_mat_atom_feats(self):
        tb_atom_feats = fetch_table('elements')[self.afn]
        ele_configs   = np.zeros((tb_atom_feats.shape[0], 0), dtype=float)
        if 'block' in self.afn:
            tb_atom_feats.loc[tb_atom_feats.loc[:, 'block'] == 's', 'block'] = 0.0
            tb_atom_feats.loc[tb_atom_feats.loc[:, 'block'] == 'p', 'block'] = 1.0
            tb_atom_feats.loc[tb_atom_feats.loc[:, 'block'] == 'd', 'block'] = 2.0
            tb_atom_feats.loc[tb_atom_feats.loc[:, 'block'] == 'f', 'block'] = 3.0
        if 'electronic_configuration' in self.afn:
            electronic_configuration = tb_atom_feats.pop('electronic_configuration')
            ele_configs = np.zeros((electronic_configuration.shape[0], 4), dtype=float)
            for i, config in electronic_configuration.iteritems():
                for orbit in config.split():
                    if '[' in orbit: continue
                    for j, qn in enumerate('spdf'):
                        if qn not in orbit: continue
                        _, k = orbit.split(qn)
                        if k == '': k = 1
                        ele_configs[i, j] += int(k)

        atom_feats = np.nan_to_num(np.hstack([np.array(tb_atom_feats, dtype=float), ele_configs])[:96, :])
        ion_engs = np.zeros((atom_feats.shape[0], 1))

        for i in range(0, ion_engs.shape[0]):
            ion_eng = element(i + 1).ionenergies

            if 1 in ion_eng:
                ion_engs[i, 0] = ion_eng[1]
            else:
                ion_engs[i, 0] = 0

        self.mat_atom_feats = np.hstack((atom_feats, ion_engs))
        if self._scale_atom: 
            self.mat_atom_feats = self.mat_atom_feats / np.std(self.mat_atom_feats, axis=0)
        self.n_atom_feats = self.mat_atom_feats.shape[1]