
class DistributionData:
    def __init__(
            self, atom_feature, rdf_feature, bdf_feature, 
            idx_atom, idx_ele1, idx_ele2, idx_nbr, gap_gga, gap_hse, 
            element, std_rdf, std_bdf, fn, id):

        self.atom_feature = atom_feature
        self.rdf_feature  = rdf_feature
        self.bdf_feature  = bdf_feature

        self.idx_atom = idx_atom
        self.idx_ele1 = idx_ele1
        self.idx_ele2 = idx_ele2
        self.idx_nbr  = idx_nbr
        self.gap_gga  = gap_gga
        self.gap_hse  = gap_hse
        self.element  = element
        self.std_rdf  = std_rdf
        self.std_bdf  = std_bdf

        self.fn = fn
        self.id = id
