
class Elastic():

    """
    Args:
        structure (Atoms): input structure to be optimized and run.
        strain_states (list of Voigt-notation strains): list of ratios of nonzero elements
            of Voigt-notation strain, e.g. [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), etc.].
        devis (list of floats, or list of list of floats): values of strain to multiply
            by for each strain state, i.e. devi for the perturbation along the strain
            state direction, e.g. [-0.01, -0.005, 0.005, 0.01].  If a list of lists,
            devis must correspond to each strain state provided.
        conventional (bool): flag to convert input structure to conventional structure,
            defaults to False.
        sym_reduce (bool): whether or not to apply symmetry reductions.
    """

    def __init__(self, conventional=False):
        self.conventional = conventional


    def get_deformed_matrixs(self, strain_states=None, devis=None):

        strains = []
        if strain_states is None:
            strain_states = self._get_default_strain_states()
        if devis is None:
            devis = [np.linspace(-0.01, 0.01, 5)] * len(strain_states)
        if np.array(devis).ndim == 1:
           devis = [devis] * len(strain_states)
        for state, devi in zip(strain_states, devis):
            strains.extend([Strain.from_voigt(s * np.array(state)) for s in devi])
        self._check_strains(strains)

        deformations = [s.get_deformation_matrix() for s in strains]
        return deformations
    

    def get_deformed_structures(self, structure:Atoms, deformations=None, sym_reduce=True):

        pym_structure = AseAtomsAdaptor.get_structure(structure)
        if self.conventional:
            pym_structure = SpacegroupAnalyzer(pym_structure).get_conventional_standard_structure()
        if deformations is None:
            deformations = self.get_deformed_matrixs()
        if sym_reduce:
            # Note this casts deformations to a TensorMapping with unique deformations as keys to symmops
            deformations = symmetry_reduce(deformations, pym_structure)
        deformed_pym_structures = []
        for i in deformations:
            deformed_pym_structures.append(DeformStructureTransformation(i).apply_transformation(pym_structure))
        deformed_structures = [AseAtomsAdaptor.get_atoms(i) for i in deformed_pym_structures]
        if sym_reduce:
            return deformed_structures, deformations
        else:
            return deformed_structures


    def _get_default_strain_states(self):
        """
        Generates a list of "strain-states"
        """

        inds = [(i,) for i in range(6)]
        strain_states = np.zeros((len(inds), 6))
        for n, i in enumerate(inds):
            np.put(strain_states[n], i, 1)
        strain_states[:, 3:] *= 2
        return strain_states.tolist()
    

    def _check_strains(self, strains):

        strains = [strain for strain in strains if not (abs(strain) < 1e-10).all()]
        vstrains = [strain.voigt for strain in strains]
        if np.linalg.matrix_rank(vstrains) < 6:
            raise ValueError("Strain list is insufficient to fit an elastic tensor")



class Substrate():

    def __init__(self, films:list, subs:list):

        self.films = [AseAtomsAdaptor.get_structure(film) for film in films]
        self.subs = [AseAtomsAdaptor.get_structure(sub) for sub in subs]
    

    def film_match_sub(self, film_max_miller:int=1, sub_max_miller:int=1, save:bool=True):

        all_matches = []
        for film in self.films:
            one_match = []
            sa = SubstrateAnalyzer(film_max_miller, sub_max_miller)

            for substrate in self.subs:
                matches_by_orient = self.groupby_itemkey(
                    sa.calculate(film, substrate, lowest=True),
                    "substrate_miller")

                lowest_matches = [min(g, key=lambda i: i.match_area)
                                for k, g in matches_by_orient]

                for match in lowest_matches:
                    db_entry = {
                        "sub_form": substrate.composition.reduced_formula,
                        "sub_orient": " ".join(map(str, match.as_dict()["substrate_miller"])),
                        "film_form": film.formula,
                        "film_orient": " ".join(map(str, match.as_dict()["film_miller"])),
                        "area": match.match_area,
                    }
                    one_match.append(db_entry)
            all_matches.append(one_match)
        if save:
            self.write(all_matches)
        else:
            return all_matches
    

    def write(self, all_matches, fpath:str='./'):

        for i, film in enumerate(self.films):
            df = pd.DataFrame(all_matches[i])
            df.sort_values("area")
            df.to_csv(f"{os.path.join(fpath, film.formula.replace(' ', ''))}.csv")
    

    def groupby_itemkey(iterable, item):
        
        return itertools.groupby(sorted(iterable, key=lambda i: i.as_dict()[item]), key=lambda i: i.as_dict()[item])