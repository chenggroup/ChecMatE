import glob
from os import path

from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.core import Spin
from sumo.electronic_structure.bandstructure import get_reconstructed_band_structure
from sumo.electronic_structure.effective_mass import fit_effective_mass, get_fitting_data

from ...pretask.inputs import MODULE_DIR
from ...pretask.sets import load_config, update_dict


class CallSumo():

    kpt_str = "[{k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}]"

    def __init__(self, fpath, user_setting=None):

        filenames = glob.glob(path.join(fpath, "vasprun.xml"))
        assert filenames != []

        self.filenames = filenames
        self.output_dir = path.dirname(filenames[0]) 
        if len(filenames) != 1:
             self.output_dir=path.dirname(self.output_dir) 

        default_setting = load_config(fpath=MODULE_DIR/"template"/"SumoSet.yaml")
        self.setting = update_dict(default_setting, user_setting) if user_setting else default_setting


    def run_bandstats(self, logfile="sumo-bandstat.log", run_type="both"):

        bandstrs = []
        add = bandstrs.append
        
        for f in self.filenames:
            vaspr = BSVasprun(f, parse_projected_eigen=False)
            bs = vaspr.get_band_structure(line_mode=True)
            add(bs)

        bs = get_reconstructed_band_structure(bandstrs, force_kpath_branches=False)

        if bs.is_metal():
            with open(path.join(self.output_dir, logfile), 'w') as flog:
                flog.write("ERROR: System is metallic!")
            raise TypeError("ERROR: System is metallic!")

        self.bandgap(bs, logfile)
        if run_type == "both":
            self.effective_mass(bs, logfile)
    

    def bandgap(self, bs, logfile):

        bg_data = bs.get_band_gap()
        indirect_bg = None if bg_data["direct"] else bg_data["energy"]
        direct_data = bs.get_direct_band_gap_dict()

        if bs.is_spin_polarized:
            direct_bg = min(spin_data["value"] for spin_data in direct_data.values())

        else:
            direct_bg = direct_data[Spin.up]["value"]
            
        if logfile:
            vbm_data = bs.get_vbm()
            cbm_data = bs.get_cbm()
            self.__log_bandgap(direct_bg, indirect_bg, vbm_data, cbm_data, logfile, bs.is_spin_polarized)
        
        else:
            return direct_bg, indirect_bg


    def __log_bandgap(self, direct_bg, indirect_bg, vbm_data, cbm_data, logfile, spin_polarized: bool=False):

        vbm_info = self.__get_bandedge_info(vbm_data, spin_polarized)
        cbm_info = self.__get_bandedge_info(cbm_data, spin_polarized)
        
        with open(path.join(self.output_dir, logfile), 'w') as flog:

            if indirect_bg:
                flog.write(f"Indirect band gap: {indirect_bg:.3f} eV\n")
            flog.write(f"Direct band gap: {direct_bg:.3f} eV\n")

            for idx, info in enumerate([vbm_info, cbm_info]):
                if idx == 0:
                    flog.write(f"\nValence band maximum:\n")
                else:
                    flog.write(f"\nConduction band maximum:\n")
                    
                flog.write(f"  Energy: {info[0]:.3f} eV\n")
                flog.write(f"  k-point: {info[1]}\n")
                flog.write(f"  k-point location: {info[2]}\n")
                if info[3]:
                    flog.write(f"  k-point indices: {info[3]}\n")
                flog.write(f"  k-point indices: {info[4]}\n")

    
    def __get_bandedge_info(self, edge_data, spin_polarized: bool=False):

        e = edge_data["energy"]
        kpt = edge_data["kpoint"]
        kpt_str = self.kpt_str.format(k=kpt.frac_coords)
        k_indices = ", ".join(map(str, edge_data["kpoint_index"]))
        k_loc = kpt.label if kpt.label else None

        b = edge_data["band_index"]
        if spin_polarized:
            b_indices = list((
                ", ".join(list((str(i+1) for i in b[key]))) 
                + f"({key.name.capitalize()})" for key in b.keys()
            ))
            b_indices = ", ".join(b_indices)
        else:
            b_indices = ", ".join(list((str(i+1) for i in b[Spin.up])))

        return e, kpt_str, k_loc, k_indices, b_indices
        

    def effective_mass(self, bs, logfile):

        vbm_data = bs.get_vbm()
        cbm_data = bs.get_cbm()

        setting = self.setting.get("bandstats")
        numb_sample_points = setting.get("sample-points", 3)
        parabolic = setting.get("parabolic", True)

        hole_data = self.__get_mass_data(bs, vbm_data, numb_sample_points, parabolic)
        elec_data = self.__get_mass_data(bs, cbm_data, numb_sample_points, parabolic)

        if logfile:
            with open(path.join(self.output_dir, logfile), 'a') as flog:
                if parabolic:
                    flog.write("\nUsing parabolic fitting of the band edges\n")
                else:
                    flog.write("\nUsing nonparabolic fitting of the band edges\n")

            self.__log_effective_mass(hole_data, logfile, bs.is_spin_polarized, "m_h")
            self.__log_effective_mass(elec_data, logfile, bs.is_spin_polarized)
        
        else:
            return {"hole_data": hole_data, "electron_data": elec_data}


    def __get_mass_data(self, bs, edge_data, numb_sample_points, parabolic):

        extremas = []
        e_extend = extremas.extend
        for spin, bands in edge_data["band_index"].items():
            e_extend(
                list((
                    (spin, band, kpoint)
                    for band in bands
                    for kpoint in edge_data["kpoint_index"]
                ))
            )
        
        mass_data = []
        d_extend = mass_data.extend
        for extrema in extremas:
            d_extend(
                get_fitting_data(bs, *extrema, numb_sample_points)
            )
        
        for data in mass_data:
            data["effective_mass"] = fit_effective_mass(
                data["distances"], data["energies"], parabolic=parabolic
            )
        
        return mass_data


    def __log_effective_mass(self, mass_data, logfile, spin_polarized:bool=False, mass_type="m_e"):
        
        kpt_str = self.kpt_str
        with open(path.join(self.output_dir, logfile), 'a') as flog:

            if mass_type=="m_e":
                flog.write("\nElectron effective masses:\n")
            else:
                flog.write("\nHole effective masses:\n")

            for data in mass_data:
                s = f" ({data['spin'].name})" if spin_polarized else ""

                band_str = f"band {data['band_id'] + 1}{s}"

                start_kpoint = data["start_kpoint"]
                end_kpoint = data["end_kpoint"]

                kpoint_str = kpt_str.format(k=start_kpoint.frac_coords)
                if start_kpoint.label:
                    kpoint_str += f" ({start_kpoint.label})"
                kpoint_str += " -> "
                kpoint_str += kpt_str.format(k=end_kpoint.frac_coords)
                if end_kpoint.label:
                    kpoint_str += f" ({end_kpoint.label})"

                flog.write(f"  {mass_type}: {data['effective_mass']:.3f} | {band_str} | {kpoint_str}\n")