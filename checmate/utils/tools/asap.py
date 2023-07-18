import glob
import numpy as np
from os import path
from scipy.spatial.distance import cdist

from asaplib.data.xyz import ASAPXYZ
from asaplib.hypers.hyper_soap import universal_soap_hyper
from asaplib.hypers.hyper_acsf import universal_acsf_hyper
from asaplib.cli.func_asap import cluster_process, set_reducer, map_save
from asaplib.cluster.ml_cluster_fit import LAIO_DB, sklearn_DB
from asaplib.cluster.ml_cluster_tools import get_cluster_size, get_cluster_properties
from asaplib.reducedim.dim_reducer import Dimension_Reducers

from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects

from ... import wflog
from ...pretask.inputs import load_config
from ...pretask.sets import update_dict


class CallASAP():

    def __init__(
        self,
        fxyz,
        output_dir,
        user_setting=None,
        properties=None,
        **kwargs
        ):

        self.asapxyz = ASAPXYZ(
            fxyz=fxyz,
            **kwargs
        )
        self.output_dir = output_dir
        self.dm = None
        self.dm_atomic = None
        self.km = None
        self.km_atomic = None

        if properties:
            self.asapxyz.load_properties(properties)

        self.setting = load_config(fname="AsapSet")
        if user_setting:
       	    update_dict(self.setting, user_setting)


    def gen_desc(self, whether_to_write: bool=False):

        asapxyz = self.asapxyz
        desc_setting = self.setting.get("gen_desc")
        desc_spec = self.get_desc_spec()

        asapxyz.compute_global_descriptors(
            desc_spec_dict=desc_spec,
            keep_atomic=desc_setting.get("peratom", False),
            n_process=desc_setting.get("n_process", 1)
        )

        asapxyz.save_state(path.join(self.output_dir, desc_setting.get("prefix", "ASAP-desc")))
        if whether_to_write:
            asapxyz.write(path.join(self.output_dir, desc_setting.get("prefix", "ASAP-desc")))
        else:
            self.dm = asapxyz.fetch_computed_descriptors(list(desc_spec.keys()))
            assert self.dm is not None
            if desc_setting.get("peratom", False):
                self.dm_atomic = asapxyz.fetch_computed_atomic_descriptors(list(desc_spec.keys()))


    def get_desc_spec(self):
        
        desc_setting = self.setting.get("gen_desc")
        desc_type = desc_setting.get("type", "soap")

        if desc_type == "cm":
            desc_spec = {desc_type:{'type': "CM"}}
        
        else:
            global_species = self.asapxyz.get_global_species()
            atomic_desc_spec = self.__get_atomic_desc_spec(
                desc_type, desc_setting.get(desc_type), global_species)

            reducer_spec = dict(set_reducer(
                desc_setting.get("reducer_type", "average"), 
                desc_setting.get("element_wise", False),
                desc_setting.get("zeta", 1)
            ))

            desc_spec = {}
            for k,v in atomic_desc_spec.items():
                desc_spec[k]={
                    'atomic_descriptor': {k:v},
                    'reducer_function': reducer_spec
                }
        
        return desc_spec
        

    def __get_atomic_desc_spec(self, desc_type, setting, global_species):

        if desc_type == "soap":
            atomic_desc_spec = self.__soap_spec(setting, global_species)
        
        elif desc_type == "acsf":
            atomic_desc_spec = self.__acsf_spec(setting, global_species)

        else:
            raise TypeError("The unknown type of atomic descriptor. Supported: acsf and soap.")

        return atomic_desc_spec


    def __soap_spec(self, soap_setting, global_species):
        
        usoap = soap_setting.get("usoap", "minimal")
        cutoff = soap_setting.get("cutoff", None)
        nmax = soap_setting.get("nmax", None)
        lmax = soap_setting.get("lmax", None)

        if usoap and cutoff is None and nmax is None and lmax is None:   
            soap_spec = universal_soap_hyper(global_species, usoap, dump=False)
    
        elif cutoff is not None and nmax is not None and lmax is not None:
            soap_spec = {'soap1': {'type': 'SOAP',
                                'cutoff': cutoff,
                                'n': nmax,
                                'l': lmax,
                                'atom_gaussian_width': soap_setting.get("sigma", 0.5)}}
        else:
            raise IOError("Please either use universal soap or specify the values of cutoff, nmax, and lmax.")
        
        rbf = soap_setting.get("rbf", "gto")
        crossover = soap_setting.get("crossover", False)

        for k in soap_spec.keys():
            soap_spec[k]['rbf'] = rbf
            soap_spec[k]['crossover'] = crossover

        return soap_spec


    def __acsf_spec(self, acsf_setting, global_species):

        cutoff = acsf_setting.get("cutoff", None)
        uacsf = acsf_setting.get("uacsf", "minimal")

        if cutoff is not None:
            acsf_spec = universal_acsf_hyper(global_species, cutoff, dump=False, verbose=False)
        else:
            acsf_spec = universal_acsf_hyper(global_species, uacsf, dump=False, verbose=False)

        return acsf_spec


    def map(self, whether_to_write: bool=False, dm=None, dm_atomic=None):

        map_setting = self.setting.get("map")
        ua = map_setting.get("use_atomic_descriptors", False)
        species = map_setting.get("only_use_species", None)

        if dm is not None or dm_atomic is not None:
            map_dm = dm
            map_dm_atomic = dm_atomic
        
        else:
            update_dict(self.setting, {"gen_desc":{"peratom": ua}})
            self.gen_desc()
            if ua:
                map_dm = None
                map_dm_atomic = self.__get_dm_atomic_species(species)
            else:
                map_dm = self.dm
                map_dm_atomic = self.dm_atomic
        
        reduce_dict = self.get_reduce_dict()
        km = self.map_process(map_dm, reduce_dict)
        km_atomic = self.map_process(map_dm_atomic, reduce_dict)

        if whether_to_write:
            map_save(
                path.join(self.output_dir, map_setting.get("prefix", "ASAP-map")), 
                map_setting.get("outmode", "xyz"),
                self.asapxyz,
                km,
                km_atomic,
                f"{map_setting.get('type', 'pca')}_reducer",
                species)

        else:
            self.km = km
            self.km_atomic = km_atomic


    def __get_dm_atomic_by_species(self, species=None):

        frames = self.asapxyz.frames
        dm_atomic = self.dm_atomic

        if species is None:
            return dm_atomic
        
        else:
            all_species = np.hstack(i.get_atomic_numbers() for i in frames)
            assert len(all_species) == len(dm_atomic) and species in all_species
            dm_atomic_species = np.vstack(dm_atomic[idx] for i, idx in enumerate(all_species) if i==species)
            return dm_atomic_species

    
    def get_reduce_dict(self):
        
        map_setting = self.setting.get("map")
        map_type = map_setting.get("type", "pca")
        setting = map_setting.get(map_type)
        
        if map_type == "pca":
            reduce_dict = self.__pca_reducer(setting)
        
        elif map_type == "skpca":
            reduce_dict = self.__skpca_reducer(setting)

        elif map_type == "tsne":
            reduce_dict = self.__tsne_reducer(setting)

        elif map_type == "umap":
            reduce_dict = self.__umap_reducer(setting)
        
        else:
            raise TypeError("The unknown type of reducer. Supported: pca, skpca, tsne, and umap.")

        return reduce_dict
    

    def __pca_reducer(self, pca_setting):

        reduce_dict = {
            "pca": {
                "type": "PCA",
                "parameter":{
                    "n_components": pca_setting.get("dimension", 10),
                    "scalecenter": pca_setting.get("scale", True)
        }}}
        return reduce_dict


    def __skpca_reducer(self, skpca_setting):

        scale = skpca_setting.get("scale", True)
        if scale:
            reduce_dict = {"preprocessing": {
                "type": "SCALE",
                "parameter": None
            }}
        
        reduce_dict["skpca"] = {
                "type": "SPARSE_KPCA",
                "parameter":{
                    "n_components": skpca_setting.get("dimension", 10),
                    "sparse_mode": skpca_setting.get("sparse_mode", "fps"),
                    "n_sparse": skpca_setting.get("n_sparse", -1),
                    "kernel": {
                        "first_kernel": {
                            "type": skpca_setting.get("kernel", "linear"),
                            "d": skpca_setting.get("kernel_parameter", None)
        }}}}
        return reduce_dict


    def __tsne_reducer(self, tsne_setting):

        pca = tsne_setting.get("pca", True)
        scale = tsne_setting.get("scale", True)
        
        if pca:
             reduce_dict = {"preprocessing": {
                "type": "PCA",
                "parameter": {
                    "n_components": 50,
                    "scalecenter": scale
            }}}
        elif scale:
            reduce_dict = {"preprocessing": {
                "type": "SCALE",
                "parameter": None
            }}

        reduce_dict["tsne"] = {
                "type": "TSNE",
                "parameter":{
                    "perplexity": tsne_setting.get("perplexity", 30),
                    "early_exaggeration": tsne_setting.get("early_exaggeration", 12),
                    "learning_rate": tsne_setting.get("learning_rate", 200),
                    "metric": tsne_setting.get("metric", "euclidean")
                }}
        return reduce_dict

    
    def __umap_reducer(self, umap_setting):

        scale = umap_setting.get("scale", True)
        if scale:
            reduce_dict = {"preprocessing": {
                "type": "SCALE",
                "parameter": None
            }}
        
        reduce_dict["umap"] = {
                "type": "UMAP",
                "parameter":{
                    "n_components": umap_setting.get("dimension", 10),
                    "n_neighbors": umap_setting.get("n_neighbors", 10),
                    "min_dist": umap_setting.get("min_dist", 0.1),
                    "metric": umap_setting.get("metric", "euclidean")
                }}
        return reduce_dict


    def map_process(self, map_matrix, reduce_dict):
        
        if map_matrix is None:
            return None

        else:
            dreducer = Dimension_Reducers(reduce_dict)
            proj = dreducer.fit_transform(map_matrix)
            return proj


    def cluster(self, whether_to_plot: bool=False, matrix=None):

        cluster_setting = self.setting.get("cluster")
        matrix_type = cluster_setting.get("matrix_type", "dm")
        ua = cluster_setting.get("use_atomic_descriptors", False)
        species = cluster_setting.get("only_use_species", None)

        matrix = matrix if matrix else self.get_matrix(ua, species, matrix_type)

        trainer_type = cluster_setting.get("type", "dbscan")
        assert trainer_type in ["dbscan", "fdb"]
        trainer = self.__dbscan_trainer(matrix, cluster_setting.get("dbscan")) if trainer_type=="dbscan" else self.__fdb_trainer()
        
        cluster_setting["prefix"] = path.join(self.output_dir, cluster_setting.get("prefix", "ASAP-cluster"))
        label_db = cluster_process(self.asapxyz, trainer, matrix, cluster_setting)
        label_ids = {}
        full_id = np.arange(len(label_db))
        for label in set(label_db):
            label_ids[label] = full_id[label_db == label]

        if whether_to_plot:
            self.__plot_cluster(labels=label_db, matrix=matrix, plot_setting=cluster_setting.get("plot_pca"))

        return label_ids


    def get_matrix(self, ua, species, matrix_type):

        assert matrix_type in ["dm", "km"]

        update_dict(
            self.setting, 
            {"gen_desc":{"peratom": ua}, "map":{"use_atomic_descriptors":ua, "only_use_species":species}}
        )

        if matrix_type == "dm":
            self.gen_desc()
            matrix = self.__get_dm_atomic_by_species(species) if ua else self.dm
        
        else:
            self.map()
            matrix = self.km_atomic if ua else self.km

        assert matrix is not None
        return matrix
            
        
    def __fdb_trainer(self):

        trainer = LAIO_DB()
        return trainer


    def __dbscan_trainer(self, matrix, dbscan_setting):

        metric = dbscan_setting.get("metric", "euclidean")
        eps = dbscan_setting.get("eps", None)
        if eps is None:
            n = len(matrix)
            #sub_structs = np.random.choice(np.asarray(range(n)), 50, replace=False)
            sub_structs = np.random.choice(np.asarray(range(n)), int(n/5), replace=False)
            percent = 1000/n if n<100 else 10000/n
            eps = np.percentile(cdist(matrix[sub_structs], matrix, metric), percent)
            #eps = np.percentile(cdist(matrix[sub_structs], matrix, metric), 100 * 10. / n)
            if eps>0.1:
                wflog.info("WARNING: The eps of DBSCAN is larger than 0.1, it is reseted to 0.1!")
                eps = 0.1
    
        trainer = sklearn_DB(eps, dbscan_setting.get('min_samples', 2), metric)
        return trainer


    def __plot_cluster(self, labels, matrix, plot_setting):

        proj = self.map_process(matrix, self.__pca_reducer(plot_setting))

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)

        axes = tuple(plot_setting.get("axes", (0, 1)))
        xy = proj[:, axes]
        ax.set_xlabel(f"Principal Axis {axes[0]+1}", fontsize=15, labelpad=-1)
        ax.set_ylabel(f"Principal Axis {axes[1]+1}", fontsize=15, labelpad=-3)

        annotate = plot_setting.get("annotate", None)
        self.__scatter_part(labels, fig, ax, xy, annotate)
        
        self.__cluster_label_part(labels, ax,xy)
            
        fig.savefig(path.join(self.output_dir, "cluster-pca"))

    
    def __scatter_part(self, labels, fig, ax, xy, tags):

        axscatter = ax.scatter(
            xy[:, 0], xy[:, 1],  c=np.asarray(labels), cmap=cm.summer, marker='o', s=200*200/len(xy)
        )
        
        cb = fig.colorbar(axscatter, format='%d')
        cb.ax.locator_params(nbins=5)
        cb.set_label(label='Cluster result', size=13)

        if tags is not None:
            tags = np.loadtxt(tags, dtype="str")[:]
            texts = []
            for i in range(len(tags)):
                if tags[i] !="" and str.lower(tags[i]) != "none":
                    texts.append(ax.text(xy[i, 0], xy[i, 1], tags[i],
                        ha='center', va='center', fontsize=12, color="black"))
            
    
    def __cluster_label_part(self, labels, ax, xy):

        label_unique = np.unique(labels)
        cluster_colors = [cm.summer(i) for i in np.linspace(0, 1, len(label_unique))]
        
        [_, cluster_mx] = get_cluster_properties(labels, xy[:, 0], 'mean')
        [_, cluster_my] = get_cluster_properties(labels, xy[:, 1], 'mean')
        [_, cluster_size] = get_cluster_size(labels)
        s = {}
        for k in label_unique:
            s[k] = np.log(cluster_size[k])

        for k, col in zip(label_unique, cluster_colors):

            if k >= 0:
                ax.plot(cluster_mx[k], cluster_my[k], 'o', alpha=0.5,
                        markerfacecolor=tuple(col),
                        markeredgecolor='grey', 
                        markersize=20 * s[k])

#                txt = ax.annotate(str(k), xy=(cluster_mx[k], cluster_my[k]), xytext=(0,0), 
#                    textcoords='offset points', fontsize=10, ha='center', va='center')
#                txt.set_path_effects([
#                            PathEffects.Stroke(linewidth=5, foreground='none'),
#                            PathEffects.Normal()])
            
            if k == -1:
                col = [0, 0, 0, 1]
                class_member_mask = (labels == k)
                xy = xy[class_member_mask]
                ax.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=0.5)
