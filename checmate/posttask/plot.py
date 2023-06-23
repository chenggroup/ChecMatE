import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from dpdata import MultiSystems
from scipy.stats import gaussian_kde



class DPPlot():

    def __init__(self, output_dir:str="./"):

        self.output_dir = output_dir


    def plot_MLP_error(
            self,
            train_e:list,
            train_f:list,
            test_e:list|None=None,
            test_f:list|None=None,
            plotname:str="dft_vs_mlp.png"
    ):

        label = [["$E_{MLP}$ [eV/atom]", "$E_{DFT}$ [eV/atom]"], ["$F_{MLP}$ [eV/Angstrom]", "$F_{DFT}$ [eV/Angstrom]"]]

        fig, axs = plt.subplots(2, 1, dpi=200, figsize=(6,6))
        plt.tight_layout()
        
        axs[0].set_xlim(min(train_e)-0.5, max(train_e)+0.5)
        axs[0].set_ylim(min(train_e)-0.5, max(train_e)+0.5)
        
        axs[1].set_xlim(min(train_f)-2, max(train_f)+2)
        axs[1].set_ylim(min(train_f)-2, max(train_f)+2)

        for i in [0, 1]:
            axs[i].set_aspect('equal', adjustable='box')
            axs[i].tick_params(labelsize=8)
            axs[i].set_xlabel(label[i][0], size=12)
            axs[i].set_ylabel(label[i][1], size=12)
            axs[i].plot([-10,10], [-10,10], linestyle="--", linewidth=0.5, c="black")

        axs[0].scatter(train_e[0], train_e[1], s=5, alpha=0.5)
        axs[1].scatter(train_f[0], train_f[1], s=5, alpha=0.5)

        if test_e is not None and test_f is not None:
            axs[0].scatter(test_e[0], test_e[1], s=5, alpha=0.5)
            axs[1].scatter(test_f[0], test_f[1], s=5, alpha=0.5)

        plt.savefig(fname=Path(self.output_dir)/plotname, dpi=300)

    
    def plot_distrubition(self, ms:MultiSystems, plotname:str="dataset_distribution.png"):

        fig, axs = plt.subplots(2, 1, dpi=150, figsize=(6,4))
        plt.tight_layout()

        energies, max_forces = [], []
        for i in ms:
            for j in i:
                energies.append(j.data["energies"][0])
                max_forces.append(abs(j.data["forces"][0]).max())

        axs[0].hist(energies, bins=60, linewidth=0.5, edgecolor="white")
        axs[0].set_xlabel("Energy [eV/A]")

        axs[1].hist(max_forces, bins=60, linewidth=0.5, edgecolor="white")
        axs[1].set_xlabel("Max Force [eV/Angstrom]")

        plt.savefig(fname=Path(self.output_dir)/plotname, dpi=300)




class ModelDeviPlot(): 

    def __init__(
        self, 
        output_dir: str='./',
        bounds: list=[0.1, 0.25]
    ):
        self.output_dir = output_dir
        self.bounds = bounds


    def plot_accuracy_bar(
        self,
        iter_list:list,
        accuracy: list|None=None,
        candidate: list|None=None,
#        fail: list|None=None,
        means: list|None=None,
        plotname:str="iteration_percentage.png"
    ):  
        
        accuracy = np.array(accuracy)
        candidate = np.array(candidate)
        assert len(accuracy) == len(candidate)
#        fail = np.array(fail)

        x = np.array(iter_list)
        width = 0.5
        fig, ax1 = plt.subplots(figsize=(5, 3)) if len(iter_list)<20 else plt.subplots(figsize=(5*0.25*(len(iter_list)-20), 3)) 

#        ax1.bar(x, accuracy*100+candidate*100+fail*100, width,color="midnightblue", label="failed")
        ax1.bar(x, [100]*len(iter_list), width,color="midnightblue", label="failed")
        ax1.bar(x, accuracy * 100 + candidate * 100, width,
            color="steelblue", label="candidate")
        ax1.bar(x, accuracy * 100, width, color="lightblue", label="accurate")

        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Percentage [%]", fontsize=12, color="#1f77b4")
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_xticks(x)
        ax1.tick_params(labelsize=8)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.legend(loc="lower center",
                bbox_to_anchor=(0.5, -0.35),
                ncol=3,
                borderaxespad=0,
                fontsize=10)

        if means is not None:
            ax2 = ax1.twinx()
            ax2.plot(x, means, color="orange", linewidth=1, marker="o", markersize=4, alpha=0.6)
            ax2.set_ylabel("Model deviation mean", fontsize=12, color="orange")
            ax2.tick_params(labelsize=8, axis="y", labelcolor="orange")

        plt.savefig(fname=Path(self.output_dir)/plotname, bbox_inches="tight", dpi=300)



    def plot_distribution(
        self,
        iter_list:list,
        model_devis:list,
        prefix:str="iter",
        plotname:str="model_devi_distribution.png",
        hist_bins:int|None=None
    ):
            
        numb_iter = len(iter_list)
        assert len(model_devis) == numb_iter

        fig, axs = plt.subplots(numb_iter, 1, figsize=(5, 1.5*numb_iter), dpi=150, sharex=True)
        fig.subplots_adjust(hspace=0)

        bounds = self.bounds
        xmax = bounds[1] + 0.5
        bins = np.linspace(0, xmax, int(600*xmax)) if hist_bins is None else np.linspace(0, xmax, hist_bins)
        colors = plt.get_cmap("GnBu", numb_iter+4)
        max_y = []
        for i, iter_idx in enumerate(iter_list):
            y, x = np.histogram(model_devis[i], bins)
            label = f"{prefix} {iter_idx:03d}"
            tot_num = len(model_devis[i])
            axs[i].plot(x[:-1], y/tot_num, label=label, color=colors([i+3]), alpha=1)
            max_y.append(y.max()/tot_num)
            
            axs[i].fill_between(x[:-1], y/tot_num, y2=-0.001, color=colors([i+4]), alpha=0.3)
            axs[i].legend(fontsize=8)
            
        max_y = np.array(max_y, dtype=float)
        
        for i in range(numb_iter):
            axs[i].set_xlim(0, xmax)
            axs[i].set_ylim(-0.001, max_y.max()*1.05)
            axs[i].vlines(x=bounds[0], ymin=0, ymax=max_y.max()*1.05, color="grey", linestyle="--", alpha=0.5)
            axs[i].vlines(x=bounds[1], ymin=0, ymax=max_y.max()*1.05, color="grey", linestyle="--", alpha=0.5)
            axs[i].tick_params(axis="both", which="major", labelsize=8)

        plt.xlabel("Max force deviation [eV/Angstrom]", fontsize=12)
        fig.text(0.0, 0.5, "Probability [a.u.]", va="center", rotation="vertical", size=12)
        plt.savefig(fname=Path(self.output_dir)/plotname, bbox_inches="tight", dpi=300)




class QEPlot():

    def __init__(self, output_dir:str="./"):

        self.output_dir = output_dir

    
    def plot_contour(
        self, 
        order_param:list,
        energy:list,
        colorbar_gap: int=50,
        plotname:str="Q_vs_E.png",
        prefix:str="$OP_2$",
        fmt:str="scatter"
    ):

        assert len(order_param) == len(energy)

        op = np.array(order_param)
        e = np.array(energy) - e.min()

        xy = np.vstack((op, e))
        z = gaussian_kde(xy)

        x = np.linspace(op.min(), op.max(), 100)
        y = np.linspace(0, e, 1000)
        X, Y = np.meshgrid(x, y)
        grid = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(z(grid).T, X.shape)*len(xy.transpose())

        rainbow = plt.get_cmap("rainbow_r", 128).copy()
        rainbow.set_under("white")
        rainbow.set_over("grey")

        colorbar_gap = colorbar_gap
        levels = list(range(1, int((max(Z.ravel())//colorbar_gap+2)*colorbar_gap), colorbar_gap))

        fig, ax = plt.subplots()
        ax.set_xlabel(f"Distance-weighted Steinhart order parameter ({prefix})", size=15)
        ax.set_ylabel("Energy [eV]", size=15)

        if fmt == "scatter":
            cs = ax.contour(X, Y, Z, colors="grey", levels=levels, alpha=0.8, linestyles="dotted")
            cax = ax.scatter(op, e, c=z.evaluate(points=xy)*len(xy.transpose()), cmap=rainbow, alpha=0.6, s=12)
        
        else:
            cax = ax.contourf(X, Y, Z, cmap=rainbow, levels=levels, extend='both', alpha=0.6)

        plt.colorbar(cax, label="Kernel density")
        plt.savefig(fname=Path(self.output_dir)/plotname, dpi=300)

