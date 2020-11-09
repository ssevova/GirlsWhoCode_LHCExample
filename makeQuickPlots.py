#!/usr/bin/env python
"""
Script to get data and write to dataframes for ML training/testing
"""
__author__ = "Stanislava Sevova, Elyssa Hofgard"
###############################################################################                                   
# Import libraries                                                                                                
################## 
import argparse
import sys
import os
import re
import glob
import shutil
import uproot as up
import uproot_methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
###############################################################################                                   
# Command line arguments
######################## 
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Script to get data and write to dataframes for ML training/testing")
    parser.add_argument('-i',
                        '--indir',
                        dest='indir',
                        help='Directory with input files',
                        default="examples/")
    parser.add_argument('-o',
                        '--output',
                        dest='outdir',
                        help='Output directory for plots, selection lists, etc',
                        default='outdir')
    
    return parser
###############################################################################                                   
# Dataframes for each sample
############################ 
def sampleDataframe(infiles,treename): 
    """ Open the ROOT file(s) corresponding to the treename
    and put the relevant branches into a dataframe """
    
    ## Branches to read in 
    branches = ["w","in_vy_overlap",
                "trigger_lep",
                "passJetCleanTight",
                "n_mu","n_el","n_ph","n_bjet","n_baseph",
                "ph_pt","ph_eta","ph_phi","ph_isGood",
                "met_tight_tst_et", "met_tight_tst_phi",
                "mu_pt","mu_eta", "mu_phi",
                "el_pt","el_eta", "el_phi",
                "metsig_tst",
                "eventNumber",
                "passVjetsFilterTauEl"]
    df_sample = []
   
    tree = None
    for path, file, start, stop, entry in up.iterate(
            infiles+treename+"*.root",
            treename,
            branches=branches,
            reportpath=True, reportfile=True, reportentries=True):

        print('==> Processing sample: %s ...'%path)
        tree = up.open(path)[treename]
            
        if tree is not None:
            df_sample.append(tree.pandas.df(branches,flatten=False))
    df_sample = pd.concat(df_sample)
    return df_sample

def calcmT(met,ph):
    """
    Calculates the transverse mass between the met and photon
    """
    return np.sqrt(2*met.pt*ph.pt*(1-np.cos(ph.delta_phi(met))))

def calcPhiLepMet(lep1,lep2,met,ph):
    """
    Calculates the opening angle between the leptons (Z) and the met+photon (H)
    """
    return np.abs((lep1+lep2).delta_phi(met+ph))
        
def calcAbsPt(lep1,lep2,met,ph):
    """
    Calculates the absolute boson pT
    """
    pt_lep = (lep1+lep2).pt
    pt_ph_met = (met+ph).pt    
    return np.abs(pt_ph_met-pt_lep)/pt_lep

def getLorentzVec(df,lepType):
    """ 
    Calculates Lorentz vectors for the leptons and photons for bkg/sig:
    but first converts all pTs and masses from MeV to GeV
    """
    # Converting weights to true yield
    df['w'] = df['w'] * 36000
    df['mu_pt']   = df['mu_pt'].truediv(1000)
    df['el_pt']   = df['el_pt'].truediv(1000)
    df['mu_mass'] = df['mu_mass'].truediv(1000)
    df['el_mass'] = df['el_mass'].truediv(1000)
    df['ph_pt']   = df['ph_pt'].truediv(1000)
    df['met_tight_tst_et'] = df['met_tight_tst_et'].truediv(1000)

    if lepType == 'n_mu':
        lep_pt   = np.asarray(df.mu_pt.values.tolist()) 
        lep_eta  = np.asarray(df.mu_eta.values.tolist())
        lep_phi  = np.asarray(df.mu_phi.values.tolist())
        lep_mass = np.asarray(df.mu_mass.values.tolist())
    else:
        lep_pt   = np.asarray(df.el_pt.values.tolist()) 
        lep_eta  = np.asarray(df.el_eta.values.tolist())
        lep_phi  = np.asarray(df.el_phi.values.tolist())
        lep_mass = np.asarray(df.el_mass.values.tolist())
            
    lep1 = uproot_methods.TLorentzVectorArray.from_ptetaphim(lep_pt[:,0],lep_eta[:,0],lep_phi[:,0],lep_mass[:,0])
    lep2 = uproot_methods.TLorentzVectorArray.from_ptetaphim(lep_pt[:,1],lep_eta[:,1],lep_phi[:,1],lep_mass[:,1])

    ph_pt = np.asarray(df.ph_pt.values.tolist())
    ph_eta = np.asarray(df.ph_eta.values.tolist())
    ph_phi = np.asarray(df.ph_phi.values.tolist())

    ph = uproot_methods.TLorentzVectorArray.from_ptetaphim(ph_pt[0], ph_eta[0], ph_phi[0], 0.00)
    met = uproot_methods.TLorentzVectorArray.from_ptetaphim(df['met_tight_tst_et'].to_numpy(),
                                                            0.00, df['met_tight_tst_phi'].to_numpy(), 0.00)
    return lep1,lep2,ph,met

def calcVars(df):
    
    lepType = ['n_mu','n_el']

    df_list = []
    for lep in lepType:
        df_lep = df[df[lep]==2]

        vLep1,vLep2,vPh,vMET = getLorentzVec(df_lep, lep)

        df_lep['mT'] = calcmT(vMET,vPh)
        df_lep['dphi_mety_ll'] = calcPhiLepMet(vLep1,vLep2,vMET,vPh)
        df_lep['AbsPt'] = calcAbsPt(vLep1,vLep2,vMET,vPh)
        df_lep['Ptll'] = (vLep1 + vLep2).pt
        df_lep['Ptllg'] = (vLep1 + vLep2 + vPh).pt
        df_lep['mll'] = (vLep1 + vLep2).mass
        df_lep['mllg'] = (vLep1+vLep2+vPh).mass
        df_lep['lep1pt'] = vLep1.pt
        df_lep['lep2pt'] = vLep2.pt
        df_lep['dphi_met_ph'] = np.abs(vMET.delta_phi(vPh))
    
        df_list.append(df_lep)
    
    new_df = pd.concat(df_list)

    return new_df

def makePlots(df_sig,var,units):
    ### Plot some of the distributions for the signal
    fig,ax = plt.subplots(1,1)
    sig = np.array(df_sig[var])

    # scaling based on signal yield
    max_val = np.amax(sig)

    if var == 'mll':
        xmin = 60
        xmax = 120
    else: 
        xmin = 0
        xmax = max_val
    
    if var != 'w':        
        ax.hist(sig,weights = np.abs(df_sig['w'].to_numpy()),bins=50, range=(xmin, xmax), histtype='step', color='Blue',label='sig')

    ax.set_xlabel(var+' [' + units + ']')
    ax.set_ylabel('Events')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig("w_hist_" + var+ ".pdf",format="pdf")

def main(): 
    """ Run script"""
    options = getArgumentParser().parse_args()

    ### Make output dir
    dir_path = os.getcwd()
    out_dir = options.outdir
    path = os.path.join(dir_path, out_dir)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.chdir(path)

    ### Make all the signal dataframes
    df_sig = sampleDataframe(options.indir,"HyGrNominal")

    ### Filter the dataframe based on some criteria
    ### In this case, we want:
    ### 2 electrons or muons
    ### 1 photon and 1 base photon
    ### 0 b-jets
    ### the trigger_lep must be greater than 0
    ### etc...
    df_sig = df_sig[(df_sig['n_mu']==2) | (df_sig['n_el']==2)]
    df_sig = df_sig[(df_sig['trigger_lep']>0) &
                    (df_sig['passJetCleanTight']==1) &
                    (df_sig['n_ph']==1) &
                    (df_sig['n_baseph']==1) &
                    (df_sig['n_bjet']==0) &
                    (df_sig['passVjetsFilterTauEl']==True)]

    df_sig['mu_mass'] = list(np.full((len(df_sig),2),105.6))
    df_sig['el_mass'] = list(np.full((len(df_sig),2),0.511))

    ### Calculate some new variables using the 
    ### info in the dataframe
    df_sig = calcVars(df_sig)

    ### Filter the dataframe again,
    ### this time using the new variables!
    df_sig = df_sig[(df_sig['mll'] > 66) &
                    (df_sig['mll'] < 116) &
                    (df_sig['lep1pt'] > 26) &
                    (df_sig['lep2pt'] > 7) &
                    (np.asarray(df_sig['ph_pt'].values.tolist())[0] > 25) &
                    (df_sig['met_tight_tst_et'] > 50)]

    ### Make some cool plots!
    # Variables of interest
    var = ['met_tight_tst_et','met_tight_tst_phi','mT','dphi_mety_ll','AbsPt','Ptll','mllg','lep1pt','lep2pt','mll','metsig_tst','Ptllg','dphi_met_ph','w']
    units = ['GeV','Radians','GeV','Radians','','GeV','GeV','GeV','GeV','GeV',r'$\sqrt{GeV}$','GeV','Radians','','']
    
    for i in range(0,len(var)):
        makePlots(df_sig,var[i],units[i])

    
if __name__ == '__main__':
    main()
