import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import f1_score, roc_auc_score


def convert_pathways(pathways, remove_compartment=False, exclusion_list=['Isolated', 'Dietary fiber binding',
                                                                         'Miscellaneous', 'Pool reactions',
                                                                         'Transport reactions',
                                                                         'Exchange/demand reactions',
                                                                         'Artificial reactions']):
    """Convert pathway membership file to python dictionary 

    Args:
        pathways (_type_): _description_
        remove_compartment (bool): include compartment details with metabolite ID
        :param exclusion_list: list of pathway names to remove from the data

    Returns:
        _type_: dictionary with pathway IDs as keys and members as lists of metabolites IDs
    """
    pathways = pathways.pivot(columns="subsystem", values="metabolite")
    pathways_d = {}
    for p in pathways.columns:
        pathways_d[p] = []
        for m in pathways[p]:
            if not pd.isna(m):
                if remove_compartment:
                    pathways_d[p].append(m[:-1])
                else:
                    pathways_d[p].append(m)
        pathways_d[p] = list(set(pathways_d[p]))

    for k in exclusion_list:
        pathways_d.pop(k)

    return pathways_d


class ORA:
    """
    Class for overrepresentation analysis 
    Attributes:
        
    """

    def __init__(self, DA_metabs, pathways, background_set):
        self.DA_metabs = DA_metabs
        self.pathways = pathways
        self.background_set = background_set
        self.results = None

    def over_representation_analysis(self):

        """
        Function for over representation analysis using Fisher exact test (right tailed)
        Returns:
            DataFrame of ORA results for each pathway, p-value, q-value, hits ratio
        """

        # pathway_names = self.pathways["Pathway_name"].to_dict()

        # Remove pathways not present in the dataset
        compounds_present = self.background_set
        pathways_present = {k: v for k, v in self.pathways.items() if len([i for i in compounds_present if i in v]) > 2}

        pathways_with_compounds = []
        pvalues = []
        pathway_ratio = []
        pathway_coverage = []

        for pathway in pathways_present:
            # perform ORA for each pathway
            pathway_compounds = self.pathways[pathway]
            pathway_compounds = [i for i in pathway_compounds if str(i) != "nan"]
            if not pathway_compounds or len(pathway_compounds) < 2:
                # ignore pathway if contains no compounds or has less than 3 compounds
                continue
            else:
                DA_in_pathway = len(set(self.DA_metabs) & set(pathway_compounds))
                # k: compounds in DA list AND pathway
                DA_not_in_pathway = len(np.setdiff1d(self.DA_metabs, pathway_compounds))
                # K: compounds in DA list not in pathway
                compound_in_pathway_not_DA = len(
                    set(pathway_compounds) & set(np.setdiff1d(self.background_set, self.DA_metabs)))
                # not DEM compounds present in pathway
                compound_not_in_pathway_not_DA = len(
                    np.setdiff1d(np.setdiff1d(self.background_set, self.DA_metabs), pathway_compounds))
                # compounds in background list not present in pathway
                if DA_in_pathway == 0 or (compound_in_pathway_not_DA + DA_in_pathway) < 2:
                    # ignore pathway if there are no DEM compounds in that pathway
                    continue
                else:
                    # Create 2 by 2 contingency table
                    pathway_ratio.append(str(DA_in_pathway) + "/" + str(compound_in_pathway_not_DA + DA_in_pathway))
                    pathway_coverage.append(
                        str(compound_in_pathway_not_DA + DA_in_pathway) + "/" + str(len(pathway_compounds)))
                    pathways_with_compounds.append(pathway)
                    contingency_table = np.array([[DA_in_pathway, compound_in_pathway_not_DA],
                                                  [DA_not_in_pathway, compound_not_in_pathway_not_DA]])
                    # Run right tailed Fisher's exact test
                    oddsratio, pvalue = stats.fisher_exact(contingency_table, alternative="greater")
                    pvalues.append(pvalue)
        try:
            padj = sm.stats.multipletests(pvalues, 0.05, method="fdr_bh")
            results = pd.DataFrame(
                zip(pathways_with_compounds, pathway_ratio, pathway_coverage, pvalues,
                    padj[1]),
                columns=["ID", "Hits", "Coverage", "P-value", "P-adjust"])
            # results["Pathway_name"] = results["ID"].map(pathway_names)
            # results.insert(1, 'Pathway_name', results.pop('Pathway_name'))

        except ZeroDivisionError:
            padj = [1] * len(pvalues)
            results = pd.DataFrame(zip(pathways_with_compounds, pathway_ratio, pvalues, padj),
                                   columns=["ID", "Hits", "Coverage", "P-value", "P-adjust"])
            # results["Pathway_name"] = results["ID"].map(pathway_names)
            # results.insert(1, 'Pathway_name', results.pop('Pathway_name'))

        self.results = results
        return results


def get_metrics(dict_of_PA_res: dict, z_score_df, p_val_col_name: str, id_col_name: str, norm=True, NES_pos=False):
    """Compute the confusion matrix for simPA results as well as other performance metrics

    Args:
        :param NES_pos: True if GSEA NES values are positive for significant pathways (should be used when input
        z-scores are absolute values), False if NES is to be ignored or is not relevant (e.g. ORA)
        :param dict_of_PA_res: dictionary containing PA results DataFrames. Keys represent unique reaction IDs (integer)
         and values are the corresponding PA results for that reaction KO
        :param z_score_df: DataFrame containing z-scores for each pathway
        :param norm: normalise cm by the number of samples (pathways)
        :param p_val_col_name: Name of column containing p-values or corrected p-values
        :param id_col_name: Name of column containing pathway IDs or names
    Returns:
        _type_: DataFrame containing TP, FP, TN, FN

    """
    unique_rxn_id = []
    pathway_associated = []
    # counts
    TP = []
    FP = []
    FN = []
    TN = []
    # normalised values
    TN_pct = []
    FP_pct = []

    recall = []
    precision = []
    F1 = []

    all_subsystems = z_score_df['subsystem'].tolist()
    for k, v in dict_of_PA_res.items():
        unique_rxn_id.append(k)
        subsystem = z_score_df[z_score_df['unique_id'] == k]['subsystem'].values[0]
        pathway_associated.append(subsystem)

        # can only be 1 or 0 - already normed
        if NES_pos:
            TPR = (v[(v[p_val_col_name] <= 0.05) & (v[id_col_name] == subsystem) & (v["NES"] > 0)].shape[0])
            # number of pathways with significant p that are not the target
            FPR = (v[(v[p_val_col_name] <= 0.05) & (v[id_col_name] != subsystem) & (v["NES"] > 0)].shape[0])
        else:
            TPR = (v[(v[p_val_col_name] <= 0.05) & (v[id_col_name] == subsystem)].shape[0])
            FPR = (v[(v[p_val_col_name] <= 0.05) & (v[id_col_name] != subsystem)].shape[0])
        TP.append(TPR)
        FP.append(FPR)
        FP_pct.append(FPR / (len(all_subsystems)))
        # can be 1 or 0 only - already normed
        # can result from p > thresh or ora could not run therefore pathway could not be tested 
        FNR = 1 - TPR
        FN.append(FNR)
        # all pathways affected by KOs in the dataset that were either not tested or have p greater than threshold
        TNR = len(all_subsystems) - (FPR + TPR + FNR)
        TN.append(TNR)
        TN_pct.append(TNR / (len(all_subsystems)))

        try:
            precision.append(TPR / (TPR + FPR))
        except ZeroDivisionError:
            precision.append(np.nan)

        try:
            recall.append(TPR / (TPR + FNR))
        except ZeroDivisionError:
            recall.append(np.nan)

        try:
            F1.append(f1_score(
                y_true=[1 if i == subsystem else 0 for i in v[id_col_name]],
                y_pred=[1 if i <= 0.05 else 0 for i in v[p_val_col_name]]
            ))
        except ZeroDivisionError:
            F1.append(np.nan)

    if norm:
        res_df = pd.DataFrame([unique_rxn_id, pathway_associated, TP, FP_pct, FN, TN_pct, precision, recall, F1],
                              index=['Unique_reaction_id', 'Subsystem name', 'TPR', 'FPR', 'FNR', 'TNR', 'Precision',
                                     'Recall', 'F1']).T
    else:
        res_df = pd.DataFrame([unique_rxn_id, pathway_associated, TP, FP, FN, TN, precision, recall, F1],
                              index=['Unique_reaction_id', 'Subsystem name', 'TPR', 'FPR', 'FNR', 'TNR', 'Precision',
                                     'Recall', 'F1']).T
    return res_df


def confusion_matrix(dict_of_PA_res, z_score_df, norm=True):
    """Compute the confusion matrix for simPA results

    Args:
        dict_of_PA_res (dict): dictionary containing PA results DataFrames. Keys represent unique reqction IDs (integer) and values are the corresponding PA results for that reaction KO
        norm (bool): normlaise cm by the numnber of samples (pathways)
    Returns:
        _type_: DataFrame containing TP, FP, TN, FN
    """
    unique_rxn_id = []
    pathway_associated = []
    # counts
    TP = []
    FP = []
    FN = []
    TN = []
    # normalised values
    TN_pct = []
    FP_pct = []

    all_subsystems = z_score_df['subsystem'].tolist()
    print(len(all_subsystems))
    for k, v in dict_of_PA_res.items():
        unique_rxn_id.append(k)
        subsystem = z_score_df[z_score_df['unique_id'] == k]['subsystem'].values[0]
        pathway_associated.append(subsystem)

        # can only be 1 or 0 - already normed
        TPR = (v[(v['P-value'] <= 0.05) & (v['ID'] == subsystem)].shape[0])
        TP.append(TPR)
        # number of pathways with significant p that are not the target
        FPR = (v[(v['P-value'] <= 0.05) & (v['ID'] != subsystem)].shape[0])
        FP.append(FPR)
        FP_pct.append(FPR / (len(all_subsystems)))
        # can be 1 or 0 only - already normed
        # can result from p > thresh or ora could not run therefore pathway could not be tested
        FNR = 1 - TPR
        FN.append(FNR)
        # all pathways affected by KOs in the dataset that were either not tested or have p greater than threshold
        TNR = (len(all_subsystems)) - (FPR + TPR + FNR)
        TN.append(TNR)
        TN_pct.append(TNR / (len(all_subsystems)))

    if norm:
        res_df = pd.DataFrame([unique_rxn_id, pathway_associated, TP, FP_pct, FN, TN_pct],
                              index=['Unique_reaction_id', 'Subsystem name', 'TPR', 'FPR', 'FNR', 'TNR']).T
    else:
        res_df =  pd.DataFrame([unique_rxn_id, pathway_associated, TP, FP, FN, TN],
                               index=['Unique_reaction_id', 'Subsystem name', 'TPR', 'FPR', 'FNR', 'TNR']).T
    return res_df


def get_metrics_ORA(dict_of_PA_res:dict, z_score_df, p_val_col_name:str, id_col_name:str, norm=True):
    """Compute the confusion matrix for simPA results as well as other performance metrics

    Args:
        dict_of_PA_res (dict): dictionary containing PA results DataFrames. Keys represent unique reqction IDs (integer) and values are the corresponding PA results for that reaction KO
        norm (bool): normlaise cm by the numnber of samples (pathways)
        p_val_col_name: Name of column containing p-values or corrected p-values
        id_col_name: Name of column containing pathway IDs or names
    Returns:
        _type_: DataFrame containing TP, FP, TN, FN
    """
    unique_rxn_id = []
    pathway_associated = []
    # counts
    TP = []
    FP = []
    FN = []
    TN = []
    TN2 = []
    # normalised values
    TN_pct = []
    FP_pct = []

    recall = []
    precision = []
    F1 = []

    all_subsystems = z_score_df['subsystem'].tolist()
    for k, v in dict_of_PA_res.items():

        n_pathways_testable = v.shape[0]
        unique_rxn_id.append(k)
        subsystem = z_score_df[z_score_df['unique_id'] == k]['subsystem'].values[0]
        pathway_associated.append(subsystem)

        # can only be 1 or 0 - already normed
        TPR = (v[(v[p_val_col_name] <= 0.05) & (v[id_col_name] == subsystem)].shape[0])
        TP.append(TPR)

        # number of pathways with significant p that are not the target 
        FPR = (v[(v[p_val_col_name] <= 0.05) & (v[id_col_name] != subsystem)].shape[0])
        FP.append(FPR)
        # normalised by the number of pathways testable by ORA
        try:
            FP_pct.append(FPR/(n_pathways_testable - TPR))
        except ZeroDivisionError:
            FP_pct.append(0)

        # can be 1 or 0 only - already normed
        # can result from p > thresh or ora could not run therefore pathway could not be tested 
        FNR = 1 - TPR
        FN.append(FNR)

        # all pathways affected by KOs in the dataset that were either not tested or have p greater than threshold
        # TNR = (v[(v[p_val_col_name] > 0.05) & (v[id_col_name] != subsystem)].shape[0])
        TNR2 = len(all_subsystems) - (FPR + TPR + FNR)
        # TN.append(TNR)
        TN2.append(TNR2)
        # normalised by the number of pathways testable by ORA
        # TN_pct.append(TNR/n_pathways_testable)

        try:
            precision.append(TPR / (TPR+FPR))
        except ZeroDivisionError:
            precision.append(np.nan)

        try:
            recall.append(TPR / (TPR+FNR))
        except ZeroDivisionError:
            recall.append(np.nan)

        try:
            F1.append(f1_score(
                y_true=[1 if i == subsystem else 0 for i in v[id_col_name]],
                y_pred=[1 if i <= 0.05 else 0 for i in v[p_val_col_name]]
                ))
        except ZeroDivisionError:
            F1.append(np.nan)

    if norm:
        res_df =  pd.DataFrame([unique_rxn_id, pathway_associated, TP, FP_pct, FN, TN_pct, precision, recall, F1],
        index=['Unique_reaction_id', 'Subsystem name', 'TPR', 'FPR', 'FNR', 'TNR', 'Precision', 'Recall', 'F1']).T
    else:
        res_df =  pd.DataFrame([unique_rxn_id, pathway_associated, TP, FP, FN, TN, TN2, precision, recall, F1],
        index=['Unique_reaction_id', 'Subsystem name', 'TPR', 'FPR', 'FNR', 'TNR', 'TNR2', 'Precision', 'Recall', 'F1']).T
    return res_df
