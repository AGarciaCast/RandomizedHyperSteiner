import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#

MODES_SCALABILITY = ["4+Simple"]#["3+Simple", "3+Precise", "4+Simple", "4+Precise"]
MODES_CONVERGENCE = ["3+1", "3+n", "4+1", "4+n"]
COLUMNS_TO_DROP = ["correct", "ratio"]
MEAN_STD_FORMAT = r"${:.2f} \pm {:.2f}$"
MEAN_FORMAT = r"${:.2f}$"


def read_df(title, path):
    aux = f'results_{title}.tsv'

    full_df = pd.read_csv(
    path / aux,
    sep="\t"
    )
    return full_df


def rearrange_table_scalability(mydf, modes):
    resDf = None
    for mode in modes:
        extractedDF = mydf[mydf[("mode")] == mode]
        colFilter =  [("mode",  ""),
                    ("numPoints",  ""),
                    (mode, "avgTime"),
                    (mode, "perImprov"),
                    (mode, "avgFST3")
                    ]

        if mode[0]=="4" or mode=="Euclidean":
            colFilter.append((mode, "avgFST4"))
        else:
            extractedDF = extractedDF.drop(columns=["avgFST4"])


        extractedDF.columns = pd.MultiIndex.from_tuples(colFilter)
        extractedDF.reset_index(drop=True, inplace=True)
        if resDf is None:
            resDf = extractedDF.drop(columns=["mode"])
        else:
            resDf = pd.merge(resDf,
                        extractedDF.drop(columns=["mode", "numPoints"]),
                        left_index=True, right_index=True
                        )

    
    
    return resDf



def rearrange_table_convergence(mydf, maxVal=4):
    auxDf = None
    for mode in [f"{i}+{p}" for p in["1", "n"] for i in range(3, maxVal+1) ]:
        extractedDF = mydf[mydf[("mode")] == mode]
        extractedDF[mode] = extractedDF["perImprov"]
    
        extractedDF.reset_index(drop=True, inplace=True)

        if auxDf is None:
            auxDf = extractedDF.drop(columns=["mode", "perImprov"])
        else:
            auxDf = pd.merge(auxDf,
                        extractedDF.drop(columns=["mode", "paramCurves", "perImprov"]),
                        left_index=True, right_index=True
                        )

    resDf = None
    
    for k in range(3, maxVal + 1):
        extractedDF = auxDf[["paramCurves", f"{k}+1", f"{k}+n"]]
        colFilter =  [("paramCurves",  ""),
                        (f"k={k}", f"{k}+1"),
                        (f"k={k}", f"{k}+n"),
                        ]
        
        extractedDF.columns = pd.MultiIndex.from_tuples(colFilter)

        if resDf is None:
            resDf = extractedDF
        else:
            resDf = pd.merge(resDf,
                        extractedDF.drop(columns=["paramCurves"]),
                        left_index=True, right_index=True
                        )    

        
    
    return resDf


def to_latex(df):
    return df.to_latex(
                    escape=False,
                    multirow=True,
                    sparsify=True,
                    multicolumn_format="c",
                    )



def process_df_scalability(path, title):

    raw_df = read_df(path=path, title=title)
    raw_df = raw_df.drop(columns=COLUMNS_TO_DROP )
    raw_df["perImprov"] = raw_df["perImprov"] * 100

    modes = MODES_SCALABILITY
    print(title)
    # if "CenteredGauss" in title:
    #     modes = ["Euclidean"] + modes
    #     print("hola")
    
    full_df = rearrange_table_scalability(raw_df, modes)

    for mode in modes:
    
        print(mode,
              round(np.mean(full_df[(mode, "perImprov")]), 2),
              round(np.std(full_df[(mode, "perImprov")]), 2))
    
    
    orderCol = [("numPoints", "")]
    
    
    for mode in modes:
        orderCol.append((mode, "avgFST3"))
        if mode[0]=="4" or mode == "Euclidean":
            orderCol.append((mode, "avgFST4"))
            
        orderCol.append((mode, "perImprov"))
        orderCol.append((mode, "avgTime"))    

    full_df = full_df[orderCol]

    df = (
        full_df.groupby(
        ["numPoints"],
        )
        .agg([np.mean, np.std])
        .round(2)
        )

    df_res = df.copy()
    df.index.names = [r'$n$']
    for mode in modes:
        for metric, new_name in [("avgFST3", "\# 3 pts"), ("avgFST4", "\# 4 pts"), ("perImprov", "RED"), ("avgTime", "CPU")]:
            if metric == "avgFST4" and mode[0] == "3":
                continue

            if metric=="perImprov":
                df[(mode, new_name, "")] = df.apply(
                    lambda row: MEAN_STD_FORMAT.format(
                        row[(mode, metric, "mean")], row[(mode, metric, "std")]
                    ),
                    axis=1,
                )
            else:
                df[(mode, new_name, "")] =  df.apply(
                    lambda row: MEAN_FORMAT.format(
                        row[(mode, metric, "mean")],
                    ),
                    axis=1,
                ) 
        
            for agg in ("mean", "std"):
                df = df.drop(columns=[(mode, metric, agg)])

    tex_table = to_latex(df)
    if "CenteredGauss" in title:
        header = '\\begin{table*}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lccccccccccccccc}\n\\toprule\n{} & \\multicolumn{3}{c}{Euclidean} & \\multicolumn{3}{c}{3+Simple} & \\multicolumn{3}{c}{3+Precise} & \\multicolumn{4}{c}{4+Simple} & \\multicolumn{4}{c}{4+Precise} \\\\\n \\cmidrule(lr){2-4} \n \\cmidrule(lr){5-7} \n \\cmidrule(lr){8-11} \n \\cmidrule(lr){12-15} \n \\cmidrule(lr){16-19} \n $n$ & \\# 3 pts & RED & CPU & \\# 3 pts & RED & CPU & \\# 3 pts & RED &  CPU & \\# 3 pts & \\# 4 pts & RED & CPU & \\# 3 pts & \\# 4 pts & RED & CPU \\\\\n'
    else:
        header = '\\begin{table*}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lcccccccccccccc}\n\\toprule\n{} & \\multicolumn{3}{c}{3+Simple} & \\multicolumn{3}{c}{3+Precise} & \\multicolumn{4}{c}{4+Simple} & \\multicolumn{4}{c}{4+Precise} \\\\\n \\cmidrule(lr){2-4} \n \\cmidrule(lr){5-7} \n \\cmidrule(lr){8-11} \n \\cmidrule(lr){12-15} \n $n$ & \\# 3 pts & RED & CPU & \\# 3 pts & RED &  CPU & \\# 3 pts & \\# 4 pts & RED & CPU & \\# 3 pts & \\# 4 pts & RED & CPU \\\\\n'

    caption = "Scalability analysis w/" + title + ". \\#3: number of 3-point local solutions used. \\#4: number of 4-point local solutions used. RED: reduction over MST (\%). CPU: total CPU time (sec.). "
    bottom = "}\n\\caption{" +\
                caption +\
                "}\n\\label{" +\
                f'tab:scalability_analysis' +\
                "}\n\\end{table*}"

    tex_res = "".join([header, tex_table[tex_table.find('\\midrule'):], bottom])

    print(tex_res)

    return df_res, tex_res


def process_df_convergence(path, title="Conv2Ideal", maxVal=4):

    raw_df = read_df(path=path, title=title)
    raw_df = raw_df.drop(columns=COLUMNS_TO_DROP + ["avgTime", "avgFST3", "avgFST4"])
    raw_df["perImprov"] = raw_df["perImprov"] * 100

    full_df = rearrange_table_convergence(raw_df, maxVal)
    
    df = (
        full_df.groupby(
        ["paramCurves"],
        )
        .agg([np.mean, np.std])
        .round(2)
        )

    df_res = df.copy()
    
    df.index.names = [r'$t$']
    for mode in [f"k={i}" for i in range(3, maxVal+1)]:
        for metric, new_name in [(f"{mode[2]}+1", "One pt."), (f"{mode[2]}+n", "Cluster")]:

            df[(f"${mode}$", new_name, "")] = df.apply(
                lambda row: MEAN_STD_FORMAT.format(
                    row[(mode, metric, "mean")], row[(mode, metric, "std")]
                ),
                axis=1,
            )
       
        
            for agg in ("mean", "std"):
                df = df.drop(columns=[(mode, metric, agg)])

    tex_table = to_latex(df)
    print(tex_table)
    
    
    
    return df_res, tex_table    

if __name__ == '__main__':

    process_df_convergence(Path("./Results/"), title="Conv2Ideal")
