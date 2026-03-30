from src.smithMethods import sslMethod
from src.exhaustiveMethods_local import exhaustiveMethod_local 
from src.exhaustiveMethods_global import exhaustiveMethod_global 

# Wrapper of the two methods
def heuristicSteinerDT(points, method="SLL", space="Klein", triangMeth="DT", maxgroup=4,
                        nIters=100, convDiff=1e-2, dist2Points=1e-1, precise=True, 
                        extendedResults=False, selection=None,
                        nMaxExpansions=6, slack=1,  threshold_improvement=0.0, num_epochs=200, lr=1e-2, 
                        early_stopping=True, patience=100, min_delta=1e-6, 
                          expansion_mode="sqrt"):
    
    if method == "SLL":
        selection = ("MST", (1.0, 1.0)) if selection is None else selection
        result = sslMethod(points, 
                           space=space,
                           triangMeth=triangMeth, 
                           maxgroup=maxgroup,
                           nIters=nIters,
                           convDiff=convDiff,
                           dist2Points=dist2Points,
                           precise=precise,
                           extendedResults=extendedResults,
                           selection=selection,
                           )
    
    elif method == "EXH_LOC":
        selection = ("PROB", (0.3, 0.6)) if selection is None else selection                 
        result = exhaustiveMethod_local(points, 
                           space=space,
                           triangMeth=triangMeth, 
                           maxgroup=maxgroup,
                           nIters=nIters,
                           convDiff=convDiff,
                           dist2Points=dist2Points,
                           precise=precise,
                           extendedResults=extendedResults,
                           selection=selection,
                           nMaxExpansions=nMaxExpansions,
                           slack=slack
                           )        
    elif method == "EXH_GLOB" or method == "EXH":
        selection = ("PROB", (0.3, 0.6)) if selection is None else selection                 
        result = exhaustiveMethod_global(points, 
                           space=space,
                           triangMeth=triangMeth, 
                           dist2Points=dist2Points,
                           precise=precise,
                           extendedResults=extendedResults,
                           selection=selection,
                           nMaxExpansions=nMaxExpansions,
                           slack=slack,
                           threshold_improvement=threshold_improvement,
                           num_epochs=num_epochs,
                           lr=lr, 
                           maxgroup=maxgroup,
                           nIters=nIters,
                           convDiff=convDiff,
                           early_stopping=early_stopping, patience=patience, min_delta=min_delta, 
                           expansion_mode=expansion_mode
                           )
        
      
    else:
        raise ValueError("'method' should be either 'SLL' or 'EXH_LOC' or 'EXH_GLOB'")       
     
    return result
