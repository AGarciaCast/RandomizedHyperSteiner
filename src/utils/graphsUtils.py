# Python program for Kruskal's algorithm to find 
# Minimum Spanning Tree of a given connected, 
# undirected and weighted graph 
  
# Modified from: https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/


class DisjSet:
    def __init__(self, n=0, numSteiner=0, idxVert=None):
        # Constructor to create and
        # initialize sets of n items
       
        self.parent = dict()
        self.rank = dict()
        
        if idxVert is None:
            for i in range(n):
                self.parent[f"T{i}"] = f"T{i}"
                self.rank[f"T{i}"] = 1
            

            for i in range(numSteiner):
                self.parent[f"S{i}"] = f"S{i}"
                self.rank[f"S{i}"] = 1
        else:
             for idx in idxVert:
                self.parent[idx] = idx
                self.rank[idx] = 1
    
    def numConnectedComponents(self):
        return sum([k==v for k,v in self.parent.items()])
    
    
    # Finds set of given item x
    def find(self, x):
        try:    
            # Finds the representative of the set
            # that x is an element of
            if (self.parent[x] != x):
                    
                # if x is not the parent of itself
                # Then x is not the representative of
                # its set,
                self.parent[x] = self.find(self.parent[x])
                    
                # so we recursively call Find on its parent
                # and move i's node directly under the
                # representative of this set

            return self.parent[x]
        
        except KeyError:
            self.parent[x] = x
            self.rank[x] = 1
            
            return x
            


    # Do union of two sets represented
    # by x and y.
    def union(self, x, y):
            
        # Find current sets of x and y
        xset = self.find(x)
        yset = self.find(y)

        # If they are already in same set
        if xset == yset:
            return

        # Put smaller ranked item under
        # bigger ranked item if ranks are
        # different
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset

        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset

        # If ranks are same, then move y under
        # x (doesn't matter which one goes where)
        # and increment rank of x's tree
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1



  
    
  
  
