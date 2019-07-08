
def jaccard_similarity(x,y):
    """ returns the jaccard similarity between two lists """
    sim=0
    if x and y:
      intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
      union_cardinality = len(set.union(*[set(x), set(y)]))
      #print(intersection_cardinality)
      #print(union_cardinality)
      sim=intersection_cardinality/float(union_cardinality)
    return sim