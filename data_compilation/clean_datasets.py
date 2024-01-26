import numpy as np
import pandas as pd
import json
from datetime import datetime
import types
import pickle
import matplotlib.pyplot as plt

def clean_dataframe(df):
    #prepares dataframe for feature extraction
    #cleaning operations
    #quickly confirm abstracts are lists
    #confirm each list is length 1
    #drop the dfs wtihout abstracts
    df = df[df["abstracts"].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    assert (all(df[df["abstracts"].apply(lambda x: isinstance(x, list))])), "Abstracts do not appear to be inside lists"
    assert (all(df["abstracts"].apply(lambda x: len(x) == 1))), "Abstracts are not length 1 lists. This will fail cleaning"   
    
    #coarse way of converting string to int year
    df["year"] = df["pubdate"].apply(lambda x: int(x.strip()[0:4]))
    df["uid"] = df["uid"].apply(lambda x: str(x))
    #df = df.rename(columns={"uic": "uid"})


    #grab the abstracts
    df["abstracts"] = df["abstracts"].apply(lambda x: x[0] if isinstance(x, list) else x)
    #remove empty abstracts
    df = df[df["abstracts"] != ""].reset_index(drop = True)
    #remove empty lists
    df["cited_by"] = df["cited_by"].apply(lambda x: [] if (x == [0] or x == ["0"]) else x)
    df["citing"] = df["citing"].apply(lambda x: [] if (x == [0] or x == ["0"]) else x)
    return df

def remove_extraneous_citations(citation_list, citations_filter):
    #filter citation_list so it only contains entries in citations_filter
    #this allows us to remove citations that aren't under the mesh term we searched for
    return list(set(citation_list).intersection(set(citations_filter)))

def clean_dataframe_citations(df):
    uids_in_data = df["uid"]
    #remove self from citing and cited by lists
    #remove extraneous citations
    print("Processing citing column...")
    df["citing"] = df["citing"].apply(remove_extraneous_citations, citations_filter = uids_in_data)
    print("Processing cited_by column...")
    df["cited_by"] = df["cited_by"].apply(lambda x: remove_extraneous_citations(x, uids_in_data))
    #df["citing"] = list(map(lambda x: remove_extraneous_citations(uids_in_data, x), df["citing"].tolist()))
    #df["cited_by"] = list(map(lambda x: remove_extraneous_citations(uids_in_data, x), df["cited_by"].tolist()))
    #create some useful numbers
    df["num_citing"] = df["citing"].apply(lambda x: len(x))
    df["num_cited_by"] = df["cited_by"].apply(lambda x: len(x))
    #condition to remove entries that have itself citing and cited by
    #df  = df[(df["citing"] != df["cited_by"])].reset_index(drop = True)
    #this removes any entries that contain itself as the citing and cited by value
    df[df.apply (lambda x: (len(x["citing"]) == 1) & (len(x["cited_by"]) == 1) & 
    (x["citing"] == x["cited_by"]), axis = 1)].reset_index(drop = True)
    print("Citation columns cleaning complete!")
    return df

df1 = pd.read_json("./Datasets/cvd_w_abstracts_2010.json")
df2 = pd.read_json("./Datasets/cvd_w_abstracts_2011.json")
df3 = pd.read_json("./Datasets/cvd_w_abstracts_2012.json")
df4 = pd.read_json("./Datasets/cvd_w_abstracts_2013.json")
df5 = pd.read_json("./Datasets/cvd_w_abstracts_2014.json")
df6 = pd.read_json("./Datasets/cvd_w_abstracts_2015.json")
df7 = pd.read_json("./Datasets/cvd_w_abstracts_2016.json")
df8 = pd.read_json("./Datasets/cvd_w_abstracts_2017.json")
df9 = pd.read_json("./Datasets/cvd_w_abstracts_2018.json")
df10 = pd.read_json("./Datasets/cvd_w_abstracts_2019.json")
df11 = pd.read_json("./Datasets/cvd_w_abstracts_2020.json")
df12 = pd.read_json("./Datasets/cvd_w_abstracts_2021.json")

input_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12], ignore_index = True)
df1 = clean_dataframe(input_df)
df = clean_dataframe_citations(df1)