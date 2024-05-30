library(rentrez)
library(tidyverse)
library(stringr)

setwd("~/Desktop/")

#pull up docs
#vignette("rentrez_tutorial")

##We'll test retrieval of COPD articles from just 2018, to get a sense of how many come out each year


#investigate properties of mesh temr
#copd_mesh_term = entrez_search(db = "mesh", term = "COPD")
# https://www.ncbi.nlm.nih.gov/mesh/68029424


#attempting search for COPD articles in Pubmed Central between 2003, 2018
search_results = entrez_search(db = "pmc", term = "cardiovascular disease[MeSH] AND 2009:2023[PDAT]")
#Note, more functionality for search can be found at
#https://www.ncbi.nlm.nih.gov/books/NBK25499/#_chapter4_ESearch_


#Looks like there are ~10,000 articles published on COPD between 2003 
#(when the MeSH term was defintely in use, it was defined in 2002) and 2018. 
#This entire publishing base is small enough for us to test on!


#We'll begin first only with all IDs in 2017. Then, we'll use 2018, and attempt to create co-citation
#relations between all the papers retrieved. 


#publishing data equal to 2017, and COPD term. There are only 1176 articles
search_results_2017 = entrez_search(db = "pmc", term = "COPD[MESH] AND 2009:2023[PDAT]", retmax = 2000)


##For each article in search results, find references list and create vector of them

#Note, language is not being filtered here. This will pose a problem for conventional text analytics
#We'll have to clean these later, or specify a language in the search

#first_article = search_results_2017$ids[2]
#first_article_summary = entrez_summary(db = "pmc", id = first_article)
#first_article_title = first_article_summary$title
#first_article_title


#Retrieving references

#first_article_links =  entrez_link(dbfrom = "pmc", id = first_article, db = "pmc")

#some articles contain cited by, and cited information. some don't, which is probably
#because they are not cited! We'll need to handle this if it occurs. 

#Turns out R returns NULL if the accessible value is unavailable. Perfect!

#first_article_references = first_article_links$links

#the following are vectors of the articles this article cites, and the articles citing this article
#first_article_cites = first_article_references$pmc_pmc_cites
#first_article_cited_by = first_article_references$pmc_pmc_citedby


##Todo

#Get API access to increase query count


retrieve_papers = function(search_term, batch_size = 50){
  ###Retrieve article given search term, and time period
  #Currently running into issues regarding web history, request size, etc
  search_results = entrez_search(db = "pmc", term = search_term, use_history = T, retmax = 2000)
  search_ids = search_results$ids
  #article_ids = search_results$ids
  art_vect = c()
  for(x in seq(1 ,search_results$count, 50)){
    article_summaries = entrez_summary(db = "pmc",  web_history = search_results$web_history, 
                                       retstart = x, retmax = 50, warn = F)
    articles = unlist(extract_from_esummary(article_summaries, elements = c("uid", "pubdate", "epubdate", "title")))
    # print(class(articles))
    art_vect = c(art_vect, articles)
  }
  art_mat = matrix(art_vect, ncol = 4, byrow = T)
  df = as.data.frame(art_mat, stringsAsFactors = FALSE)
  colnames(df) =c("uid", "pubdate", "epubdate", "title")
  return(df)
}



retrieve_citations = function(df){
  #takes papers retrieved from PMC and attaches their cited and citing information to them
  #returns a dataframe with this information
  cited_by_dat = list()
  citing_dat = list()
  ids_vect = df$uid
  for(x in seq(1, length(ids_vect), 50)){
    #go by batches, or get the rest
    #print(x)
    if(x + 49 < length(ids_vect)){
      desired_seq = c(x:(x+49))
      #print(desired_seq)
      ids = ids_vect[desired_seq]
    }
    else{
      desired_seq = c(x:length(ids_vect))
      #print(desired_seq)
      ids = ids_vect[desired_seq]
    }
    citation_links = entrez_link(dbfrom = "pmc", db = "pmc", id = ids, by_id = T)
    temp_cited_by_list = lapply(citation_links, function(x) x$links$pmc_pmc_citedby)
    temp_citing_list = lapply(citation_links, function(x) x$links$pmc_pmc_cites)
    cited_by_dat = c(cited_by_dat, temp_cited_by_list)
    citing_dat = c(citing_dat, temp_citing_list)
    #for the batched set of citations, retrieve the pmc_pmc_citedby and pmc_pmc_cited
    #figure out an efficient way to do it, while also preserving order....
  }
  
  
  df$cited_by = cited_by_dat
  df$citing = citing_dat
  return(df)
}



#editing search term to include cc license so we can grab abtstracts using tidy pmc
#should I try "open access[FILT]"? vs "cc license FILT ?
#https://www.ncbi.nlm.nih.gov/books/NBK3825/#pmchelp.Abstract_AB

#search_term = "COPD[MESH] AND 2017[PDAT] AND cc license[FILT]"
#search_term = "COPD[MESH] AND 2017[PDAT] AND has abstract[FILT]"
search_term = "COPD[MESH] AND 2005:2009[PDAT] AND open access[FILT] AND has abstract[FILT]"
#search_term = "Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT] AND 2019[PDAT]"
#search_term = "Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT] AND 2018[PDAT]"
#search_term = "Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT] AND 2017[PDAT]"
#search_term = "Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT] AND 2016[PDAT]"
#search_term = "Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT] AND 2014[PDAT]"
#search_term = "Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT] AND 2013[PDAT]"
#search_term = "Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT] AND 2020[PDAT]"

#note, coronary artery disease introduced in 2008
#search_term = "HIV[MESH] AND coronary artery disease[MESH] AND open access[FILT]"
#search_term = "HIV[MESH] AND Cardiovascular Diseases[MESH] AND open access[FILT] AND has abstract[FILT]"

#search_results = entrez_search(db = "pmc", term = "HIV[MESH] AND coronary artery disease[MESH]")

start_time = Sys.time()
papers_df  = retrieve_papers(search_term, batch_size = 50)



citations_df = retrieve_citations(papers_df)
library(jsonlite)

#dataframe construction complete
end_time = Sys.time()

print("Total time for paper retrieval is: ")
print(end_time - start_time)

print("Length of dataset...")
print(nrow(citations_df))

# Integrity checking
## Randomly select ids from dataframe, and individually check if citing and cited references are correct
## Return titles, etc for confirmation


desired_uids = citations_df$uid
#perfectly executing code below, for one instance
library(stringr)
library(tidypmc)
#raw_xml = pmc_xml(paste("PMC", desired_uids[719], sep = ""))
#txt = pmc_text(raw_xml) %>% filter(section == "Abstract") %>% select(section, text)
#abstract = str_c(txt$text, collapse = ",")


#Write function that applies the above function to each id along desired_uids


get_abstract = function(uid){
  #given a UID, retrieves abstract using tidypmc
  raw_xml = pmc_xml(paste("PMC", uid, sep = ""))
  txt = pmc_text(raw_xml) %>% filter(section == "Abstract") %>% select(section, text)
  abstract = str_c(txt$text, collapse = ",")
  return(abstract)
}

start_time_abs = Sys.time()
scrape_abstract = function(uid){
  abstract = tryCatch({
     return(get_abstract(uid))
  }, error = function(e){
    return("")
  })
}

library(parallel)
cores <- detectCores()

abstract_vect = mclapply(citations_df$uid, scrape_abstract, mc.cores = cores)


#abstract_vect = lapply(citations_df$uid[1:10], scrape_abstract)
end_time_abs = Sys.time()
print("Abstract request time is: ")
print(end_time_abs  - start_time_abs)

#scrape_abstract("6890344")





citations_df$abstracts = abstract_vect

#write_json(citations_df, path = "hiv_cvd_citations.json")
#df.m[is.na(df.m)] <- 0

citations_df$cited_by[(citations_df$cited_by == "NULL")] = 0
print(citations_df)

#write_json(citations_df, path = "cvd_w_abstracts_2020.json")
write_json(citations_df, path = "05_09_copd_w_abstracts.json")
##The code for this function is pretty simple, and I could probably speed it up by generating the links directly
##Using an API key, we can make up to 10 requests a second
#I'm worried that this API will not respect this at all.




#abstract_vect = c()
#for(x in 1:length(desired_uids)){
#  print(x)
#  abstract_vect = tryCatch({
#    c(abstract_vect, get_abstract(desired_uids[x]))
#  }, error = function(e){
#    c(abstract_vect, "")
#  })
#}