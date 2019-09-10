
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import networkx as nx
import numpy as np
import numpy.matlib


from nltk import pos_tag, word_tokenize


####TEXT FUNCTIONS


def wnid2lemma(strwnid):
    split_strwnid=strwnid.split('n')
    wnid=int(split_strwnid[1])
    ss=wn._synset_from_pos_and_offset('n',wnid)
    lemma=ss.lemmas()[0].name()
    return lemma

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
'''
sentence = "I am going to buy some gifts"
tagged = pos_tag(word_tokenize(sentence))

synsets = []
lemmatzr = WordNetLemmatizer()

for token in tagged:
    wn_tag = penn_to_wn(token[1])
    if not wn_tag:
        continue

    lemma = lemmatzr.lemmatize(token[0], pos=wn_tag)
    synsets.append(wn.synsets(lemma, pos=wn_tag)[0])

print(synsets)

offsets=[]
for idx,ss in enumerate(synsets):
    offsets.append(wn.ss2of(synsets[idx]))
    
print(offsets)

'''
####PROCEDURE FUNCTIONS

def keywordsimilarities(keyword,keywords):
    similarities=np.zeros(len(keywords))
    for idx2,keyword2 in enumerate(keywords):
        #t1=nltk.pos_tag(nltk.word_tokenize(keyword))[0][1] #word type
        #t2=nltk.pos_tag(nltk.word_tokenize(keyword2))[0][1] #word type
        keyword=WordNetLemmatizer().lemmatize(keyword, pos="n")
        keyword2=WordNetLemmatizer().lemmatize(keyword2, pos="n")
        similarity=computeWordSimilarity(keyword,keyword2)
        #if t1=="NN" #if word type is a noun...
        similarities[idx2]=similarity
    return similarities

def keywordsetsimilarities(keywords1,keywords2):
    similarities=np.zeros(shape=(len(keywords1),len(keywords2)))
    for idx1,keyword1 in enumerate(keywords1):
         for idx2,keyword2 in enumerate(keywords2):
             t1=nltk.pos_tag(nltk.word_tokenize(keyword1))[0][1] #word type
             t2=nltk.pos_tag(nltk.word_tokenize(keyword2))[0][1] #word type
             keyword1=WordNetLemmatizer().lemmatize(keyword1, pos="n")
             keyword2=WordNetLemmatizer().lemmatize(keyword2, pos="n")
             similarity=computeWordSimilarity(keyword2,keyword1)
             #if t1=="NN" #if word type is a noun...
             similarities[idx1][idx2]=similarity
    return similarities
             
def categories2symweigths(keywords,categories): #learning
     similarities=keywordsetsimilarities(keywords,categories)
     weights=similarities.max(0)
     return weights
    
def category2symweigths(keyword,categories): #learning
    keywords=[keyword for i in range(len(categories))]
    return categories2symweigths(keywords,categories)

def keyword2label(keyword,categories):
    keywords=[keyword for i in range(len(categories))]
    weights=categories2symweigths(keywords,categories)
    order=np.argsort(weights)[::-1]
    return categories[order[0]]
    
def text2graph(text): #parsing
     keywords=nltk.word_tokenize(text)
     #for idx,keyword in enumerate(keywords):    
     G = nx.MultiDiGraph()
     try:
         for idx,keyword in enumerate(keywords):    
             synset=wn.synsets(keyword)
             subG=closure_graph(synset[0], lambda s:  s.hypernyms())
             G.add_node(subG)
         return keywords,G
     except:    
         return keywords,G


####SYMBOLIC LEARNING REPRESENTATIONS
        
def computeWordSimilarity(word1,word2): 
        s1=wn.synsets(word1)
        s2=wn.synsets(word2)
        try:
            sim=wn.wup_similarity(s1[0],s2[0])
        except IndexError:
            sim=0
                
        return sim

####SYMBOLIC REPRESENTATIONS
        

def closure_graph(synset, fn):
    seen = set()
    graph = nx.DiGraph()

    def recurse(s):
        #print(s)
        if not s in seen:
            seen.add(s)
            graph.add_node(s.name())
            for s1 in fn(s):
                graph.add_node(s1.name())
                graph.add_edge(s.name(), s1.name())
                recurse(s1)

    recurse(synset)
    return graph

def represent_graph(G):
    index = nx.betweenness_centrality(G)
    plt.rc('figure', figsize=(12, 7))
    node_size = [index[n]*1000 for n in G]
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, node_size=node_size, edge_color='r', alpha=.3, linewidths=0)
    plt.show()


'''
#example1
sentence ="I love yellow peppers."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)

print(tagged)
print(entities)
#example2
a1=wn.synsets("car")
a2=wn.synsets("motor")
print(wn.wup_similarity(a1[0],a2[0]))
print(a1[0].lowest_common_hypernyms(a2[0]))

a3=wn.synsets("red")[0]
G=closure_graph(a3, lambda s:  s.hypernyms())
represent_graph(G)


'''