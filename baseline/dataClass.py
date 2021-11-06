import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer

class prepareData:
    def __init__(self, filename):
        self.data=self.loadData(filename)
        self.X=[]
        self.Y=[]
    
    def loadData(self,filename):
        data=[]
        with open(filename) as f:
            data = json.load(f)
        return data

    def getLength(self):
        return (len(self.X))
        

    def preprocessData(self):
        #extract words in a window
        full_forms=[]
        for i in self.data:
            acro_at=i['acronym']
            tok=i['tokens']
            full_forms.append(i['expansion'])
            n=len(tok)
            low=acro_at-5
            up=acro_at+5
            if low<0:
                low=0
            if up>n:
                up=n
            window=''
            for j in range(low,up):
                window=window+tok[j]+' '
            self.X.append(window)

        label_set=set(full_forms)
        n=len(label_set)
        l=list(label_set)

        for a in full_forms:
            for i in range(n):
                if l[i]==a:
                    self.Y.append(i)
    
    def convertVectors(self):
        vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=100)
        vectorizer.fit(self.X)
        self.X=vectorizer.transform(self.X).toarray()

    def convertGloVe(self):
      GloveVecs=[]
      for sent in self.X:
        sent_vec=np.array([0.0 for i in range(100)])
        c=0
        l=[]
        for word in sent:
          try:
            sent_vec= np.add(sent_vec,np.array(model.get_vector(word))) # add word vectors
            c=c+1
          except:
            pass 
        sent_vec=sent_vec/c # compute average
        GloveVecs.append(sent_vec)
      self.X=GloveVecs
