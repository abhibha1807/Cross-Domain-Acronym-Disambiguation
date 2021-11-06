import numpy as np
import json

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
