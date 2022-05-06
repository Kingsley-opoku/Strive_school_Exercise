import numpy as np

class GetFeature:
    
    def __init__(self, data, target_name, seq_len=6) :
        self.data=data
        self.target_name=target_name
        self.seq_len=seq_len
        self.seq, self.target=self.get_sequence()
        
    
    

    def get_sequence(self)-> np.array:

        seq_list = []
        target_list = []
        for i in range(0, self.data.shape[0] - (self.seq_len+1), self.seq_len+1):

            seq = self.data[i: self.seq_len + i]
            target = self.data[self.target_name][self.seq_len + i]
            target_list.append(target)
            seq_list.append(seq)
        
        return np.array(seq_list), np.array(target_list)

    
    # def get_target(self)-> np.array:
        
    #     target_list = []

    #     for i in range(0, self.data.shape[0] - (self.seq_len+1), self.seq_len+1):
    #         target = self.data[self.target_name][self.seq_len + i]
    #         target_list.append(target)

    #     return np.array(target_list)
        

    def get_feat_all_mean(self) -> np.array:
        #seq_list=self.get_sequence()
        
        features=[]
        for i in range(self.seq.shape[0]):
            meanx = self.seq[i].mean(axis=0)
            features.append(meanx)
        return np.array(features)

        
    def get_feat_all_std(self) -> np.array:
        #seq_list=self.get_sequence()
        
        features=[]
        for i in range(self.seq.shape[0]):
            meanx = self.seq[i].std(axis=0)
            features.append(meanx)
        return np.array(features)

    
    def get_feat_std_mean(self):
        
        feature=[]

        for i in range(self.seq.shape[0]):
            feat_sample=[]

            for j in range(self.seq.shape[2]):
                if j<=6:
                    feat_sample.append(np.mean(self.seq[i][:,j]))
                else:
                    feat_sample.append(np.std(self.seq[i][:,j]))
        
              
            feature.append(feat_sample)
        
        return np.array(feature)

    
    
    def feat_stat_each_col(self):
        
        feature=[]

        for i in range(self.seq.shape[0]):
            feat_sample=[]

            for j in range(self.seq.shape[2]):
                if j==0 or j==3:
                    feat_sample.append(np.median(self.seq[i][:,j]))
                elif j==1 or j ==5:
                    feat_sample.append(np.max(self.seq[i][:,j])-np.min(self.seq[i][:,j]))
                elif j==1 or j==10:
                    feat_sample.append(np.std(self.seq[i][:,j]))
                else: 
                    feat_sample.append(np.mean(self.seq[i][:,j]))
                

            feature.append(feat_sample)
        return np.array(feature)


    def feat_add_statistic():
        pass
        