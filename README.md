# Identication of Bacteriophage Protein Locations using Evolutionary and Structural Features
Bacteriophage, a virus that can kill a bacterium by infection and replication. Using this characteristic bacteriophage proteins can be used to discover potential antibacterial drugs. So it is very important to know the functioning of Bacteriophage in the host bacteria. Identifying the location of bacteriophage protein in a host cell is very important to understand their working mechanism. In this paper, a model is proposed to predict the presence of bacteriophage and also its location in the host cell if it is present in host cell. We used deep neural network classier to train our model. For reducing number of features for eective prediction Recursive Feature Elimination (RFE) was used. Using jackknife cross validation our model can predict whether bacteriophages are in the host cell or not with a maximum accuracy of 87.7% and can further discriminate the sub-cellular location with a maximum accuracy of 98.5%. This model outperforms the state-of-the-art predictor on standard dataset using standard evaluation criteria. Our method is readily available to use as a standalone tool from: Index Terms—Bacteriophages, proteins, locations, phages, classication, feature selection


## DataSets

### small dataset
>dataSub.csv 

### big dataset

> final.csv



## DNN Codes

### KFold_final 
> Kfold With Anova (99 features)


### Jack_knife_final 
> Jack Knife with RFE (99 features)

