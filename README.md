# MOGONET  (Multi-Omics Graph cOnvolutional NETworks)
MOGONET integrates multi-omics data using graph convolutional networks
## Fig: Illustration of MOGONET.
![image](https://user-images.githubusercontent.com/93058160/214865396-c19cc08b-8396-4cec-b2f4-ce02b3f933bc.png)

MOGONET combines GCN for multi-omics-specific learning and VCDN for multi-omics integration. For clear and concise illustration, an example of one sample is chosen to demonstrate the VCDN component for multi-omics integration. Preprocessing is first performed on each omics data type to remove noise and redundant features. Each omics-specific GCN is trained to perform class prediction using omics features and the corresponding sample similarity network generated from the omics data. The cross-omics discovery tensor is calculated from the initial predictions of omics-specific GCNs and forwarded to VCDN for final prediction. MOGONET is an end-to-end model and all networks are trained jointly.
Here is the original ![article](https://www.nature.com/articles/s41467-021-23774-w) et ![github](https://github.com/txWang/MOGONET). 
