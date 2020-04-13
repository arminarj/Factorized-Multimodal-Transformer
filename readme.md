# Code for Factorized Multimodal Transformer

This is a transformer encoder model where there are multiple attention groups in each encoder layer.
An attention group focuses on a unique combination of modalities, ranging from 1 modality to all 
three modalities. Each attention group share the number of heads. There is a unique set of 
convolutions for each head.
