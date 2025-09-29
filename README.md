# ee547-hw2--Rjz0215-  
Name: Jianzhong Ren  
USC Email: jianzhon@usc.edu  
# Brief description of your embedding architecture (Problem 2):  
Pre-processing: Abstracts are lower-cased, cleaned, tokenized, and the top-K frequent words form the vocabulary. Each abstract is represented as a multi-hot bag-of-words vector.  
Model structure: The input BoW vector is passed through a fully connected layer to a hidden layer, followed by a ReLU activation to produce the embedding vector. A second fully connected layer with ReLU maps the embedding back to the hidden layer, and a final fully connected layer with a Sigmoid activation reconstructs the original BoW vector.  
Training: The model is trained with binary cross-entropy loss on BoW reconstruction using the Adam optimizer on CPU, with the total parameter count limited to 2 million.  
Output: The encoder’s latent vector (default 64-dimensional) is saved as the embedding for each paper, together with its reconstruction loss.  
