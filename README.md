# HiC-GNN: A Generalizable Model for 3D Chromosome Reconstruction Using Graph Convolutional Neural Networks
------------------------------------------------------------------------------------------------------------------------------------
**OluwadareLab,**
**University of Colorado, Colorado Springs**

----------------------------------------------------------------------
**Developers:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Van Hovenga<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Mathematics <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: vhovenga@uccs.edu <br /><br />

**Contact:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oluwatosin Oluwadare, PhD <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: ooluwada@uccs.edu 
 
 --------------------------------------------------------------------	
## **Description of HiC Data**:
--------------------------------------------------------------------	
## **Build Instructions:**
HiC-GNN runs in a Docker-containerized environment. Before cloning this repository and attempting to build, install the [Docker engine](https://docs.docker.com/engine/install/). To install and build HiC-GNN follow these steps.

1. Clone this repository locally using the command ``git clone https://github.com/OluwadareLab/HiC-GNN.git && cd HiC-GNN``. 
2. Pull the HiC-GNN docker image from docker hub using the command ``docker pull oluwadarelab/hicgnn:latest``. This may take a few minutes. Once finished, check that the image was sucessfully pulled using ``docker image ls``.
3. Run the HiC-GNN container and mount the present working directory to the container using ``docker run --rm -it --name hicgnn_cont -v ${PWD}:/HiC-GNN oluwadarelab/hicgnn``. 

--------------------------------------------------------------------	
## **Content of Folders:**
- Data: This folder contains two Hi-C contact maps in the coordinate list format for chromosome 19 of the GM12878 cell line- one at 1mb resolution and one at 500kb resolution. 

--------------------------------------------------------------------	
## **Scripts**

There are three python scripts used in this study. We describe their purposes and usage below.

### HiC-GNN_main.py
This script takes a single Hi-C contact map as an input and utilizes it to train a HiC-GNN model. 

**Inputs**: 
1. A Hi-C contact map in either matrix format or coordinate list format.

**Outputs**: 
1. A .pdb file of the predicted 3D structure corresponding to the input file in ```Outputs/input_filename_structure.pdb```.
2. A .txt file depicting the optimal conversion value, the dSCC value of the output structure, and final MSE loss of the trained model in ```Outputs/input_filename_log.txt```.
3. A .pt file of the trained model weights corresponding to the input file in ```Outputs/input_filename_weights.pt```
4. A .txt of the normalized Hi-C contact map corresponding to the KR normalization of the input file in ```Data/input_filename_matrix_KR_normed.txt```.
5. A .txt file of the embeddings corresponding to the input file in ```Data/input_filename_embeddings.txt```. 
6. (In the case that the input file was in list format) A .txt file of the input file in matrix format in ```Data/input_filename_matrix.txt```.

**Usage**: ```python HiC-GNN_main.py input_filepath```

* **positional arguments**: <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath```: Path of the input file. <br />

* **optional arguments**: <br />	
	&nbsp;&nbsp;&nbsp;&nbsp;-h, --help  show this help message and exit<br />
	&nbsp;&nbsp;&nbsp;&nbsp;-c, --conversions <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;String of conversion constants of the form '[lowest, interval, highest]' for a set of equally spaced conversion factors, or of the form '[conversion]' for a single conversion factor. Default value: '[.1,.1,2]' <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-bs, --batchsize <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Batch size for embeddings generation. Default value: 128. <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-ep, --epochs <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of epochs used for embeddings generation. Default value: 10. <br />	
	&nbsp;&nbsp;&nbsp;&nbsp;-lr, --learningrate <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learning rate for training GCNN. Default value: .001. <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-th, --threshold <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loss threshold for training termination. Default value: 1e-8. <br />
    
* **Example**: ```python HiC-GNN_main.py Data/GM12878_1mb_chr19_list.txt```

### HiC-GNN_generalize.py
This script takes in two Hi-C maps in coordinate list format. The script generates embeddings for the first input map and then trains a model using the map and the corresponding embeddings. The script then generates embeddings for the second input map and aligns these embeddings to those of the first input map and tests the model generated from the first input using these aligned embeddings. The output is a structure corresponding to the second input generalized from the model trained on the first input. The script searches for files corresponding to the raw matrix format, the normalized matrix format, the embeddings, and a trained model for the inputs in the current working directory. For example, if the input file is ```input.txt```, then the script checks if ```Data/input_matrix.txt```, ```Data/input_matrix_KR_normed.txt```, and ```Data/input_embeddings.txt``` exists. If these files do not exist, then the script generates them automatically.

**Inputs**: 
1. A Hi-C contact map in either matrix format or coordinate list format.

**Outputs**: 
1. A .pdb file of the predicted 3D structure corresponding to the second input file in ```Outputs/input_2_generalized_structure.pdb```.
2. A .txt file depicting the optimal conversion value and the dSCC value of the output structure```Outputs/input_2_generalized_log.txt```.
3. A .pt file of the trained model weights corresponding to the first input file in ```Outputs/input_1_weights.pt```.
4. A .txt of the normalized Hi-C contact map corresponding to the KR normalization of both input files in ```Data/input_matrix_KR_normed.txt``` if these files don't exist already.
5. A .txt file of the embeddings corresponding to the input files in ```Data/input_embeddings.txt``` if these files don't exist already. 
6. A .txt file of the input files in matrix format in ```Data/input_matrix.txt``` if these files don't exist already.

**Usage**: ```python HiC-GNN_generalize.py input_filepath1 input_filepath2```

* **positional arguments**: <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath1```: Path of the input file with which a model will be trained and later generalized on ```input_filepath2```. <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath2```: Path of the input file with which a generalized structure corresponding to a model trained on ```input_filepath1``` will be generated. <br />

* **optional arguments**: <br />	
	&nbsp;&nbsp;&nbsp;&nbsp; Same as ```HiC-GNN_main.py```
	
* **Example**: ```python HiC-GNN_generalize.py Data/GM12878_1mb_chr19_list.txt Data/GM12878_500kb_chr19_list.txt```

### HiC-GNN_embed.py
This script takes a single Hi-C contact map as an input and utilizes it to generate node embeddings. 

**Inputs**: 
1. A Hi-C contact map in either matrix format or coordinate list format.

**Outputs**: 
1. A .txt file of the embeddings corresponding to the input files in ```Data/input_embeddings.txt```.
2. (In the case that the input file was in list format) A .txt file of the input file in matrix format in ```Data/input_filename_matrix.txt```.

**Usage**: ```python HiC-GNN_generalize.py input_filepath```

* **positional arguments**: <br />
&nbsp;&nbsp;&nbsp;&nbsp;```input_filepath1```: Path of the input file with which a embeddings will be generated. <br />

* **optional arguments**: <br />	
	&nbsp;&nbsp;&nbsp;&nbsp;-h, --help  show this help message and exit<br />
	&nbsp;&nbsp;&nbsp;&nbsp;-bs, --batchsize <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Batch size for embeddings generation. Default value: 128. <br />
	&nbsp;&nbsp;&nbsp;&nbsp;-ep, --epochs <br />
		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of epochs used for embeddings generation. Default value: 10. <br />	
		
* **Example**: ```python HiC-GNN_embed.py Data/GM12878_1mb_chr19_list.txt```
