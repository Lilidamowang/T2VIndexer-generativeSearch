# T2VIndexer
Implementation for paper "T2VIndexer: A Generative Video Indexer for Efficient Text-Video Retrieval" in ACM MM 2024.

-------

## Introduction
Here you will find the source code I used in my paper. Due to time constraints, the code may appear somewhat disorganized, but it includes all the essential parts for anyone interested in our work to benefit from.

## Code Structure
The current code structure might not be very clear, but it encompasses all the necessary functional modules. I plan to reorganize the code and provide a more user-friendly interface in the future \[ Though as a PhD student, my time is often limited, especially facing with some PROJECT DEADLINE! Thanks for your patience! :) \].

## Usage

> **I highly suggest that directly copy the model file like `main_models.py` and `Model/transformers/` to your own project, that will be easier to use this model XD.**

To utilize the model, begin by navigating to the `Model` directory, where you will find the primary model file located at `main_models.py` and additional transformer components within the `Model/transformers/` subdirectory.

Next, prepare your dataset according to the specifications detailed in our published paper and the accompanying code provided in the `DatasetProcess/` directory.

Once your data is ready, initiate the training process by executing the shell script:
```
bash Model/train.sh
```

## Future Plans ~~(画饼 in Chinese)~~

I intend to (MAYBE):

* Reorganize the code structure
* Provide more detailed documentation and comments
* Optimize the user interface for better usability


## Contact
If you have any questions, please contact me via:

* Email: liyili@iie.ac.cn or monlilirua@gmail.com (~~Usually receives spam in google. XD~~)

> **Although this project may useless, please star if possiable!**

## Acknowledgment
Our code is built based on NCI project, and more details can be found here:
https://github.com/solidsea98/Neural-Corpus-Indexer-NCI. THANKS!!! Without this project, I won't finish this paper.
