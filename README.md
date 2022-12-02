# Siamese Transformer Networks for Key-Point Analysis

Project for the Human Languages Technologies course @ [University of Pisa](https://www.unipi.it/index.php/english)

<img src="imgs/unipi_logo.png" align="right" alt="Unipi logo">

Authors: [Alberto Marinelli](https://github.com/AlbertoMarinelli), [Valerio Mariani](https://github.com/sd3ntato)


### Abstract
Pretrained language models are nowadays becoming standard approaches to tackle various Natural Language Processing tasks. This is the reason why we decided to experiment with using **Transformers** in a **Siamese network** to solve these problems and understand how they work. Specifically, due to the extensive pre-training of the available language models and the small dataset, a pre-trained model of **BERT** and its variant **RoBERTa** inside a Siamese network were chosen.
In particular, this architecture was used to tackle task 1 in the [IBM's shared task 2021](https://github.com/ibm/KPA_2021_shared_task).  In this task, we are given a set of debatable topics, a set of key points, and a set of arguments, supporting or contesting the topic. We are then asked to match each argument to the topic it is supporting or contesting.
<br /><br />

### Running the Project
All the code for experimentation is reported in apposite [notebook](https://github.com/AlbertoMarinelli/Siamese-Transformer-Networks-for-Key-Point-Analysis/tree/master/notebooks) that can be runned on Google Colab, to speed up the computation required for the training and inference phase it is suggested to change the runtime type to GPU.
<br /><br />

### Main results
Despite the relatively small size of the Dataset we were given - expecially when compared to the corpus these models were trained on - both SBERT and Siamese-RoBERTa obtained some nice generalization of the ability to select the Key-Point of a sentence in a set of possible ones from the training set to the development (validation) set we were given. In fact, using the given metrics on the validation set, we obtained \~75% Mean Average Precision with SBERT and \~80% Mean Average Precision with Siamese-RoBERTA.
<br /><br />

<h3 align="left">Languages and Tools</h3>
<p align="left"></a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a><a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>
