# Masters Thesis in Mathematical Modelling and Computation

## Introduction

My Masters Thesis, "Using Convolutional Neural Networks and Data Science to Explore Patterns of Religiosity over Centuries" was submitted and defended in January 2020.
It was developed out of a project done by German psychology researchers who were using patterns on religious gravestones in their research on connections between religion and lifespan.

The main purpose of the thesis was automization of symbol detection on gravestones.

## Technologies used

To get imagery data of religious gravestones, the website [Find a Grave](https://www.findagrave.com) was used. The website could also be used for other information which has been registered.

A Mask R-CNN neural network was trained and used to detect seven different religious symbols. The seven symbols are:
- Cross
- Angel
- Dove
- Star of David
- Praying hands
- Bible
- Persons

For training, several hundreds of gravestones had to be manually segmented using the [VGG Image Annotator](https://annotate.officialstatistics.org). Data augmentation was also used in training. Training also used L2 regression for regularization.

## Results

For the result, a gravestone is labelled as religious if it has 1 or more religious symbol detected, otherwize it is lablled unreligious.

The results of the overall performance of the neural network on validation data reported that the mAP was 82.5%, with doves being the most misclassified symbol. 

For the correlation between religiosity and average age, it was found that in the United States that the correlation between county level religiosity and difference of mean age was positive, so that similar results were found to what had been found before the project was initialized. 
