matlabPro
=========
This repo contains matlab implementation of image prcessing and reconstruction.

"pca_cmu_pie.m" and "pca_dataset.m" implement Principal Component Analysis (PCA)
for to different datasets, namely dataset.7z and CMU-PIE.

"reconstruct.m" takes images from one dataset and trains itself. When one gives it an image
that doesnt belong to that dataset it tries to reconstruct it using the traning dataset.

"svm_classify.m" takes input of any given data and convert into a Support Vector Machine (SVM)
input file, after this one can run command line svm to get the accuracy.

"ROC.m" takes the dataset and outputs the ROC curves.




                                                    - END -
