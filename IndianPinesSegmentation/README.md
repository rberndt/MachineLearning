# Indian Pines HyperSpectral Image Segmentation

The Indian Pines dataset consists of images of a field taken at different frequencies.
Using these images, researchers try to develop algorithms that segment the fields into correct clusters.

## Unsupervised

Uses the kmeans clustering algorithm and a window smoothing algorithm

    python kmeansSimple.py

-31.43% accuracy

## Semisupervised

Iterates through many values for number of clusters and number of iterations for the kmeans clustering algorithm

    python kmeansAdvanced.py

-39.97% accuracy
