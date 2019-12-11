# farm
=======
Family of AutoRegressive Models

This is a content dump for CS 236. 

There was some trouble in adding PixelCNN++ code as it was originally cloned and git acts weird with sub-repos. We have copied over most of the relevant changed files outside of the sub-repos.

MADE models can be trained like so.  The below line trains a model with two hidden layers (1000 nodes each) on 8 orderings.

```python train.py -m=model_name -q=1000,1000 -o=8```

See the comment at the bottom of `experiments.py` for more examples.
