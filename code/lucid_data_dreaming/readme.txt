This is not necessary for the simple version of PReMVOS but is used for the ACCV paper version and the CVPR challenge winning version (but not the ECCV challenge winning version).

To use this, the original Lucid Data Dreaming code needs to be downloaded from here:

https://github.com/ankhoreva/LucidDataDreaming

And then it can be run with my script (run.py) to generate image augmentations per image sequence.

Warning: This code is very slow, it could be sped up immensely with some optimisations, we however just simply ran 1000 processes in parallel, on 1000 different CPUs in order to get this to finish quick enough.
