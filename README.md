# VFX_NTU_2020
Homework directory of Visual effects

# HDR
## Requirement
Please ensure that you have installed the packages below before running the program
1. opencv-python
2. numpy
3. matplotlib

## Procedure
The following are the procedure running on example data `nightsight`.
### Image Alignment
`python image_alignment.py`
### HDR
`python hdr.py`
The default sampling method is downsampling (actually this method is not suitable to `nightsight` image set).
If you want to use other sampling method, simply modity line 216 in `hdr.py`.

## Result
The recovered HDR image will be stored at `./image/<data folder>/radiance_map.hdr`.
The tone-mapped images are stored in folder `./image/<data folder>/tone_mapping`.