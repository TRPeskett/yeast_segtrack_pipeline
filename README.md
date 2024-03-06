# Introduction:

This is a data analysis pipeline that takes in a stack of tiff images of yeast
microscopy and fluorescence images, and provides data structures containing :
- smaller movies (space-wise), pieces of the bigger movie broken by microfluidic trap
- segmentation of those smaller movies using Yeaz2 model allowing to get individual cells
- accurate tracking of segmented cells using btrack
- heuristic detection of the mother cell for each trap
- extraction of the fluorescence data for each mother cell
- summary plots for each trap

The pipeline contains the midap tool that also allows manual correction of both segmentation and tracking.

Please report any bug or other issue to: nmarounina@ethz.ch

# Installation:

How to run the pipeline:
- conda installation
- expected input data : tiff movie, being a 3D stack of images, the 3rd dimension being the time. The stack can contain single fluorescence images (fluorescence Z-stack are not handled).
- necessary input arguments
- example run
- example run if no template for the trap

# Description of the pipeline:
- Step 1 : do the segmentation of the full movie 
  - in : tiff movie in ./input folder (example: mymovie.tif)
  - out : 
    - a h5 file containing the result of the segmentation step in the ./input folder (mymovie.h5)
    - a tif movie where the fluor frames has been overwritten (./output/mymovie_no_fluor.tif)
    - a h5 file where the fluor frames has been overwritten (./output/mymovie_no_fluor.h5)
    
- Step 2 : separate the traps from the movie 
  - in : no_fluor tif and h5 files from previous step, optional : a template for a trap. If template is not provided, a creation of such template will be prompted.
  - out : folders in ./output/<timestamp>/split_data, one for each trap detected in the original movie. Each folder contains a tif and h5 files representing the cropped movie and the cropped segmentation data or a given trap

- step 3 : adapt the output format for midap 
  - in : system of folder and files created by the previous step
  - out : 
    - shift-corrected movie, saved as ./output/<timestamp>/split_data/<trap_number>/shift_corrected.tif where the traps are centered and the "drift" of the move has been compensated for
    - an additional folder, ./output/<timestamp>/split_data/<trap_number>/midap, containing two folders : cut_im and seg_im, where the movie and segmentation data are broken by frame and stored in png and tif formats, respectively.

- step 4 : Tracking 
  - in : ./output/<timestamp>/split_data/<trap_number>/midap folders created previously
  - out : A set of files summarising the tracking :
    - ./output/<timestamp>/split_data/<trap_number>/midap/segmentation_bayesian.h5
    - ./output/<timestamp>/split_data/<trap_number>/midap/track_output_bayesian.csv
    - ./output/<timestamp>/split_data/<trap_number>/midap/tracking_bayesian.h5
    
- step 5 : Identify the mother cell 
  - in : ./output/<timestamp>/split_data/<trap_number>/midap/track_output_bayesian.csv
  - out : ./output/<timestamp>/split_data/<trap_number>/tracking_with_mother.csv
  
- step 6 : Get the fluorescence values of each mother cell
  - in : track_output_bayesian.csv, tracking_with_mother.csv, the original movie containing fluorescence data
  - out : ./output/<timestamp>/split_data/<trap_number>/tracking_with_fluor.csv, summary_trap_<trap_number>.png

# Miscellaneous:
- what to do if my tiff movie does not contain fluorescence images
- rerun tracking if segmentation has been corrected
- what to do if I have no template