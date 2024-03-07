# Introduction:

This is a data analysis pipeline that takes in a stack of tiff images of yeast
microscopy (bright-field) and fluorescence data, and provides data structures containing :
- smaller movies, pieces of the bigger movie broken by microfluidic trap
- segmentation of those smaller movies using YeaZ2 model allowing to get object outlining individual cells
- tracking of segmented cells using btrack
- heuristic detection of the mother cell for each trap
- extraction of the fluorescence data for each mother cell
- summary plots for each trap

The pipeline contains the midap tool that also allows manual correction of both segmentation and tracking.

Please report any bug or other issue to: nmarounina@ethz.ch


# Installation:

How to install and run the pipeline with the provided example:

Create and activate the conda environment :

`git clone https://gitlab.ethz.ch/sis/yeast_segtrack_pipeline.git`

`cd yeast_segtrack_pipeline`

`conda env create --file=environment.yml`

`conda activate segtrack`

Download the weights for the Yeaz2 model and edit the .env file to rectify the path, so it points to newly downloaded weights.
From https://github.com/rahi-lab/YeaZ-GUI :

Download the parameters for segmenting phase contrast images from: https://drive.google.com/file/d/1tcdl34Aq11mrPVlyu0Qd4rUigw_6948b.

(>>>these are the weights that have been used for the test case>>>)
Download the parameters for segmenting bright-field images from: https://drive.google.com/file/d/1vnhkp54McM836yczh4F-YYJwPahbTsY0

Download the parameters for segmenting fission images form: https://drive.google.com/file/d/1h_Wz2d3UY0jkGtMrhl32iEqbOQVXsmKS.

Create the folder for the output data:

`mkdir output`

Install midap :

`git clone https://github.com/Microbial-Systems-Ecology/midap.git`

`cd midap` 

`pip install -e .`

Run the example provided in the repo:

 `cd ..`

 `python main.py -i ./input/small_movie.tif -t ./input/template.tif -fo 9 -fs 6 -tx 41 -ty 21`

On a 2022 MacBook Pro with Apple M2 chip it takes ~8 min to complete the test run.

## To run on a different tif movie: 

### The input movie :
The expected format is a 3D stack of images in tif format, the 3rd dimension being the time. 
The stack can contain single fluorescence images in between two bright-field images (Z-stack in fluorescence are not yet handled).
Fluorescence images has to be regularly spaced withing the stack, but they can start anytime within it,
e.g. there is a fluorescence image every 8 frames, starting from the frame 21; 
in the example movie, there is a fluorescence frame every 6 frames, starting at the frame num. 10.

### Mandatory input arguments :
mandatory input arguments are :
- -i should be followed by the path to the input tif movie
- -fo is the "fluorescence offset" parameter and should be the number of frames in between the beginning of the movie and the first fluorescence image.
- -fs is the "fluorescence step" parameter and should be the spacing of fluorescence frames after they start to appear in the movie

Please compare the example movie to the example command for an example on how to set those parameters.

-tx is the width, in pixels of the center of the trap
-ty is the height, in pixels of the center of the trap

### Optional input arguments :
The entire pipeline, with all steps runs as the default behavior. To exclude some steps from the pipeline please add one or several of the following flags :

- -no_s --no_segmentation : Exclude the segmentation step from the pipeline
- -no_fm --no_format_midap : Exclude the formatting of the data for midap
- -no_tr --no_tracking : Exclude the tracking step
- -no_m --no_mother : Exclude the detection of the mother cell
- -no_fl --no_fluor : Exclude the treatment of the fluorescence frames

- -t expects a path to the template. If this argument is omitted, it will prompt the creation of a template 

### How to proceed if no template is available for a given movie ?
You will have to run the pipeline in two steps.

First, run the segmentation and tracking steps, excluding the post-treatment steps. -t argument can be omitted,
and the values for -tx and -ty arguments, while mandatory, do not really matter and can be any integer value:

`python main.py -i ./input/small_movie.tif -fo 9 -fs 6 -tx 9831579 -ty 4397856 -no_m -no_fl`

This run will prompt the creation of the template. Next, you would need to determine the size of the center of the trap.
To do so, open the template image in Preview and try selecting a rectangle with your mouse, as shown here :
![size of the trap](./add-ons/Screenshot.png)

It was not possible to capture it with a screenshot, but as long as one toggles with the size of the rectangle, 
one can see its dimensions, in pixels. Those are the numbers that you will have to save for the -tx -ty arguments in the next step.

Run the second command by excluding the steps that have already run :

`python main.py -i ./input/small_movie.tif -t ./output/<timestamp>/<template_name>.tif -fo 9 -fs 6 -tx <the_right_value> -ty <the_right_value> -no_s -no_fm -no_tr`



# Detail of inputs/outputs of each step of the pipeline:
- Step 1 : does the segmentation of the full movie 
  - in : tiff movie in ./input folder (example: mymovie.tif)
  - out : 
    - a h5 file containing the result of the segmentation step in the ./input folder (mymovie.h5)
    - a tif movie where the fluor frames has been overwritten (./output/mymovie_no_fluor.tif)
    - a h5 file where the fluor frames has been overwritten (./output/mymovie_no_fluor.h5)
    
- Step 2 : separates the traps from the movie 
  - in : no_fluor tif and h5 files from previous step, optional : a template for a trap. If template is not provided, a creation of such template will be prompted.
  - out : folders in ./output/<timestamp>/split_data, one for each trap detected in the original movie. Each folder contains a tif and h5 files representing the cropped movie and the cropped segmentation data or a given trap

- step 3 : adapts the output format for midap 
  - in : system of folder and files created by the previous step
  - out : 
    - shift-corrected movie, saved as ./output/<timestamp>/split_data/<trap_number>/shift_corrected.tif where the traps are centered and the "drift" of the move has been compensated for
    - an additional folder, ./output/<timestamp>/split_data/<trap_number>/midap, containing two folders : cut_im and seg_im, where the movie and segmentation data are broken by frame and stored in png and tif formats, respectively.

- step 4 : tracking 
  - in : ./output/<timestamp>/split_data/<trap_number>/midap folders created previously
  - out : A set of files summarising the tracking :
    - ./output/<timestamp>/split_data/<trap_number>/midap/segmentation_bayesian.h5
    - ./output/<timestamp>/split_data/<trap_number>/midap/track_output_bayesian.csv
    - ./output/<timestamp>/split_data/<trap_number>/midap/tracking_bayesian.h5
    
- step 5 : identifies the mother cell 
  - in : ./output/<timestamp>/split_data/<trap_number>/midap/track_output_bayesian.csv
  - out : ./output/<timestamp>/split_data/<trap_number>/tracking_with_mother.csv
  
- step 6 : gets the fluorescence values of each mother cell
  - in : track_output_bayesian.csv, tracking_with_mother.csv, the original movie containing fluorescence data
  - out : ./output/<timestamp>/split_data/<trap_number>/tracking_with_fluor.csv, summary_trap_<trap_number>.png

# Miscellaneous:
- What to do if my tiff movie does not contain fluorescence images ?
  - use the -no_fl flag and provide dummy integer values for the mandatory -fo and -fs options

- If some segmentation images has been corrected, the tracking and post-treatment step 
- need to be re-run to update the rest of the outputs

- I have a new movie where the trap shape is the same but the movie is slightly tilted - can I reuse a template from another movie ?
  - unfortunately, no. The algorithm does not register the rotation of the template, and you would need to re-create a new template