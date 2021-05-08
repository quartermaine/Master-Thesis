#!/usr/bin/env bash

usage() {
  cat - >&2 <<EOF
NAME
    download.sh - Brief description

SYNOPSIS
    download.sh [-h|--help]
    		[-p|--pipeline <arg>]
                [-s|--strategy <arg>]
                [-d|--derivative <arg>]]
       
REQUIRED ARGUMENTS
  pipeline   
  strategy   
  derivative 

OPTIONS
  -h, --help
          Print help menu 

  -p, --pipeline <arg>
                      options :   
                                [-ccs 
			         -cpac
			         -dparsf 
			         -niak]

  -s, --strategy <arg>
                      options :
			        [-filt_global (band-pass filtering and global signal regression)
			         -filt_noglobal (band-pass filtering only)
			         -nofilt_global (global signal regression only)
			         -nofilt_noglobal (neither)]

  -d, --derivative <arg>
                       options :
                                [- alff (Amplitude of low frequency fluctuations)
                                 - degree_binarize (Degree centrality with binarized weighting)
                                 - degree_weighted (Degree centrality with correlation weighting)
			         - eigenvector_binarize (Eigenvector centrality with binarized weighting)
			         - eigenvector_weighted (Eigenvector centrality with correlation weighting)
			         - falff (Fractional ALFF)
			         - func_mask (Functional data mask)
			         - func_mean (Mean preprocessed functional image)
			         - func_preproc (Preprocessed functional image)
			         - lfcd (Local functional connectivity density)
			         - reho (Regional homogeneity)
			         - rois_aal (Timeseries extracted from the Automated Anatomical Labeling atlas)
		 	         - rois_cc200 (" " from Cameron Craddock's 200 ROI parcellation atlas)
			         - rois_cc400 (" " " 400 ROI parcellation atlas)
			         - rois_dosenbach160 (" " from the Dosenbach160 atlas)
			         - rois_ez (" " from the Eickhoff-Zilles atlas)
			         - rois_ho (" " from the Harvard-Oxford atlas)
			         - rois_tt (" " from the Talaraich and Tournoux atlas)
			         - vmhc (Voxel-mirrored homotopic connectivity)]

  --     
          Specify end of options

EOF
}

fatal() {
    for i; do
        echo -e "${i}" >&2
    done
    exit 1
}


while [ "$#" -gt 0 ]; do

  case "$1" in
    -h|--help) usage;exit 0;;
    -p|--pipeline) PIPELINE="$2"; shift 2;;
    -s|--strategy) STRATEGY="$2"; shift 2;;
    -d|--derivative) DERIVATIVE="$2"; shift 2;;

    --pipeline=*) PIPELINE="${1#*=}"; shift 1;;
    --strategy=*) STRATEGY="${1#*=}"; shift 1;;
    --derivative=*) DERIVATIVE="${1#*=}"; shift 1;;
    --pipeline|--strategy|--derivative) echo "$1 requires an argument" >&2; exit 1;;

    -*) fatal "Unknown option '${1}' " "see '${0} --help' for usage" ;;
    *) handle_argument "$1"; shift 1;;
  esac
done


# Get control subjects
#awk -F, '$8 == 2' Phenotypic_V1_0b_preprocessed1.csv > controls.txt

# Get ASD subjects
#awk -F, '$8 == 1' Phenotypic_V1_0b_preprocessed1.csv > asd.txt

#cat controls.txt | cut --delimiter=, -f7 > control_subjects.txt

#cat asd.txt | cut --delimiter=, -f7 > asd_subjects.txt

MaxThreads=10

cwd=$(pwd)
# First download ASD, DX 2

# First download controls, DX 1

# Loop over control subjects
temp=`cat  ${cwd}/control_subjects.txt`
Subjects=()
subjectstring=${temp[$((0))]}
Subjects+=($subjectstring)

threads=0
for SubjectNumber in {0..572}; do

    Subject=${Subjects[$((${SubjectNumber}))]}
    echo "Now processing $SubjectNumber $Subject"
    wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/$PIPELINE/$STRATEGY/$DERIVATIVE/${Subject}_$DERIVATIVE.nii.gz --directory-prefix=${cwd}/DATA/CONTROL
    
    ((threads++))
    if [ "$threads" -eq "$MaxThreads" ]; then
        wait
	threads=0
    fi

done

# Loop over ASD subjects

temp=`cat  ${cwd}/asd_subjects.txt`
Subjects=()
subjectstring=${temp[$((0))]}
Subjects+=($subjectstring)

threads=0
for SubjectNumber in {0..538}; do

    Subject=${Subjects[$((${SubjectNumber}))]}
    echo "Now processing $Subject"
    wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/$PIPELINE/$STRATEGY/$DERIVATIVE/${Subject}_$DERIVATIVE.nii.gz --directory-prefix=${cwd}/DATA/ASD
  
    ((threads++))
    if [ "$threads" -eq "$MaxThreads" ]; then
        wait
	threads=0
    fi

done

echo "Downloading of files with "pipeline: $PIPELINE strategy: $STRATEGY derivative: $DERIVATIVE" completed!"




