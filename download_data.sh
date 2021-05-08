#!/bin/bash

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
listOfDerivatives="alff degree_binarize degree_weighted dual_regression eigenvector_binarize eigenvector_weighted falff lfcd reho vmhc"

for d in $listOfDerivatives; do

# Loop over control subjects
temp=`cat  ${cwd}/control_subjects.txt`
Subjects=()
subjectstring=${temp[$((0))]}
Subjects+=($subjectstring)

threads=0
for SubjectNumber in {0..572}; do

    Subject=${Subjects[$((${SubjectNumber}))]}
    echo "Now processing $Subject"
    wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/${d}/${Subject}_${d}.nii.gz --directory-prefix=${cwd}/DATA/${d}/CONTROL

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
    wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/${d}/${Subject}_${d}.nii.gz --directory-prefix=${cwd}/DATA/${d}/ASD

    ((threads++))
    if [ "$threads" -eq "$MaxThreads" ]; then
        wait
	threads=0
    fi

done

done
#mv ASD_* ASDS
#mv CONTROL_* CONTROLS
