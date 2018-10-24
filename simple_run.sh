#!/bin/bash

# Download and unzip DAVIS train/val if it is not already there (~800MB)
if [ ! -d "./data/DAVIS" ]; then
  echo "################# DOWNLOADING DAVIS DATA #################"
  wget -P ./data https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
  echo "################# UNZIPPING DAVIS DATA #################"
  unzip ./data/DAVIS-2017-trainval-480p.zip -d ./data
fi

# Download and unzip weights if they are not already there  (~3GB)
if [ ! -d "./weights/PReMVOS_weights" ]; then
  echo "################# DOWNLOADING WEIGHTS #################"
  wget -P ./weights https://www.vision.rwth-aachen.de/media/resource_files/PReMVOS_weights.zip
  echo "################# UNZIPPING WEIGHTS #################"
  unzip ./weights/PReMVOS_weights.zip -d ./weights
fi

# Run method on sequences defined in ./seq_to_run.txt

# Calculate optical flow
FLOW_LOC=./output/intermediate/flow
if [ ! -d "$FLOW_LOC" ]; then
  echo "################# GENERATING FLOW #################"
  ./code/optical_flow_net-PWC-Net/script_pwc_multi.py
fi

# Generate general proposals
GENERAL_PROP_LOC=./output/intermediate/general_proposals
if [ ! -d "$GENERAL_PROP_LOC" ]; then
  echo "################# GENERATING GENERAL PROPOSALS #################"
  PROP_NET_GENERAL_WEIGHTS=./weights/PReMVOS_weights/proposal_net/general_weights/proposal_general_weights
  ./code/proposal_net/train.py --forward "$GENERAL_PROP_LOC" --agnostic --second_head --forward_dataset DAVIS --load "$PROP_NET_GENERAL_WEIGHTS" --davis_name "$PWD/seq_to_run.txt"
fi

# Generate specific proposals
SPECIFIC_PROP_LOC=./output/intermediate/specific_proposals
if [ ! -d "$SPECIFIC_PROP_LOC" ]; then
  echo "################# GENERATING SPECIFIC PROPOSALS #################"
  PROP_NET_SPECIFIC_WEIGHTS=./weights/PReMVOS_weights/proposal_net/general_weights/proposal_general_weights
  ./code/proposal_net/train.py --forward "$SPECIFIC_PROP_LOC" --agnostic --second_head --forward_dataset DAVIS --load "$PROP_NET_SPECIFIC_WEIGHTS" --davis_name "$PWD/seq_to_run.txt"
fi

# Combine general and specific proposals
COMBINED_PROP_LOC=./output/intermediate/combined_proposals
if [ ! -d "$COMBINED_PROP_LOC" ]; then
  echo "################# GENERATING COMBINED PROPOSALS #################"
  ./code/proposal_net/combine_general_and_specific.py
fi

# Refine proposals
REFINED_PROP_LOC=./output/intermediate/refined_proposals
if [ ! -d "$REFINED_PROP_LOC" ]; then
  echo "################# GENERATING REFINED PROPOSALS #################"
  cd code
  REFINEMENT_CONFIG=./refinement_net/configs/run
  echo "$REFINEMENT_CONFIG"
  ./refinement_net/main.py "$REFINEMENT_CONFIG"
  cd ..
fi

# Add ReID to proposals
ReID_PROP_LOC=./output/intermediate/ReID_proposals
if [ ! -d "$ReID_PROP_LOC" ]; then
  echo "################# GENERATING ReID PROPOSALS #################"
  cd code
  ReID_CONFIG=./ReID_net/configs/run
  echo "$ReID_CONFIG"
  ./ReID_net/main.py "$ReID_CONFIG"
  cd ..
fi

# Run merging algorithm
MERGE_OUT_LOC=./output/final
if [ ! -d "$MERGE_OUT_LOC" ]; then
  echo "################# Merging #################"
  cd code
  ./MergeTrack/merge.py
  cd ..
fi
