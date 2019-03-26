#!/bin/bash
search_dir=./job_scripts/RoboschoolInvertedPendulum-v1
for entry in "$search_dir"/*
do
  if [ -f "$entry" ];then
    sbatch "$entry"
  fi
done