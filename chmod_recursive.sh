#!/bin/bash

function chmod_recursive() {
    for file in `ls $1`;do
        if [ -d $1"/"$file ];then
            echo "dicrectory: $file"
            chmod a+r $1"/"$file
            chmod a+w $1"/"$file
            chmod_recursive $1"/"$file
        else
            echo "file: $file"
            chmod a+r $1"/"$file
            chmod a+w $1"/"$file
        fi
    done
}

chmod_recursive $1
