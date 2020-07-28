#!/bin/bash

for fn in `ls`;do
    echo $fn
    chmod a+r $fn
    chmod a+w $fn
done
