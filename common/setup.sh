#!/bin/bash
for f in chap/*.tex
do
	line=$(head -n 1 $f)
	if [[ $line != %!TEX* ]]
	then
		sed -i "1i%!TEX root = ../$1" $f
		echo added tex root to $f
	fi
done
