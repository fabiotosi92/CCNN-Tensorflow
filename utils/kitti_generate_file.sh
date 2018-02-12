#!bin/bash

disp="$1"
gt="$2"
start_index="$3"
end_index="$4"
file="$5"

for i in $(seq -f "%06g" $start_index $end_index)
do
	echo "$disp/"$i"_LR.png;$gt/"$i"_10.png" >> $file	
done



