#!bin/bash

if test "$#" -eq 5; then
	disp="$1"
	gt="$2"
	start_index="$3"
	end_index="$4"
	file="$5"
else
	disp="$1"
	start_index="$2"
	end_index="$3"
	file="$4"
fi

for i in $(seq -f "%06g" $start_index $end_index)
do
	if test "$#" -eq 5; then
		echo "$disp/"$i"_LR.png;$gt/"$i"_10.png" >> $file
	else
		echo "$disp/"$i"_LR.png;" >> $file
	fi
		
done



