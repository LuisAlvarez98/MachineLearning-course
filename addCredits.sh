if [ -z $1 ]
then
    echo "Missing Param of path of the .py files to add credit"
else
    array=($(ls $1*.py))
    for i in "${array[@]}"
    do
        python3 utils/modifier.py $i $2
    done
    echo "Done Modifying the values"
fi