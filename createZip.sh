#! /bin/bash

if [ -z $1 ] || [ -z $2 ]
then
    echo "Missing one of the two parameters required"
    echo "First I need the zip title"
    echo "Then I need the path for the files to zip"
else
    zipName="A01378844_A01194173_A01194101_A00821976-$1.zip"
    if [ "$3" == "--credit" ]
    then
        bash addCredits.sh $4
    fi
    cd $2
    pyFiles=($(ls *.py))
    pdfFiles=($(ls *.pdf))
    csvFiles=($(ls *.csv))
    zip $zipName "${pyFiles[@]}" "${pdfFiles[@]}" "${csvFiles[@]}" ../requirements.txt -j
fi