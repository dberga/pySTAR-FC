## Download Original Files
mkdir DeepGazeII
export fileid=1kYUwoatqQUS5EabeeSDc6gRmCysnVZ6N && export filename=DeepGazeII.zip
#wget --no-check-certificate -O $filename 'https://docs.google.com/uc?export=download&id='$fileid
if ! type "gdown" > /dev/null; then 
	pip install gdown
fi
gdown --id $fileid -O $filename
unzip $filename -d DeepGazeII
rm $filename
cd DeepGazeII && mv deep_gaze/* . && rmdir deep_gaze && cd ..

## Update .py files (with contrib paths)
export fileid=174CHXHOQmzSP0fjB0fBzX2aIs-P6VhVo && export filename=DeepGazeII.zip
gdown --id $fileid -O $filename
unzip $filename -d DeepGazeII

## create dynamic link to ICF
ln -s DeepGazeII/ ICF



