
#download SALICON (caffe)
mkdir SALICON
export fileid=18kf3sM2AoZJ6rlvZX2kwBY7TNlhMja-C && export filename=SALICON.zip
#wget --no-check-certificate -O $filename 'https://docs.google.com/uc?export=download&id='$fileid
if ! type "gdown" > /dev/null; then 
	pip install gdown
fi
gdown --id $fileid -O $filename
unzip $filename -d SALICON
rm $filename

#download SALICONtf
git clone https://github.com/ykotseruba/SALICONtf.git

