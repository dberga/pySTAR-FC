import os
import sys

sys.argv=[sys.argv[0],None,None,None,None,None]

test_path='datasets/pascal-s/images_parsed/'
keywords=os.listdir(test_path)

sys.argv[4]='datasets/pascal-s/boundingboxes/'
sys.argv[5]='datasets/pascal-s/boundingboxes_mask/'

for k,keyword in enumerate(keywords):
    sys.argv[1]=keyword
    sys.argv[2]="".join([test_path,keyword])
    sys.argv[3]='datasets/pascal-s/masks'
    os.system("".join(["python3 src/test_params.py ",sys.argv[1]," ",sys.argv[2]," ",sys.argv[3]," ",sys.argv[4]," ",sys.argv[5]]))

