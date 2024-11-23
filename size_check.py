import imagesize

import os

i=0

for i in os.listdir('~/akshat/dip_project/dip/test/Masks'):
    #print(i)
    if imagesize.get('~/akshat/dip_project/dip/test/Masks/'+i) != (369,369):
        print(imagesize.get('~/akshat/dip_project/dip/test/Masks/'+i))
        
for i in os.listdir('~/akshat/dip_project/dip/test/Images'):
    #print(i)
    if imagesize.get('~/akshat/dip_project/dip/test/Masks/'+i) != (369,369):
        print(imagesize.get('~/akshat/dip_project/dip/test/Masks/'+i))
