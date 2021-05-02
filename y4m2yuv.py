import os
path = '/public/rawvideo/test_18/com_yuv/'
file_list = os.listdir(path)
for file in file_list:
    name = file[:-4]
    a = path+file
    b = path+name
    size = file[-15:-8]
    #os.system('ffmpeg -i '+a+' '+'/public/rawvideo/test_yuv/'+name+'.yuv')
    #os.system('ffmpeg -pix_fmt yuv420p -s '+size+' -r 25 -i '+a+' -c:v libx265 -b:v 200k -x265-params pass=1:log-level=error -f null /dev/null')
    os.system('ffmpeg -pix_fmt yuv420p -s '+size+' -r 25 -i '+a+' -c:v libx265 -b:v 200k -x265-params pass=2:log-level=error ''/public/rawvideo/test_18/com_mkv/'+name+'.mkv')
    #os.mkdir('/public/rawvideo/test_18/raw_png/'+name)

    #os.system('ffmpeg -i '+a+' /public/rawvideo/test_18/raw_png/'+name+'/%3d.png')