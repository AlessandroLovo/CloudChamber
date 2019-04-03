import os

path = os.environ['DIRECTORY']
for filename in os.listdir(path):
    if (filename.endswith(".h264")):
    	print('found '+filename )
    	name, dot, extension=filename.partition('.')
        os.system("ffmpeg -framerate 24 -i {0} -c copy {1}.mp4".format(path+filename, path+name))
        os.system("ffmpeg -i {1}{0}.mp4 -c copy -f segment -segment_time 0.001 -segment_list list.ffcat -reset_timestamps 1 {1}seg{0}%d.mp4".format(name, path))
        for segfile in os.listdir(path):
        	if segfile.startswith("seg{0}".format(name)):
        		segname, dot, extension=segfile.partition('.')
        		os.system("ffmpeg -i {1}{0}.mp4 -vf 'tblend=addition,framestep=10' {1}out{0}-%d.png".format(segname, path))	#framestep=5
        		#os.system("ffmpeg -i {0}.mp4 out{0}-%d.png".format(segname))
    else:
        continue
