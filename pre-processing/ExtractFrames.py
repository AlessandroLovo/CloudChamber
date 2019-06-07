import os
import sys
import time


def ExtractFrames(folder, data, do_segment=True, start=0, end=-1, segment_time=10, framestep=6, only_one_video = False, video_name=''):
    
    path = folder+data+'/'                                              # with fps = 24 and framestep = 6, we have a frame every 0.25 s
    if not folder.endswith('/'):
        path = folder+'/'+data+'/'
    frames_path=path+"frames/"
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    
    segments_path = path
    if do_segment:
        segments_path=path+"segments/"
        if not os.path.exists(segments_path):
            os.makedirs(segments_path)
    
    start_time = time.time()
        
    for filename in os.listdir(path):
        if (filename.endswith(".h264")):
            if not only_one_video:
                print('found '+filename)
            name, dot, extension=filename.partition('.')
            if only_one_video:
                if name != video_name:
                    continue
            os.system("ffmpeg -framerate 24 -i {0} -c copy {1}.mp4".format(path+filename, segments_path+name))
            if do_segment:
                os.system("ffmpeg -i {1}{0}.mp4 -c copy -f segment -segment_time {2} -segment_list list.ffcat -reset_timestamps 1 {1}seg{0}_%03d.mp4".format(name, segments_path, segment_time))
                count = 0
                for segfile in os.listdir(segments_path):
                    if segfile.startswith("seg{0}".format(name)):
                        if count == end:
                            break
                        if count < start:
                            count += 1
                            continue
                        count += 1
                        segname, dot, extension=segfile.partition('.')
                        sname, dot, number = segname.partition('_')
                        os.system("ffmpeg -i {1}{0}.mp4 -vf 'tblend=addition,framestep={6}' {2}outvid-{3}-{4}_{5}-%03d.png".format(segname, segments_path, frames_path, data, name, number, framestep))	#framestep=5
                        #os.system("ffmpeg -i {0}.mp4 out{0}-%03d.png".format(segname))
            else:
                os.system("ffmpeg -i {1}{0}.mp4 -vf 'tblend=addition,framestep={4}' {2}outvid-{3}-{0}-%03d.png".format(name, path, frames_path, data, framestep))	#framestep=5
    
        else:
            continue
    
    end_time = time.time()
    
    print('Time required for the Extraction: '+str(end_time - start_time))


if __name__ == '__main__':
    folder = str(sys.argv[1])#che termina con '/'
    data = str(sys.argv[2])# solo numeri
    
    start = 0
    end = -1
    if len(sys.argv) == 5:
        start = int(sys.argv[3])
        end = int(sys.argv[4])
        if end == 0:
            end = -1
    
    do_segment = True
    
    ExtractFrames(folder, data, do_segment, start, end, segment_time=10,framestep=5)