from __future__ import print_function
from cartoonize import worker
import argparse
import multiprocessing
import time
import datetime
from multiprocessing import Queue, Pool
from queue import PriorityQueue
import cv2

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

def video(args):
    """
    Read and apply object detection to input video stream
    """

    # Set the multiprocessing logger to debug if required
    if args["logger_debug"]:
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)

    # Multiprocessing: Init input and output Queue, output Priority Queue and pool of workers
    input_q = Queue(maxsize=args["queue_size"])
    output_q = Queue(maxsize=args["queue_size"])
    output_pq = PriorityQueue(maxsize=3*args["queue_size"])
    pool = Pool(args["num_workers"], worker, (input_q,output_q))
    
    # created a threaded video stream and start the FPS counter
    vs = cv2.VideoCapture("inputs/{}".format(args["input_videos"]))
    fps = FPS().start()

    # Define the codec and create VideoWriter object
    if args["output"]:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('outputs/{}.avi'.format(args["output_name"]),
        #                       fourcc, vs.get(cv2.CAP_PROP_FPS),
        #                       (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #                        int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        print("fps: ", vs.get(cv2.CAP_PROP_FPS), ", width:", int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), ",height:", int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # 这里由于网络的输出时1280*720，如果要原视频宽高输出需要resize回原来长宽
        out = cv2.VideoWriter('outputs/{}.avi'.format(args["output_name"]),
                              fourcc, vs.get(cv2.CAP_PROP_FPS),
                              (1280,720))
    # Start reading and treating the video stream
    if args["display"] > 0:
        print()
        print("=====================================================================")
        print("Starting video acquisition. Press 'q' (on the video windows) to stop.")
        print("=====================================================================")
        print()

    countReadFrame = 0
    countWriteFrame = 1
    nFrame = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    print("********:", nFrame)
    firstReadFrame = True
    firstTreatedFrame = True
    firstUsedFrame = True
    while True:
        # Check input queue is not full
        if not input_q.full():
            # Read frame and store in input queue
            ret, frame = vs.read()
            if ret:            
                input_q.put((int(vs.get(cv2.CAP_PROP_POS_FRAMES)),frame))
                countReadFrame = countReadFrame + 1
                if firstReadFrame:
                    print(" --> Reading first frames from input file. Feeding input queue.\n")
                    firstReadFrame = False

        # Check output queue is not empty
        if not output_q.empty():
            # Recover treated frame in output queue and feed priority queue
            output_pq.put(output_q.get())
            if firstTreatedFrame:
                print(" --> Recovering the first treated frame.\n")
                firstTreatedFrame = False
                
        # Check output priority queue is not empty
        if not output_pq.empty():
            prior, output_frame = output_pq.get()
            if prior > countWriteFrame:
                output_pq.put((prior, output_frame))
            else:
                countWriteFrame = countWriteFrame + 1
                # output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                output_rgb = output_frame

                # Write the frame in file
                if args["output"]:
                    write_path = "/home/alpha/Pictures/cartonnize/"+str(countWriteFrame)+".jpg"
                    cv2.imwrite(write_path, output_rgb)
                    out.write(output_rgb)

                # Display the resulting frame
                if args["display"]:
                    cv2.imshow('frame', output_rgb)
                    fps.update()

                if firstUsedFrame:
                    print(" --> Start using recovered frame (displaying and/or writing).\n")
                    firstUsedFrame = False

                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("Read frames: %-3i %% -- Write frame: %-3i %%" % (int(countReadFrame/nFrame * 100), int(countWriteFrame/nFrame * 100)), end ='\r')
        if((not ret) & input_q.empty() & output_q.empty() & output_pq.empty()):
            break


    print("\nFile have been successfully read and treated:\n  --> {}/{} read frames \n  --> {}/{} write frames \n".format(countReadFrame,nFrame,countWriteFrame-1,nFrame))
    
    # When everything done, release the capture
    fps.stop()
    pool.terminate()
    vs.release()
    if args["output"]:
        print("this is output++++++++++++++++++++++++++")
        out.release()
        print("this is output--------------------------")
    cv2.destroyAllWindows()
    print("over") 
