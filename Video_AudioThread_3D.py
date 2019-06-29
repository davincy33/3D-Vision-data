import threading
import cv2
import pyaudio
import wave
import time
import numpy as np
import matplotlib.pyplot as plt
import math
# DEFINING AUDIO AND VIDEO RECORDING FUNCTIONS
def AudioRec(FORMAT,CHANNELS,RATE,CHUNK):
    WAVE_OUTPUT_FILENAME = "Audio1.wav"
    RECORD_SECONDS = 5
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Recording...")
    frame = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frame.append(data)

    print("Stopped Recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(p.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frame))
    waveFile.close()
    return

def VideoRec():
    cap = cv2.VideoCapture(1)
    cap.set(3, 160)
    cap.set(4, 120)
    cap2 = cv2.VideoCapture(2)
    cap2.set(3, 160)
    cap2.set(4, 120)
    w1 = int(cap.get(3))
    h1 = int(cap.get(4))
    print w1, h1
    # Checking if Camera has Opened Properly
    if (cap.isOpened() == False):
        print('Camera 1 Not Found')
    if (cap2.isOpened() == False):
        print("Camera 2 Not Found")

    # function for manual correlation
    def man_corr(a, b):
        n1 = len(a)
        n2 = len(b)
        a_copy = np.zeros(n1)
        b_copy = np.zeros(n2)
        # Calculating square
        sq_sum1 = 0
        sq_sum2 = 0
        for i in range(0, n1):
            a_copy[i] = a[i] ** 2
            sq_sum1 = sq_sum1 + a_copy[i]
        # print sq_sum1,a_copy
        for i in range(0, n2):
            b_copy[i] = b[i] ** 2
            sq_sum2 = sq_sum2 + b_copy[i]
        # print sq_sum2,b_copy
        sum = 0
        r = np.zeros(1)
        s = n1
        diff = 1
        if (n1 != n2):
            if n1 > n2:
                diff = n1 - n2 + 1
                r = np.zeros(diff)

                s = n1
                for q in range(0, n1 - n2):
                    b.append(0)

            else:
                diff = n2 - n1 + 1
                s = n2
                r = np.zeros(diff)
                for q in range(0, n2 - n1):
                    a.insert(0, 0)

        # print a,b
        for l in range(0, diff):
            for n in range(0, s):
                if n - l >= 0:
                    sum = sum + (a[n] * b[n - l])

            r[l] = sum / math.sqrt(sq_sum1 * sq_sum2)
            sum = 0
        return r

    # Function for splitting frames
    def Split(Frame, cam_no, currentFrame):
        name = 'Frame ' + str(cam_no) + '_' + str(currentFrame) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, Frame)
        return

    # Function of getting differences and blended frames
    def Rec_Play(delay, frameC):
        diff_count = 0
        if (delay > 0):
            for i in range(0, frameC - delay):
                cam1 = cv2.imread('Frame 1_' + str(i) + '.jpg')
                cam2 = cv2.imread('Frame 2_' + str(i + delay) + '.jpg')

                diff = cv2.subtract(cam1, cam2)
                print('Creating Difference frame ' + str(diff_count))
                cv2.imwrite('Difference_Frame ' + str(diff_count) + '.jpg', diff)
                blend = cv2.add(cam1, diff)
                print('Creating Blended frame ' + str(diff_count))
                cv2.imwrite('Blended_Frame ' + str(diff_count) + '.jpg', blend)
                diff_count += 1
            print diff_count
        else:
            delay1 = abs(delay)
            for i in range(0, frameC - delay1):
                cam2 = cv2.imread('Frame 2_' + str(i) + '.jpg')
                cam1 = cv2.imread('Frame 1_' + str(i + delay1) + '.jpg')
                diff = cv2.subtract(cam2, cam1)
                print('Creating Difference frame ' + str(diff_count))
                cv2.imwrite('Difference_Frame ' + str(diff_count) + '.jpg', diff)
                blend = cv2.add(cam2, diff)
                print('Creating Blended frame ' + str(diff_count))
                cv2.imwrite('Blended_Frame ' + str(diff_count) + '.jpg', blend)
                diff_count += 1
            print diff_count

        return diff_count

    # Function to get Blended frames of camera 1 and difference
    def Merge(bframe):
        out3 = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (w1, h1))
        currentFrame = 0
        while (currentFrame <= bframe):
            frame = cv2.imread('Blended_Frame ' + str(currentFrame) + '.jpg')
            out3.write(frame)
            currentFrame += 1
        return

    # Function to perform Correlation
    def Correlation(image1, image2):
        img = cv2.imread(image1)
        img2 = cv2.imread(image2)
        ha, wa, bppa = np.shape(img)
        hb, wb, bppb = np.shape(img2)

        sum_matrix = 0
        sum_matrix1 = 0
        img_mean1 = img2
        img_mean = img
        for i in range(0, ha):
            for j in range(0, wa):
                m = img[i][j]
                ps = m[0] + m[1] + m[2]
                pavg = ps / 3
                img_mean[i][j] = pavg
                sum_matrix = sum_matrix + pavg
        mean1 = sum_matrix / (ha * wa)
        img_mean = img_mean / mean1

        ##  normalization for image 2
        for i in range(0, hb):
            for j in range(0, wb):
                m = img2[i][j]
                ps = m[0] + m[1] + m[2]
                pavg = ps / 3
                img_mean1[i][j] = pavg
                sum_matrix1 = sum_matrix1 + pavg
        mean2 = sum_matrix1 / (hb * wb)
        img_mean1 = img_mean1 / mean2
        # print mean2
        # print sum_matrix1
        # print img_mean1
        # Converting 2D image to 1-D vector
        # adding column pixels

        f_mat1 = np.zeros(wa)  # The final 1D matrix of image 1
        c_sum = 0
        for p in range(0, wa):
            for q in range(0, ha):
                e = img_mean[q][p]

                c_sum = c_sum + e[0]

            f_mat1[p] = c_sum
            c_sum = 0

        # Converting 2D image2 to 1D vector
        f_mat2 = np.zeros(wb)
        c_sum = 0
        for p in range(0, wb):
            for q in range(0, hb):
                e = img_mean1[q][p]
                c_sum = c_sum + e[0]

            f_mat2[p] = c_sum  # THe final 1D matrix of image 2
            c_sum = 0

        correlation = man_corr(f_mat1, f_mat2)
        return correlation
        # print np.corrcoef(f_mat1,f_mat2)

    # Creating VideoWriter Object for camera 1 and camera 2
    out1 = cv2.VideoWriter('Cam1_out.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (w1, h1))
    out2 = cv2.VideoWriter('Cam2_out.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (w1, h1))
    currentFrame = 0
    frame_count = 0
    # Loop for capturing and displaying frame by frame
    while (1):
        ret, frame = cap.read()
        if ret:
            out1.write(frame)
            Split(frame, 1, currentFrame)

            cv2.imshow('Frame1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):

                cap.release()
                cv2.destroyWindow('Frame1')
            elif cv2.waitKey(1) == 27:
                break

        ret, frame2 = cap2.read()
        if ret:
            out2.write(frame2)
            Split(frame2, 2, currentFrame)
            cv2.imshow('Frame2', frame2)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                cap2.release()
                cv2.destroyWindow('Frame2')
            elif cv2.waitKey(1) == 27:
                break

        currentFrame += 1
        frame_count += 1
    out1.release()
    out2.release()
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
    print('Number of Frames = ' + str(frame_count))
    x = np.zeros(frame_count)
    for p in range(0, frame_count):
        print('Iteration ' + str(p + 1))
        x[p] = Correlation('Frame 1_' + str(frame_count / 2) + '.jpg', 'Frame 2_' + str(p) + '.jpg')
    print('Correlation array:')
    print x
    max = np.max(x)
    print('Maximum Correlation :' + str(max))
    max_ind = list(x).index(max)
    print ('Index of Max: ' + str(max_ind))
    delay = max_ind + 1 - (frame_count / 2)
    if (delay > 0):
        print('Cam 2 lagging')
        print('Frame Delay= ' + str(delay))
    else:
        print('Cam 1 lagging')
        print('Frame Delay=' + str(delay))
    Bframe_no = Rec_Play(delay, frame_count)
    Merge(Bframe_no)
    plt.plot(x)
    plt.show()
    return

if __name__ == "__main__":
    t1=threading.Thread(target = AudioRec, args=(pyaudio.paInt16, 2, 44100, 1024, ))
    t2=threading.Thread(target = VideoRec)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
