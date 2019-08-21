# 3D-Vision-data
A method to extract stereo image and convert to 3D Video.

This involves recording a video of the same scenario using two different UVC cameras.
Which replicates how each human eye perceives an image (binocular vision).
The videos are subsequently broken down into their respective frames and stored as .jpg images. 
However,there would be a delay of a finite number of frames between the two cameras. 
This delay is calculated by performing a cross correlation between the frames. 
One dimensional correlation is employed , after compressing the 2D images into 1D vectors.
By this we will be able to determine which camera is lagging.
The most correlated frames from each of the cameras are subtracted.
Thus,most of the pixels in the difference frame will be 0(black). 
So, it becomes easy to store the data as it takes up less space,and also can be transmitted with ease.
This difference frame can then be later on superimposed onto the lagging frame to get back the original recorded frame. 
The leading video and the superimposed video are made to play side by side and viewed using a VR box. 
The video appears to be in 3D and it takes up less memory.
