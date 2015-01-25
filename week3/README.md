# Week 3

1. Implement a histogram equalization function. If using Matlab, compare your implementation with Matlab’s built-in function.

2. Implement a median filter. Add different levels and types of noise to an image and experiment with different sizes of support for the median filter. As before, compare your implementation with Matlab’s.

3. Implement the non-local means algorithm. Try different window sizes. Add different levels of noise and see the influence of it in the need for larger or smaller neighborhoods. (Such block operations are easy when using Matlab, see for example the function at http://www.mathworks.com/help/images/ref/blockproc.html). Compare your results with those available in IPOL as demonstrated in the video lectures.

4. Consider an image and add to it random noise. Repeat this N times, for different values of N, and add the resulting images. What do you observe?

5. Implement the basic color edge detector. What happens when the 3 channels are equal?

6. Take a video and do frame-by-frame histogram equalization and run the resulting video. Now consider a group of frames as a large image and do histogram equalization for all of them at once. What looks better? See this example on how to read and handle videos in Matlab:

    ```matlab
    xyloObj = VideoReader('xylophone.mp4');

    nFrames = xyloObj.NumberOfFrames;
    vidHeight = xyloObj.Height;
    vidWidth = xyloObj.Width;

    % Preallocate movie structure.
    mov(1:nFrames) = struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'), 'colormap', []);

    % Read one frame at a time.
    for k = 1 : nFrames
        im = read(xyloObj, k);

        % here we process the image im
        mov(k).cdata = im;
    end

    % Size a figure based on the video's width and height.
    hf = figure;
    set(hf, 'position', [150 150 vidWidth vidHeight])

    % Play back the movie once at the video's frame rate.
    movie(hf, mov, 1, xyloObj.FrameRate);
    ```

7. Take a video and do frame-by-frame non-local means denoising. Repeat but now using a group of frames as a large image. This allows you for example to find more matching blocks (since you are searching across frames). Compare the results. What happens if now you use 3D spatio-temporal blocks, e.g., 5×5×3 blocks and consider the group of frames as a 3D image? Try this and compare with previous results.

8. Search for “camouflage artist liu bolin.” Do you think you can use the tools you are learning to detect him?
