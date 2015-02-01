# Week 4

1. Add Gaussian and salt-and-pepper noise with different parameters to an image of your choice. Evaluate what levels of noise you consider still acceptable for visual inspection of the image.

2. Apply median filter to the images you obtained above. Change the window size of the filter and evaluate its relationship with the noise levels.

3. Practice with Wiener filtering. Consider for example a Gaussian blurring (so you know exactly the H function) and play with different values of K for different types and levels of noise.

4. Compare the results of non-local-means from the previous week (use for example the implementation in www.ipol.im) with those of Wiener filtering.

5. Blur an image applying local averaging (select different block sizes and use both overlapping and not overlapping blocks). Apply to it non-local means. Observe if it helps to make the image better. Could you design a restoration algorithm, for blurry images, that uses the same concepts as non-local-means?

6. Make multiple (N) copies of the same image (e.g., N=10). To each copy, apply a random rotation and add some random Gaussian noise (you can test different noise levels). Using a registration function like imregister in Matlab, register the N images back (use the first image as reference, so register the other N-1 to it), and then average them. Observe if you manage to estimate the correct rotation angles and if you manage to reduce the noise. Note: Registration means that you are aligning the images again, see for example http://www.mathworks.com/help/images/ref/imregister.html or http://en.wikipedia.org/wiki/Image_registration

7. Apply JPEG compression to an image, with high levels of compression such that the artifacts are noticeable. Can you apply any of the techniques learned so far to enhance the image, for example, reduce the artifacts or the blocking effects? Try as many techniques as you can and have time to do.

8. Apply any image predictor as those we learned in Week 2. Plot the histogram of the prediction error. Try to fit a function to it to learn what type of distribution best first the prediction error.
