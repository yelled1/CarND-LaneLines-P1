#  **Finding Lane Lines on the Road** 
"""
# pipeline on a series of individual images, and later apply the result to a video stream 
"raw-lines-example.mp4" (also contained in this repository) 
Once you have a result that looks roughly like 
"raw-lines-example.mp4", 
you'll need to get creative and try to average and/or extrapolate the line segments you've detected 
to map out the full extent of the lane lines. 
You can see an example of the result you're going for in the video "P1_example.mp4"
Ultimately,
1) Just one line for the left side of the lane, 
2) and one for the right.
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# In[1]:
"""
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#get_ipython().magic('matplotlib inline')

# In[2]:
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
# printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
#plt.show()

"""
 **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
 cv2.inRange()` for color selection  
 cv2.fillPoly()` for regions selection  
 cv2.line()` to draw lines on an image given endpoints  
 cv2.addWeighted()` to coadd / overlay two images
 cv2.cvtColor()` to grayscale or change color
 cv2.imwrite()` to output images to file  
 cv2.bitwise_and()` to apply a mask to an image
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**
# Below are some helper functions to help get you started. They should look familiar from the lesson!
"""
import imp
import P1hlpF as P1h
imp.reload(P1h)

# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:

import os
for f in os.listdir("test_images/"):
    print(f)


# run your solution on all test_images and make copies into the test_images directory).

# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# We can test our solution on two provided videos:
# `solidWhiteRight.mp4`
# `solidYellowLeft.mp4`
# In[5]:
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# In[ ]:

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    return result


# Let's try the one with the solid white lane on the right first ...
# In[ ]:
'''
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')



# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.
# In[ ]:
# **At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[ ]:

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Reflections
# 
# Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?
# 
# Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!
# 

# ## Submission
# 
# If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:

#HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
'''
