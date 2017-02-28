#**This Write up is to reflect the reviewer's last point
Reflection describes the current pipeline, identifies its potential shortcomings and suggests possible improvements. There is no minimum length. Writing in English is preferred but you may use any language.
Below is redo:

*provided me with a theoretical analysis of your algorithm in the "Reflections" section, and what changes could be made to make the API more robust with various environmental factors :) Just a few sentences would suffice.

Theoretical:
1) Transformed the original video to grayscale then 
2) appied Gaussian blurr (Reviewer suggested Kernel_size of 3 or less) 
3) sent the result into Canny Edge detection 
4) Masked Area of interest and cut them to left & right to find each lane seperately, which allowed me to bolder & give more custom approach to the dashed lines right and solid left lanes on the center.
5) With Hough space

Improvements in the future?
Possibly preprocess yellow speration (vias color separation), which were utilized on later project to imporve results. We know for a fact that this works.
Instead of doing left & right split, as I did. Use horizontal segments on y axis (up/down)? This was also utilized in last project when searching cars as sliding windows.
The lane lines should have some reasonable distance 550 pixels apart above front of car. Also utilized in advanced lane finding example.
And have certain m & b combo in Hough space at 1st horiz seg forming 
m = +45~ degree & -45~ on right & b = 150 & 840 
This will eliminate false lines like center divider or false lines
at 2nd seqment will have similar m & b characteristics.
Plus, attempt to Add Loess (piecewise linear) to the lane marks as well? This is something I really would like to try, but this would require more familiarity with cv2.
Also, eliminate regions when horizonal lines are encountered  = hood or hoizon. This mask was also utilized in last project.
