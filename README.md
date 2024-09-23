# Instructions to Generate data

  This code was created off of a fork of "https://github.com/AAndrade1223/HandWritingTracker", from the same lab, which used the Varmax method as advised by the lab PI Dr. Gokhan Kul. The code was modified to save outputs into a logit file, use multiple datasets, and have unknown classes (classes that do not appear in the the training data).
  The original run instructions can be found in README_OLD.md, however, most of the optional arguments are not used. 


  Possible arguments to `src/main.py`:
  1. ~~-ct or --confidencethreshold :~~ Not used due to only looking at logit file
  2. ~~-vt or --variancethreshold :~~ Not used due to only looking at logit file
  3. ~~-tr or --trainpath :~~ Not used as there are many possible datasets
  4. ~~-v  or --validpath :~~ Not used as there are many possible datasets
  5. ~~-te or --testpath :~~ Not used as there are many possible datasets
  6. -r  or --resultpath : Should still work but may not work with evaluation
  7. -c  or --buildCSV : Should still work, flag to build csv file of results. Default True
  8. -p  or --plotResults : Should still work, flag to plot images with results. Default True
  9. -o  or --overrideExits : Should still work, flag to override any early exits. Default False
  10. -d or --dataset: ADDED, select what dataset, possible values being "MNIST", "Flowers102", "Food101", "FasionMNIST", "Random", "Covertype"

  The code can be run by:
  1. `pip install -r requirements.txt` to install the required libraries.
  2. `python src/main.py`, it is suggested to specify --dataset.

Results of image classifications are stored in HandWritingTracker/{dataset}/runs/{datetime}/results. 
Note that --dataset "Random" is just randomly generated vectors.

# Instructions evaluate data

Once Data is generated it can then be evaluated using `src/F1_graphing`. It tries to grab the three most recent runs of the specified dataset to compile into F1 scores. The three most recent runs we completed have been saved to the git repository so this should not cause an error. It outputs four blocks in the following format,

```
"NameOfDataset", "TypeOfThreshold"
Thresholds: "dictionary of threshold values"
"Type" Mean: "F1 score"   (Once per type being run)
"Type" Mean: "Known-F1 score"\|"Unknown-F1 score"  (Once per type being run)
```

The meanings are as follows:
  * "NameOfDataset" - The name of the current dataset, this is listed so that you don't forget when running lots of tests.
  * "TypeOfThreshold" - This is the selection of thresholding method being used, we have four,
    * None - This means that the default thresholds are being used, we tried to pick good values for the thresholds but they may be biased.
    * A - This method is using the ROC score of the data to find the best possible threshold. Specifically it is looking for the true positive rate to be 80% of the total recall if you were using Softmax
    * B - This is just looking for the maximum recall score as it progresses along the points in the ROC graph. The ROC graph is used because it shows the possible important threshold points. But this is slow due to there being a lot of possible points.
    * C - This is looking for the maximum F1 score along the ROC graph points. As such it probably does the best but takes a long time.
  * "dictionary of threshold values" - This is a dictionary that contains all of the thresholds that the "TypeOfThreshold" found. Values above these thresholds are considered to be known/unknown depending on the algorithm. Soft is always zero due to the definition that it never classifies anything as unknown.
  * "Type" - This lists the type of the algorithm being used. 
  * "F1 score" - The F1 score, from 0 to 1, of all of the data weighted based on the number of appearances of a given class.
  * "Known-F1 score" - This is the F1 score when the data is filtered to be only known classes. That is classes that have appeared in the training dataset.
  * "Unknown-F1 score" - This is the F1 score when the data is filtered to only be the unknown classes. That is the classes that have only appeared during the testing phases.

These values were then copied into Excel sheets for graphing. We did not use a python based graphing library because we wanted to be able to check the underlying data of each graph which was not easy to do with code generated graphs. 


This file is less automated than the other files, it only has one argument
  * -d or --dataset: Possible dataset to use.

Note: Energy scores work the opposite way from the other scores so we often need to invert them.


# Other Information
There were some other possible threshold selection methods, d and e but they were not fully planned out for the paper. Also all of the graphs made for the paper are available with data in the `Excel/` folder. 

The CICIDS data was generated from a different project and brought over for analysis here so it cannot be rerun using this library.
