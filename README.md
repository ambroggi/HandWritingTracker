--- Instructions to Run ---

  The initial run instructions can be found in README_OLD.md, however, most of the optional arguements are not used

1. -ct or --confidencethreshold : Not used due to only looking at logit file
2. -vt or --variancethreshold : Not used due to only looking at logit file
3. -tr or --trainpath : Not used as there are many possible datasets
4. -v  or --validpath : Not used as there are many possible datasets
5. -te or --testpath : Not used as there are many possible datasets
6. -r  or --resultpath : Should still work but may not work with evaluation
7. -c  or --buildCSV : Should still work
8. -p  or --plotResults : Should still work
9. -o  or --overrideExits : Should still work
10. -d or --dataset: ADDED, select what dataset, possible values: "MNIST", "Flowers102", "Food101", "FasionMNIST", "Random", "Covertype"

  The code can be run by:
  1. `pip install -r requirements.txt`
  2. `python src/main.py` with whatever optional arguements you want

Results of image classifications are stored in HandWritingTracker/{dataset}/runs/{datetime}/results. 

--- Instructions evaluate data ---

WORK IN PROGRESS
