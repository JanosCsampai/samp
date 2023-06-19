# Report 3

### Best Training Result on 20 frame long scenes.
Prediction            |  Ground Truth
:-------------------------:|:-------------------------:
![](trajnet++/trajnetplusplusbaselines/visualizations/highD_v0.12/lstm_social_None.epoch25_modes1/pred_scene1071.png)|  ![Original](trajnet++/trajnetplusplusbaselines/visualizations/highD_v0.12/lstm_social_None.epoch25_modes1/gt_scene1071.png)


- Almost identical
- Rather short scenes
- No lane changes

### Questions and Ideas:
- Dataset possibly too straight forward (not enough interactions between cars) ?
- Other Datasets from **levelXdata** might be interesting in that case
- Filter highD scenes to include more lane changes


### Next Steps:
1. Evaluate a scene from a different dataset (woven planet or intersection dataset by **levelXdata**) to see what the model is actually doing:
Is it just continuing its direction at the same velocity?
2. Currently am taking a closer look at how Trajnet++ is structured
3. Design the more structured network and figure out how to integrate it into Trajnet++