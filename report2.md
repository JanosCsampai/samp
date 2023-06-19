# Report 2

### **Approaches of converting the highD data:**
1. Take all trajectories, convert to Trajnet++ Format, use Trajnet++ to create the scenes and categories:
![Test](trajnet++/trajnetplusplusbaselines/visualizations/trajnetplusplusdataset/test_private/gt_scene15249.png)
    - Was done like this in the MPFAV
    - HighD Data works way better for clean conversion to Trajnet++ (Scenes now actually make sense since the recording location is fixed)
    - Scenes tend to be a bit short with the default 8, 12 frame split

2. Use tracksMeta.csv to create scenes and categories
    - not necessary since Trajnet++ already "calculates" starting and ending frame for all trajectories
    - (could be used to include more information at some point)

### **tars.cps.in.tum.de**  provided server
~~Trajnet++ seems to have problems with the driver of the GPU~~

After changing package versions trajnet++ uses, works great and utilizes GPU!


### Next Steps:
1. Do a test training run on the converted data with the Bachelor Praktikum parameters
2. Design the more structured network and figure out how to integrate it into Trajnet++
(Trajnet++ does in fact use Pytorch)