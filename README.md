<p align="center">
<a href="https://spacenet.ai"><img src="sn_logo.png" width="350" alt="SpaceNet LLC"></a>
</p>
<h1 align="center">SpaceNet 6: Multi-Sensor All Weather Mapping</h1>
<h2 align="center">Competitor Solutions</h2>
<br>

## Summary
The five subdirectories in this repository comprise the code for the winning solutions of SpaceNet 6: Multi-Sensor All Weather Mapping Challenge hosted by TopCoder. Each subdirectory contains the competitors' written descriptions of their solution to the challenge. See the blog post on CosmiQ Works' blog [The DownlinQ](https://medium.com/the-downlinq/spacenet-6-announcing-the-winners-df817712b515) for an additional summary.

Competitors’ scores in the SpaceNet 6: Multi-Sensor All Weather Mapping Challenge compared to the baseline model. The overall score represents the SpaceNet Metric (x 100) for the entire scoring set. We also report model precision (ratio of false predictions) and recall (ratio of missed ground truth polygons):

![alt text](/performance_table.png)

The model architectures, ensemble and pre-training schemes, as well as training and inference time for each of the winning solutions. Note that the total contribution to the total NN’s ensembled is listed in parentheses in the Architectures column. Total training and inference times are calculated based upon an AWS p3.8xlarge instance with 4 NVIDIA Tesla V100 GPUs:

![alt text](/approach_table.png)

## Model Weights
All SpaceNet6 model weights can be downloaded from our s3 bucket:
```
aws s3 ls s3://spacenet-dataset/spacenet-model-weights/spacenet-6/
                           PRE 1-zbigniewwojna/
                           PRE 2-MaksimovKA/
                           PRE 3-SatShipAI/
                           PRE 4-motokimura/
                           PRE 5-selim_sef/
                           PRE baseline/
```

Questions about SpaceNet? Check out our website at https://spacenet.ai.
