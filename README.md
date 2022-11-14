```cmd
conda create -n xxx python=3.8
conda activate xxx
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install pytorch-lightning
pip install pandas
pip install numpy
pip install sklearn
pip install pathlib2 
pip install Pillow
```





# Multi-instance learning 
Usage:
Data
Each instance is feature vector with fixed length. A bag contains variable number of these instances. Each instance has an id specifying, which bag does it belong to. Ids of instances are stored in vector with length equal to number of instances.
Create an instance of MilDataset by passing it instances, ids and labels of bags.