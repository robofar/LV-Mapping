echo Creating the dataset path...

mkdir -p data
cd data

mkdir -p ipb_car
cd ipb_car

echo Downloading example data subset from IPB car dataset, 801 frames ...
wget -O ipbcar_test_subset.zip -c https://uni-bonn.sciebo.de/s/567hLWiUrk7KvQY/download

echo Extracting dataset...
unzip ipbcar_test_subset.zip

rm ipbcar_test_subset.zip

cd ../../..
