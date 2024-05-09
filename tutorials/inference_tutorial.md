# Inference Tutorial
## Minimum Working Example
A minimum working example can be found in `single_infer.py`. This will download an image from the FEMA CAP repository and pass it through the NN.

## Batch Processing and Deployment
The barebones environment and code needed to run inference can be found in `labeling/container`. A Dockerfile is included for those who wish to run a containerized version, and the scripts can be found in `labeling/container/inference`. There are two inference scripts provided:

- `url_list_infer.py` will run inference on a list of HTTP URLs which point to images. Each image will be downloaded and the results of inference will be written to a CSV file. The script will also parse latitude and longitude information from the EXIF metadata in the image and emit this along with the classifier results in the CSV file.
- `file_list_infer.py` will run inference on a list of paths to local images. Each image will be downloaded and the results of inference will be written to a CSV file. The script will also parse latitude and longitude information from the EXIF metadata in the image and emit this along with the classifier results in the CSV file.
- `aws_list_infer.py` will run inference on a list of S3 bucket URLs, in the format `s3://<bucket_name>/<tag>`. This confers considerable speed advantages within the AWS ecosystem (eg: running the inference on an EC2 instance while pulling from s3). It's also considerably cheaper to keep data within AWS than to pull things off of it. The results of inference will be written to a CSV file. The script will also parse latitude and longitude information from the EXIF metadata in the image and emit this along with the classifier results in the CSV file.

Currently, both scripts run example tasks where the `URLListDataset` or `AWSListDataset` classes are constructed with example lists. It should be relatively simple to modify this example to pass a different list (or, to run from the command line, perhaps `sys.argv[1:]`).

To run any file as-is, just set up your environment and run `python <file_name.py>`. For `file_list_infer.py` you need to supply an argument containing a list of filepaths to read, one path per line: `python file_list_infer.py file_list.txt`.

