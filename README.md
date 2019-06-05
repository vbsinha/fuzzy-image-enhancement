# [Fuzzy Image Enhancement](https://github.com/vbsinha/fuzzy-image-enhancement/)

Implementation of the Fuzzy Inference System for image enhancement as described in the paper:

YoungSik Choi & Raghu Krishnapuram. A Robust Approach to Image Enhancement Based on Fuzzy Logic. _IEEE Transactions on Image Processing_, ( Volume: 6 , Issue: 6 , Jun 1997 )

The repository contains the implementation for the filters **A**, **B**, **C**, **R1**, **R2**, **R3** and **R3-Crisp**, as proposed in the paper. Moreover we also add an **R4** and **R4-Crisp** filter which is a small modification of R3. In R4 we change the consequent of Rule 1 to filter B and consequent of Rule 2 to filter A (ie. small -> A and medium -> B). Also to compare against other out-of-box new methods, this implementation has facility for **Median**, **Sharpen**, **Gauss**, **TV Chambolle**, **TV Bregman** and **Biltaeral** filter.

## Setup

### Prerequisites
* Python 3.7
* pip (for setting up dependencies)

### Setting up dependencies
Use
```bash
$ pip install -r requirements.txt
```

## Running Experiments

First create the recommended directory structure.
```bash
mkdir images/noisy images/enhanced images/tc
```
Within the images directory, the directory original stores original greyscale images, noisy directory stores the noise added images, and enhanced stores the enhanced images obatined by applying filters.

To add noise to images (note that the image must be stored in `images/original` directory. Only pass the imagename as IMGNAME not the path):
```bash
python noise.py --image IMGNAME
```
Change the parameters of noise from within `noise.py` if needed.

To run the filters:
```bash
python main.py --image NOISYIMGPATH --method METHOD [--original ORIGINALIMGPATH]
```
NOISYIMGPATH is relative to `images/` typically something like `noisy/Cameraman_5_5_100.png`, similarly for ORIGINALIMGPATH (eg. `original/Cameraman.png`). When original is specified the RMSE is reported.
Possible options for METHOD are 'A', 'B', 'C', 'Med', 'R1', 'R2', 'R3', 'R3Crisp', 'R4', 'R4Crisp', 'All' and 'Plot'. 'All' applies all the filters. 'Plot' plots the weights of filter A, B and C and saves to `images/tc`. 'Med' applies the median filter.

To run the other out-of-box filters:
```bash
python compare.py --image NOISYIMGPATH --method METHOD [--original ORIGINALIMGPATH]
```
Possible options for METHOD are 'Sharpen', 'Gauss', 'TVC' (TV Chambolle), 'TVB' (TV Bregman) and 'Bil' (Biltaeral). Other arguments are same as for `main.py`.

## License
This code is provided using the [MIT License](LICENSE).

---
This project was a part of the course MA6040: Fuzzy Logic Connectives: Theory and Applications, offered in Spring 2019 at IIT Hyderabad.

Team members: [Vaibhav Sinha](https://vbsinha.github.io) and [Prateek Kumar](https://prateekkumarweb.github.io).