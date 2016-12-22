## A Recommender System Supporting Multiple Algorithms
 
Team members : Sewon Min, Chaofan Han
 
### Project Abstract
In the current era of information explosion, recommender systems gain more and
more popularity as they can make appropriate and customized recommendations to
users based on users’ past behavior and profiles. It would be very interesting to
explore how commonly used algorithms of a recommender system work through
code implementation and compare their performance and time costs.

### Data & Tools
- Amazon Review Data: http://snap.stanford.edu/data/web-Amazon-links.html
- Programming language & Libraries: Python 2.7, Numpy, Pandas, Tensorflow
- Database Management System: MySQL
- Computing Instance: AWS EC2 m4.large

### Task Assignment:
- Data Preprocessing (CH)
- Data Reader (CH)
- Content-based Recommender System (CH)
- Collaborative Filtering (CH)
- Weight Learned (SM)
- Latent Factor Model (SM)
- Bias Extension (SM)

### Dataset Profile
| Dataset range | Data size | Min # of reviews in each product | Min # of reviews in each user | # of unique users | # of unique items |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Large | 1121296 | 100 | 10 | 46824 | 5801 |
| Small | 39126 | 150 | 15 | 1504 | 165 |



### Instructions

#### 1. Download data
You need to have download the data first and it will take several minutes..

```bash
chmod +x download.sh; ./download.sh 
```

#### 2. Preprocess data and Create DB
```bash
python prepro/preprocess.py
```

#### 3. Run Recommenders
```bash
python -m model.main
```
It builds a recommender with the training data and also evaluates the performance on the test data. If you want to specify a certain recommender system, you can use '--recom'.

Content Based : 'cb'
Collaborative Filtering : 'cf'
Weight Learned : 'l'
Latent Factor : 'lf'
Latent Factor with Bias Extension : 'blf'

For example, if you want to run Weight Learned Recommender,
```bash
python -m model.main --recom l
```
If you want to run both Content Based and Collaborative Filtering Recommenders,
```bash
python -m model.main --recom cb cf
```
It runs a recommender on the small dataset by default. If you want to run on a large dataset, you can use '--small False'. Batch size is 128 by default. If you want to change it, you can use '--batch_size'. For example,
```bash
python -m model.main --small False --batch_size 256
```

### References
- [1] G. Adomavicius and A. Tuzhilin. 2005. “Towards the next generation of recommender systems: a survey of the state-of-the-art and possible extensions.” IEEE Transactions on Knowledge and Data Engineering 17 (6): 734– 749.
- [2] Y Koren, R Bell, C Volinsky. 2009. “Matrix factorization techniques for recommender systems.” Computer 42 (8): 30-37.
- [3] J. McAuley, C. Targett, J. Shi, A. van den Hengel. 2015. “Image-based recommendations on styles and substitutes.” SIGIR.
