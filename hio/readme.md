#about
this project aims to answer MDA queries under LDP

# code
* exp.py: main entrance for experiment
* environment.yml: conda environment
* validate_scripts: folder that contains small experiments to verify small ideas
* data: folder of datasets
* results: folder of results
* draw: folder of figures and scripts to draw figures in paper (currently directly draw to paper folder) 
* client: folder that generate client-side data （mainly read data)
* server: folder that contains server-side code
  * fo: the frequency oracles, for both perturb and aggregate
  * query_handler: implementation of different query processors
  * server.py
  * hdtree.py
  * aggregator (deprecated): a collection of aggregators from FOs
  
# data structures
* support queries like: select avg(age) from foo where income=1 and native-country=United-States
* different algorithms:
  * framework level
    * exact
    * round (not supported)
    * average
  * aggregator level
    * hdtree: a very general class that can express HI, HIO, FO, SC, and even more
     * it should be decomposed into different files for HI, HIO, FO, and SC (this will make the code cleaner)
  * fo level
    * rr
    * ue
    * lh
* estimation tree: 
  * this is used to calculate all possible values in the domain, but it will be infeasible when the domain is large
  * therefore a subset of the domain can be used 
  * the tree is actually a dictionary, it has several levels
    * d: number of attributes
      * A: tuple of the d attributes - each attribute is a tuple of two info, attr index and the layer
        * V: tuple of the d values in these dimensions (attributes)

# datasets
In the dataset folder:
 1. uci contains smaller datasets (around 30k) from uci repository 
  a. adult
  b. ipums
  c. bank 

     
1. install and use anaconda
    1. create environment in the server: conda env create -f environment.yml; source activate db_code
    2. share the environment: conda env export > environment.yml
    
# run the experiment
1. change the dataset by user_type
2. change eps, \rho, |T|, m by epsilon, v, p, d (use ipums500k for p)
3. ord1 for domain size 64, ord1h for 1024
4. ord2 for 64*1024, ord2l for 256*256
5. q11 added to the end indicates query types
6. selectivity is controled automatically controlled by fixing the query in server.translate
7. some data points does not make sense: 7(a) 7(d) 0.8, 7(b) 7(e) 2, maybe check code and run again