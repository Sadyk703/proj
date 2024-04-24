List of files in the General folder:

Matlab functions:

psout.m - output function for particle swarm optimization (PSO). Extracts the swarm every 200 iterations as well as the final one. The extracted swarm provides starting points for the multistart algorithm.

cols.m - returns number of columns in a matrix.

rows.m - returns number of rows in a matrix.

Matlab scripts:

GA_optim.m - a script file that performs Genetic Algorithm optimization, followed by the Multistart using points from the final population as initial values. 

PSO_optim.m - a script file that performs PSO optimization, followed by the Multistart using points from the final swarm as initial values. 

Folders:

csminwel - contains the csminwel minimization algorithm by Sims.

gensys - contains the gensys solution algorithm by Sims (2002).

gensys_mod - contains the gensys algorithm modified to allow for indeterminacy. The modification follows Lubik and Schorfheide (2003). 
