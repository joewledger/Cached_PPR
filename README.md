##Indexed PPR

This repository contains the code for investigating the effect of using indexed proximity vectors for speeding the computation of Personalized PageRank (PPR) queries as well as top-K query results.

[PPR is an algorithm for computing the network proximity](http://ilpubs.stanford.edu:8090/596/1/2003-35.pdf) of all nodes in a network to a set of query nodes provided by the user.

Although there are several ways of computing the results of PPR queries, the most commonly used is the Power Iteration method. In the Power Iteration method, the network in question is represented as a column normalized matrix W of dimension (|V|, |V|) (where nodes are represented by the indices of the rows and columns of the matrix, and edges weights are represented by entries in the matrix). A "restart vector" of dimensions (|V|,1) is denoted as r<sub>q</sub>, where each query node q in the set of query nodes Q provided by the user gets an entry 1 / |Q| at location q in the vector. In the standard implementation of PPR, a "start vector" x<sub>q</sub><sup>0</sup> that is identical to the restart vector is used as a starting point for the computation. In addition, the user provides a value for alpha (unfortunately github markdown does not support greek letters) that is the "restart probability" of the calculation. It affects how "global" our computation is, or in other words how detailed we want our computation to be. Lower values of alpha correspond to more global computations.

Now that we have the initizations of all the variables, we can use the following iterative formula to compute the final node scores.

<p align="center">
	<img src="https://cloud.githubusercontent.com/assets/6250320/18111886/1d3c9974-6ef1-11e6-991b-2900f00fa161.PNG"/>
</p>

We continue to compute x<sub>q</sub><sup>t+1</sup> until the L1-distance between x<sub>q</sub><sup>t+1</sup> and x<sub>q</sub><sup>t</sup> is less than a threshold epsilon (usually set at 1E-10). At this point we say that x<sub>q</sub><sup>t+1</sup> has converged to x<sub>q</sub><sup>*</sup>, and we return this vector as the final scores for every node in the network.

Our contribution is speeding the PPR query computation by using indexed proximity vectors. Essentially we move some of the computation of the PPR calculation to a pre-computation step, so when a user submits a query it is proceesed more quickly.

In the pre-computation (or indexing step), we calculate partial PPR query results for every node in the network. When a user submits a query set Q, we build a start vector from the partial proximity vectors we stored. The goal is to build a start vector that is closer to the final converged vector than the standard start vector would be.

The results of this investigation have been quite positive. We have applied our method to the standard PPR implementation as well as a novel method for computing PPR queries, [CHOPPER](http://www.kdd.org/kdd2016/papers/files/rpp0347-coskunA.pdf). A selection of results are shown below.

<p align="center">
	<img src="https://cloud.githubusercontent.com/assets/6250320/18112484/eb9386ee-6ef5-11e6-8550-9a0d28188acb.png" />
</p>

<p align="center">
	<img src="https://cloud.githubusercontent.com/assets/6250320/18112491/fb1239a8-6ef5-11e6-9f6e-22efc6eb38fe.png" />
</p>

We are currently working to apply our indexing method to several PPR top-k query algorithms. These include SQUEEZE and the top-k version of CHOPPER. 